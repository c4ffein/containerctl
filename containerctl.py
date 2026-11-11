#!/usr/bin/env python3
"""
containerctl - KISS containers CLI/TUI manager
MIT License - Copyright (c) 2026 c4ffein
WARNING: I don't recommend using this as-is. This is a PoC, and usable by me because I know what I want to do with it.
- You can use it if you feel that you can edit the code yourself and you can live with my future breaking changes.
"""

import argparse
import atexit
import collections
import dataclasses
import datetime
import difflib
import hashlib
import json
import os
import select
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import termios
import textwrap
import threading
import time
import tomllib
import tty
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Heuristics: guesses, kept here where they are easy to find and change.
# Seconds a container may stay under its spawning name (see SPAWNING_PREFIX) before the Issues tab warns about it.
# A spawn is rarely that fast, a dead one is dead at once: past this it is "maybe slow, maybe stuck", never a verdict.
SPAWN_WARN_AFTER = 10

# =============================================================================
# Errors (a log) and issues (detected from current state)
# =============================================================================
#
# Two kinds, two tabs:
#   - an *issue* is something that is true right now and wants an action from you: docker is unreachable, a
#     listing failed, a config file doesn't parse (error level: something is broken), a shortcut two projects
#     define, a world-readable secrets file, too many local image versions (warning level: works, but look at
#     it). Detected from state already in hand on every fetch, never stored, gone by itself once fixed. Nothing
#     to clear: its `hint` is the action. Whatever is wrong tick after tick is an issue, however briefly: a
#     daemon that drops for ten seconds is one, for ten seconds.
#   - an *error* is the trace of something that happened once: an action or a build that failed, what a CLI run
#     ran into, and whatever we don't handle yet (a crash of ours), which becomes an issue the day we do. Logged
#     through log_error into a ring of the newest EVENTS_MAX, which the Errors tab lists newest first: a log,
#     nothing more. The fetcher doesn't write to it what it can show as an issue. A CLI run prints what it logged
#     on stderr when it is done.
# The same fact reaches either: the calls that can fail hand the outcome back (`read_projects`, a `Listing`), the
# fetcher turns it into issues, a one-shot caller logs it (`load_projects`, `Listing.logged`).

WARNING, ERROR = "warning", "error"


@dataclass(frozen=True, slots=True)
class Event:
    ts: float
    cat: str
    msg: str
    ctx: dict
    echoed: bool = False  # already said on stderr when it happened: see log_error

    @property
    def time(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.ts))


@dataclass(frozen=True)
class Issue:
    level: str  # WARNING or ERROR
    cat: str
    msg: str
    hint: str = ""  # what makes it go away
    detail: str = ""  # the raw material, one item per line: what docker said, the rows we couldn't read


def by_severity(issues: list[Issue]) -> list[Issue]:
    """Errors first, then warnings, each by category"""
    return sorted(issues, key=lambda i: (i.level != ERROR, i.cat))


class ConfigError(Exception):
    """A config file that can't be used. Raised by the parsers, caught per file by read_configs: `cat` is the
    category it is reported under, the message says what is wrong and where."""

    def __init__(self, cat: str, msg: str):
        super().__init__(msg)
        self.cat = cat


EVENTS_MAX = 10_000  # the oldest go, the newest stay
_errors: collections.deque[Event] = collections.deque(maxlen=EVENTS_MAX)  # oldest first
# The fetcher thread logs while the main thread lists. The GIL makes a lone append safe; a snapshot racing an
# append that evicts is not, and free-threaded builds have no GIL, so every writer and every walker takes this
# lock. Callers never do: log_error and friends handle it. (`bool(_errors)` is one len read, no lock needed.)
_errors_lock = threading.Lock()


def log_error(category: str, message: str, context: dict | None = None, echo: bool = False) -> None:
    """Record a non-fatal error. `echo` also says it on stderr right now, for a flow that has the terminal and a
    user waiting on it (a spawn): it stays in the log (the TUI's Errors tab), and main's closing summary of a
    CLI run leaves it out, as already said."""
    with _errors_lock:
        _errors.append(Event(time.time(), category, message, context or {}, echoed=echo))
    if echo:
        print(f"Error: {message}", file=sys.stderr)


def get_errors() -> list[Event]:
    """A snapshot of the log, oldest first"""
    with _errors_lock:
        return list(_errors)


def clear_errors() -> int:
    """Empty the log; returns how many entries went"""
    with _errors_lock:
        count = len(_errors)
        _errors.clear()
        return count


def log_issues(issues: list[Issue]) -> None:
    """For a one-shot caller with no Issues tab to show them in: what was detected goes to the log"""
    for issue in issues:
        log_error(issue.cat, issue.msg)


# =============================================================================
# Docker Data Models
# =============================================================================

VALID_ID_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
VALID_IMAGE_REF_CHARS = VALID_ID_CHARS + "/:.@"  # repo/name:tag, registry.domain, name@sha256:<digest>
VALID_PATH_CHARS = VALID_ID_CHARS + "/:.~+ "  # Paths can have slashes, dots, tildes, etc.
VALID_REPO_URL_CHARS = VALID_ID_CHARS + "/:@."  # Git URLs: git@github.com:user/repo.git or https://...
VALID_PORT_PROTOCOLS = {"tcp", "udp"}


def is_valid_id(id_str: str) -> bool:
    """A name we hand to docker (container, volume, network, user, ...): a-z A-Z 0-9 - _ and, as docker allows
    them in names, dots. Never empty, and never starting with `-` (docker would read an option) or `.`"""
    return bool(id_str) and id_str[0] not in "-." and all(c in VALID_ID_CHARS or c == "." for c in id_str)


def is_valid_image_ref(ref_str: str) -> bool:
    """Check if image reference contains only valid characters (allows /, :, . for repo:tag); like is_valid_id,
    never empty and never starting with `-` or `.`"""
    return bool(ref_str) and ref_str[0] not in "-." and all(c in VALID_IMAGE_REF_CHARS for c in ref_str)


def has_path_traversal(path_str: str) -> bool:
    """Check if a path contains '..' components"""
    return ".." in path_str.split("/")


def is_valid_home(path_str: str) -> bool:
    """A project's `home`: an absolute path inside the container, safe as an argv word, a `-w` and a path"""
    return path_str.startswith("/") and path_str.rstrip("/") != "" and is_valid_path(path_str)


def is_valid_path(path_str: str) -> bool:
    """Check if path contains only valid characters and no traversal"""
    return all(c in VALID_PATH_CHARS for c in path_str) and not has_path_traversal(path_str)


def is_valid_repo_url(url_str: str) -> bool:
    """Check if repo URL contains only valid characters and no traversal"""
    return all(c in VALID_REPO_URL_CHARS for c in url_str) and not has_path_traversal(url_str)


def _is_uint(text: str, upper: int) -> bool:
    """ASCII decimal digits only (str.isdigit also takes unicode digits), within [0, upper]"""
    return text.isascii() and text.isdigit() and int(text) <= upper


def _is_port_range(text: str) -> bool:
    """`80` or `4000-4010`"""
    parts = text.split("-")
    return len(parts) in (1, 2) and all(_is_uint(p, 65535) for p in parts)


def _is_ipv4(text: str) -> bool:
    parts = text.split(".")
    return len(parts) == 4 and all(_is_uint(p, 255) for p in parts)


def is_valid_port_spec(spec_str: str) -> bool:
    """Check a `-p` publish spec: [host_ip:]host_port[-host_port]:container_port[-container_port][/tcp|udp]

    Also accepts just the container port (docker then picks a free host port). Host ip is IPv4 only, KISS."""
    body, _, proto = spec_str.rpartition("/")
    if not body:  # no "/": rpartition put the whole string in `proto`
        body, proto = proto, "tcp"
    if proto not in VALID_PORT_PROTOCOLS:
        return False
    parts = body.split(":")
    if len(parts) == 3:
        if not _is_ipv4(parts[0]):
            return False
        parts = parts[1:]
    return len(parts) in (1, 2) and all(_is_port_range(p) for p in parts)


def deserialize(cls, data):
    """Decoded JSON -> an instance of dataclass `cls`, checked against the class itself: its fields are the schema.
    A field without a default is required; a value must be an instance of its field's type (isinstance takes
    `str | None` and friends as they are); keys the class doesn't declare are dropped, docker adds columns over
    time. ValueError saying what is wrong. Shallow on purpose: a `list` or `dict` field isn't looked into.
    (Needs real annotations: under `from __future__ import annotations` a field's type is a string.)"""
    if not isinstance(data, dict):
        raise ValueError(f"Not an object: {type(data).__name__}")
    fields = dataclasses.fields(cls)
    unset = dataclasses.MISSING
    missing = [f.name for f in fields if f.name not in data and f.default is unset and f.default_factory is unset]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    given = {f.name: data[f.name] for f in fields if f.name in data}
    types = {f.name: f.type for f in fields}
    wrong = [f"{name} ({type(value).__name__})" for name, value in given.items() if not isinstance(value, types[name])]
    if wrong:
        raise ValueError(f"Wrong types: {', '.join(wrong)}")
    return cls(**given)


class Row:
    """A row of a docker listing: a dataclass, deserialized as one, whose KEY field (what we hand back to docker
    to name it) must be a safe identifier"""

    KEY = "ID"  # not annotated: class attributes, not fields
    VALID = staticmethod(is_valid_id)

    @classmethod
    def from_dict(cls, data):
        """A row of the listing; ValueError when it is not one we can use"""
        row = deserialize(cls, data)
        if not cls.VALID(getattr(row, cls.KEY)):
            raise ValueError(f"Invalid {cls.KEY}: {getattr(row, cls.KEY)}")
        return row


@dataclass
class Container(Row):
    """Docker container data model"""

    ID: str
    Names: str
    Image: str
    Status: str
    State: str = ""
    Command: str = ""
    CreatedAt: str = ""
    Ports: str = ""
    Networks: str = ""
    Mounts: str = ""
    Labels: str = ""
    Size: str = ""
    RunningFor: str = ""
    LocalVolumes: str = ""
    Platform: dict | str | None = None  # an object on recent docker, unused here

    @property
    def Project(self) -> str:
        """The project this container belongs to, read off its cctl- name; empty for any other container"""
        return self.Names.removeprefix(CONTAINER_PREFIX) if self.Names.startswith(CONTAINER_PREFIX) else ""


@dataclass
class Image(Row):
    """Docker image data model"""

    ID: str
    Repository: str
    Tag: str
    Size: str
    CreatedAt: str = ""
    CreatedSince: str = ""
    Digest: str = ""
    Containers: str = ""
    SharedSize: str = ""
    UniqueSize: str = ""
    VirtualSize: str = ""


@dataclass
class Volume(Row):
    """Docker volume data model"""

    KEY = "Name"

    Name: str
    Driver: str
    Mountpoint: str = ""
    Scope: str = ""
    Labels: str = ""
    Size: str = ""
    Status: str = ""
    Availability: str = ""
    Group: str = ""
    Links: str = ""


@dataclass
class Network(Row):
    """Docker network data model"""

    ID: str
    Name: str
    Driver: str
    Scope: str = ""
    CreatedAt: str = ""
    Internal: str = ""
    IPv4: str = ""
    IPv6: str = ""
    Labels: str = ""


# A docker call that only asks a question (a listing, an inspect) gets a few seconds to answer: a daemon that hangs
# then reads as a failed call, which every caller already reports, instead of freezing whoever asked. Three budgets:
# - QUERY_TIMEOUT: the default, for a question somebody asked (a CLI command, a key in the TUI). These answer in
#   well under a second on a daemon that is fine; the margin is for a loaded host.
# - POLL_TIMEOUT: the TUI's background fetch, every 2s. Short: a listing that missed its turn is asked again on the
#   next one. A listing slower than this shows as a failed one, its tab empty until one makes it. That should not
#   happen (they answer in milliseconds), and is not a flaw to design around here: if it does, will fix then.
# - SIZE_TIMEOUT: `docker ps` with sizes walks every container's filesystem: over ten seconds is its normal.
# Calls that do something (run, pull, build, stop, clone over exec) take the time they take: no timeout there,
# Ctrl-C is the way out (see main, App.handle_key).
QUERY_TIMEOUT = 6
POLL_TIMEOUT = 1
SIZE_TIMEOUT = 60
TIMED_OUT = 124  # exit code _run gives a call that timed out (as timeout(1) does)
NO_ANSWER = "containerctl: no answer after"  # what _run puts on stderr for a call that timed out


# What a failed docker call says on stderr (lowercased substring) -> what we say, and what to do about it. First
# match wins: the socket's "permission denied" also mentions connecting to the daemon. Anything else shows as
# UNKNOWN_DOCKER_FAILURE with the raw stderr in the issue's details: read it there, then add its line here.
DOCKER_FAILURES = [
    ("command not found", "docker is not installed", "install docker, or fix PATH"),
    ("permission denied", "No permission on the docker socket", "add yourself to the docker group"),
    ("cannot connect to the docker daemon", "Docker daemon is not running", "start docker"),
    ("is not running", "Container is not running", "start it, then try again"),
    (NO_ANSWER, "Docker did not answer in time", "is the daemon hung? look at `systemctl status docker`"),
]
UNKNOWN_DOCKER_FAILURE = ("Unknown docker failure", "its details have what docker said")
# `docker exec` also hands over the stderr of what it ran, and an `ls: Permission denied` in there is not the
# socket's. What docker itself says (client or daemon; lowercased substrings), when the call is an exec: anything
# else is the command's own, and docker_failure says it as that (see `inside`).
DOCKER_VOICE = ("error response from daemon", "docker daemon", "docker: command not found", NO_ANSWER)
# How docker says "there is no such thing" (lowercased substrings: "No such image: x", "network x not found"), the
# one failure of a "does it exist?" call that is an answer: see Docker._absent. A wording missing here is not read
# as "absent" but as an unknown failure, raw stderr in its details: add it then.
NO_SUCH = ("no such", "not found")


@dataclass(frozen=True)
class Failure:
    """Why a docker call failed: `msg` and `hint` in our words, `detail` the command and raw stderr for the log"""

    msg: str
    hint: str
    detail: str


def docker_failure(args: list[str], result: subprocess.CompletedProcess, inside: str = "") -> Failure:
    """Our words for a failed docker call. `inside` names the command an exec ran in the container: unless the
    stderr is docker's own voice, the failure is that command's, and is said as such."""
    stderr = (result.stderr or "").strip()
    detail = f"docker {' '.join(args)}" + (f": {stderr}" if stderr else f": exit code {result.returncode}")
    text = stderr.lower()
    if inside and not any(voice.lower() in text for voice in DOCKER_VOICE):
        said = stderr.splitlines()[0] if stderr else f"exit code {result.returncode}"
        return Failure(f"`{inside}` failed in the container: {said}", "its details have all it said", detail)
    msg, hint = next(((m, h) for needle, m, h in DOCKER_FAILURES if needle in text), UNKNOWN_DOCKER_FAILURE)
    return Failure(msg, hint, detail)


class DockerError(Exception):
    """A docker call failed where there is no honest way to go on: reading a failed `docker ps` as "no such
    container" would have `check` say there is nothing to check and `up` create what already exists. Raised by
    the lookups, caught at the edges (main, the TUI's key handler), which say why and stop."""

    def __init__(self, failure: Failure):
        super().__init__(failure.msg)
        self.failure = failure


@dataclass
class Listing:
    """What a docker listing came back with: its rows, or why there are none. An empty list and a failed one are
    not the same thing, and only the caller knows what a failure should become (an issue, a log line)."""

    rows: list = field(default_factory=list)
    failure: Failure | None = None
    skipped: list[str] = field(default_factory=list)  # the lines that weren't a usable row, and why: the rest counts

    def logged(self) -> list:
        """The rows, for a one-shot caller: a failure is logged and reads as no rows, skipped lines are logged"""
        if self.failure:
            log_error("docker.command", self.failure.detail)
        for line in self.skipped:
            log_error("docker.row_parse", line)
        return self.rows

    def required(self) -> list:
        """The rows, for a caller that can't honestly go on without them: DockerError on a failure (skipped lines
        are logged)"""
        if self.failure:
            raise DockerError(self.failure)
        return self.logged()


LISTINGS = ("containers", "images", "volumes", "networks")  # the four docker listings, by tab key


# =============================================================================
# Docker Inspect Data Models
# =============================================================================


class Inspected(Row):
    """What `docker inspect` says of one object: a Row (a dataclass whose KEY field must pass VALID), of which
    there is one, not a listing. So one we can't use is not skipped but None, after logging under CAT.
    Docker.inspect fetches one: `Docker.inspect(NetworkInspect, name)`."""

    KEY = "Id"
    TYPE = ""  # what `docker inspect --type` calls it
    CAT = ""

    @classmethod
    def from_dict(cls, data):
        try:
            return super().from_dict(data)
        except ValueError as e:
            log_error(cls.CAT, str(e), {"data": str(data)[:200]})
            return None


@dataclass
class ContainerInspect(Inspected):
    """Docker container inspect data model (from docker inspect)"""

    TYPE = "container"
    CAT = "docker.container_inspect_parse"

    Id: str
    Name: str
    Created: str
    State: dict
    Image: str
    Config: dict
    Path: str = ""
    Args: list | None = field(default_factory=list)
    ResolvConfPath: str = ""
    HostnamePath: str = ""
    HostsPath: str = ""
    LogPath: str = ""
    RestartCount: int = 0
    Driver: str = ""
    Platform: str = ""
    MountLabel: str = ""
    ProcessLabel: str = ""
    AppArmorProfile: str = ""
    ExecIDs: list | None = None
    HostConfig: dict | None = field(default_factory=dict)
    GraphDriver: dict | None = field(default_factory=dict)
    Mounts: list | None = field(default_factory=list)
    NetworkSettings: dict | None = field(default_factory=dict)

    @property
    def labels(self) -> dict:
        """Get container labels from Config"""
        return self.Config.get("Labels") or {}

    @property
    def name_clean(self) -> str:
        """Get container name without leading slash"""
        return self.Name.lstrip("/")


@dataclass
class ImageInspect(Inspected):
    """Docker image inspect data model (from docker image inspect)"""

    TYPE = "image"
    CAT = "docker.image_inspect_parse"
    VALID = staticmethod(is_valid_image_ref)  # sha256:<digest>

    Id: str
    Created: str
    Architecture: str
    Os: str
    Size: int
    RepoTags: list | None = field(default_factory=list)
    RepoDigests: list | None = field(default_factory=list)
    Parent: str = ""
    Comment: str = ""
    DockerVersion: str = ""
    Author: str = ""
    Config: dict | None = field(default_factory=dict)
    GraphDriver: dict | None = field(default_factory=dict)
    RootFS: dict | None = field(default_factory=dict)
    Metadata: dict | None = field(default_factory=dict)


@dataclass
class VolumeInspect(Inspected):
    """Docker volume inspect data model (from docker volume inspect)"""

    KEY = "Name"
    TYPE = "volume"
    CAT = "docker.volume_inspect_parse"

    Name: str
    Driver: str
    Mountpoint: str
    Scope: str
    CreatedAt: str = ""
    Labels: dict | None = None
    Options: dict | None = None


@dataclass
class NetworkInspect(Inspected):
    """Docker network inspect data model (from docker network inspect)"""

    TYPE = "network"
    CAT = "docker.network_inspect_parse"

    Id: str
    Name: str
    Created: str
    Scope: str
    Driver: str
    EnableIPv4: bool = True
    EnableIPv6: bool = False
    IPAM: dict | None = field(default_factory=dict)
    Internal: bool = False
    Attachable: bool = False
    Ingress: bool = False
    ConfigFrom: dict | None = field(default_factory=dict)
    ConfigOnly: bool = False
    Containers: dict | None = field(default_factory=dict)
    Options: dict | None = field(default_factory=dict)
    Labels: dict | None = field(default_factory=dict)


# =============================================================================
# Project Configuration
# =============================================================================

CONFIG_DIR = Path.home() / ".config" / "containerctl"
PROJECTS_DIR = CONFIG_DIR / "projects"
SERVICES_DIR = CONFIG_DIR / "services"
IMAGES_DIR = CONFIG_DIR / "images"  # <name>.toml + <name>.Dockerfile [+ <name>/ build context]
SECRETS_FILE = CONFIG_DIR / "secrets.toml"  # flat `key = "value"`, 0600, never versioned; [secrets] picks keys from it

# Who a spawned container is for, and that user's home *inside* it, unless the project says otherwise (`user`,
# `home`: see ProjectConfig). Everything spawn puts in goes under that home, owned by that user.
# Nothing we run in a project container is left to the image's default `USER`: it runs as the project's user (the
# clones, the checks, the `enter` shell), or as ROOT for the one thing that takes root: handing over what
# `docker cp` wrote (it always writes as root). So an image whose USER is root, or anyone, works the same.
CONTAINER_USER = "dev"
ROOT = "root"
CONTAINER_HOME = "/home/dev"

CONTAINER_PREFIX = "cctl-prj-"
# A project container is made under this prefix and only renamed to cctl-prj-<name> once everything is in it
# (key, clones, env files). The name is the state: whatever ended a spawn halfway (Ctrl-C, a closed terminal, a
# crash, the power), what it left is not a project container, so `enter` never attaches to half a workspace. It
# was never entered either, so it holds nothing of yours: the next spawn asks, removes it and starts over.
SPAWNING_PREFIX = "cctl-spawning-"
SERVICE_PREFIX = "cctl-svc-"
NETWORK_PREFIX = "cctl-net-prj-"  # one per project; mirrors CONTAINER_PREFIX
LABEL_PROJECT = "cctl.project"
LABEL_REPO = "cctl.repo"
LABEL_SERVICE = "cctl.service"
LABEL_IMAGE = "cctl.image"  # on images built from a recipe
LABEL_RECIPE = "cctl.recipe"  # sha256 of the recipe (Dockerfile + context) the image was built from
LOCAL_VERSION_PREFIX = "cctl-local-"  # each build is <repo>:cctl-local-NNNNNN; the recipe tag is re-pointed at it
IMAGE_VERSIONS_WARN = 3  # `image ls` nags past this many local versions of one image

# Host keys of the forges a spawn's SSH key talks to, as each publishes them out of band (checked 2026-11-08):
#   github.com  https://api.github.com/meta  ("ssh_keys")
#   gitlab.com  https://gitlab.com/help/instance_configuration
# Appended to the dev user's known_hosts at spawn in place of `ssh-keyscan`, which trusts whatever answers the first
# connection. ssh then enforces them on every connection: a rotated or spoofed key is a loud MITM warning, never a
# silent accept. A real rotation means updating these lines and re-spawning.
KNOWN_HOSTS = (
    "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl\n"
    "gitlab.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAfuCHKVTjquxvt6CM6tdG4SLp1Btn/nOeHHE5UOzRdf\n"
)


@dataclass
class EnvFile:
    """Environment file mapping"""

    src: str
    dest: str


# Git branch/ref names may contain slashes (e.g. feature/foo), repo dir names may not
VALID_REPO_NAME_CHARS = VALID_ID_CHARS + "-._"
VALID_BRANCH_CHARS = VALID_REPO_NAME_CHARS + "/"
VALID_REL_DIR_CHARS = VALID_REPO_NAME_CHARS + "/"  # shortcut targets: dirs relative to the workspace


def is_valid_rel_dir(dir_str: str) -> bool:
    """A relative directory under the workspace: no leading slash, no '..', plain charset"""
    return (
        bool(dir_str)
        and not dir_str.startswith("/")
        and all(c in VALID_REL_DIR_CHARS for c in dir_str)
        and not has_path_traversal(dir_str)
    )


VALID_ENV_NAME_CHARS = VALID_ID_CHARS.replace("-", "")  # [A-Za-z_][A-Za-z0-9_]*: what every shell accepts


def is_valid_env_name(name: str) -> bool:
    """Shell-safe environment variable name"""
    return bool(name) and not name[0].isdigit() and all(c in VALID_ENV_NAME_CHARS for c in name)


def _parse_name_table(raw, path: Path, table: str, value_ok: Callable[[str], bool]) -> dict[str, str]:
    """A `[table]` of `ENV_NAME = "string"`. ConfigError when anything is malformed: a half-applied environment
    is worse than none."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError("project.config", f"Invalid '{table}' (must be a table) in {path}")
    for name, value in raw.items():
        if not is_valid_env_name(name):
            raise ConfigError("project.config", f"Invalid env name '{name}' in [{table}] of {path}")
        if not isinstance(value, str) or not value_ok(value):
            raise ConfigError("project.config", f"Invalid value for '{name}' in [{table}] of {path}")
    return dict(raw)


def parse_env_table(raw, path: Path) -> dict[str, str]:
    """`[env]`: `NAME = "value"`, plain values for the `enter` shell's environment"""
    return _parse_name_table(raw, path, "env", lambda v: "\x00" not in v)


def parse_secrets_table(raw, path: Path) -> dict[str, str]:
    """`[secrets]`: `NAME = "key"`, the key of a secrets.toml entry whose value is looked up at `enter`.
    Only names and keys live here, so a value can never land in a versioned project file."""
    return _parse_name_table(raw, path, "secrets", is_valid_env_name)


@dataclass
class RepoSpec:
    """A single git repo to clone into a project's workspace"""

    url: str
    dir: str = ""  # subdirectory under the workspace; derived from url when empty
    branch: str = ""

    @property
    def name(self) -> str:
        """Directory name the repo clones into"""
        if self.dir:
            return self.dir
        return self.url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")


@dataclass
class ProjectConfig:
    """Project configuration loaded from TOML"""

    name: str
    image: str
    repos: list[RepoSpec] = field(default_factory=list)
    ssh_key: str = ""
    env_files: list[EnvFile] = field(default_factory=list)
    shortcuts: dict[str, str] = field(default_factory=dict)  # alias -> dir relative to the workspace
    env: dict[str, str] = field(default_factory=dict)  # [env]: plain values for the `enter` shell's environment
    secrets: dict[str, str] = field(default_factory=dict)  # [secrets]: env name -> key in secrets.toml
    user: str = CONTAINER_USER  # the image's user: owns what spawn puts in, runs the clones, checks and shell
    home: str = CONTAINER_HOME  # that user's home in the image: /home/<user> (/root for root) unless said

    @property
    def workspace(self) -> str:
        """Where the repos are cloned, inside the container"""
        return f"{self.home}/workspace"

    @property
    def container_name(self) -> str:
        return f"{CONTAINER_PREFIX}{self.name}"

    @property
    def spawning_name(self) -> str:
        """Its container's name for as long as the spawn is not done: see SPAWNING_PREFIX"""
        return f"{SPAWNING_PREFIX}{self.name}"

    @property
    def repos_display(self) -> str:
        """Short summary of repos for list views"""
        if not self.repos:
            return "-"
        if len(self.repos) == 1:
            return self.repos[0].url
        return f"{len(self.repos)} repos"

    @classmethod
    def from_toml(cls, path: Path) -> "ProjectConfig":
        """Parse a project config from a TOML file; ConfigError when it can't be used"""
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as e:
            raise ConfigError("project.config", f"Failed to load {path}: {e}") from None

        name = path.stem
        if not is_valid_id(name):
            raise ConfigError(
                "project.config", f"Invalid project name '{name}': must be a-zA-Z0-9-_. (not starting with - or .)"
            )

        image = data.get("image", "")
        if not image:
            raise ConfigError("project.config", f"Missing 'image' in {path}")
        if not is_valid_image_ref(image):
            raise ConfigError("project.config", f"Invalid image ref '{image}' in {path}")

        # Accept legacy single `repo = "url"` OR `repos = [...]`, but not both —
        # defining both is ambiguous (which order? is it a mistake?), so reject it.
        # Entries in `repos` may be plain URL strings or tables with url/dir/branch.
        if data.get("repo") and data.get("repos"):
            raise ConfigError("project.config", f"Define either 'repo' or 'repos', not both, in {path}")
        raw_repos: list = []
        if data.get("repo"):
            raw_repos.append(data["repo"])
        raw_repos.extend(data.get("repos", []))

        repos: list[RepoSpec] = []
        seen_dirs: set[str] = set()
        for entry in raw_repos:
            if isinstance(entry, str):
                url, rdir, branch = entry, "", ""
            elif isinstance(entry, dict):
                url = entry.get("url", "")
                rdir = entry.get("dir", "")
                branch = entry.get("branch", "")
            else:
                raise ConfigError("project.config", f"Invalid repo entry '{entry}' in {path}")
            if not url or not is_valid_repo_url(url):
                raise ConfigError("project.config", f"Invalid repo URL '{url}' in {path}")
            if rdir and (not all(c in VALID_REPO_NAME_CHARS for c in rdir) or rdir in (".", "..")):
                raise ConfigError("project.config", f"Invalid repo dir '{rdir}' in {path}")
            if branch and not all(c in VALID_BRANCH_CHARS for c in branch):
                raise ConfigError("project.config", f"Invalid repo branch '{branch}' in {path}")
            spec = RepoSpec(url=url, dir=rdir, branch=branch)
            if spec.name in seen_dirs:
                raise ConfigError("project.config", f"Duplicate repo dir '{spec.name}' in {path}")
            seen_dirs.add(spec.name)
            repos.append(spec)

        ssh_key = ""
        if data.get("ssh_key"):
            ssh_key = str(Path(data["ssh_key"]).expanduser())
            if not is_valid_path(ssh_key):
                raise ConfigError("project.config", f"Invalid ssh_key path '{ssh_key}' in {path}")

        env_files = []
        for ef in data.get("env_files", []):
            src = ef.get("src", "")
            dest = ef.get("dest", "")
            if src and dest:
                src_expanded = str(Path(src).expanduser())
                if not is_valid_path(src_expanded):
                    raise ConfigError("project.config", f"Invalid env_file src path '{src}' in {path}")
                if not is_valid_path(dest) or has_path_traversal(dest):
                    raise ConfigError("project.config", f"Invalid env_file dest path '{dest}' in {path}")
                env_files.append(EnvFile(src=src_expanded, dest=dest))

        # [shortcuts]: `containerctl enter <alias>` lands in <workspace>/<dir> of this project
        shortcuts: dict[str, str] = {}
        raw_shortcuts = data.get("shortcuts", {})
        if not isinstance(raw_shortcuts, dict):
            raise ConfigError("project.config", f"Invalid 'shortcuts' (must be a table) in {path}")
        for alias, rel_dir in raw_shortcuts.items():
            if not alias or not is_valid_id(alias):
                raise ConfigError("project.config", f"Invalid shortcut name '{alias}' in {path}")
            if not isinstance(rel_dir, str) or not is_valid_rel_dir(rel_dir):
                raise ConfigError("project.config", f"Invalid shortcut dir '{rel_dir}' for '{alias}' in {path}")
            shortcuts[alias] = rel_dir

        # [env] plain values and [secrets] env name -> secrets.toml key: both reach the `enter` shell's environment
        env = parse_env_table(data.get("env"), path)
        secrets = parse_secrets_table(data.get("secrets"), path)
        if both := sorted(set(env) & set(secrets)):
            raise ConfigError("project.config", f"Names in both [env] and [secrets] in {path}: {', '.join(both)}")

        # Who the image is built for, whatever its default USER: all we do in it is done as this user (see
        # CONTAINER_USER). To enter as someone else once: `enter <project> user=<user>`. Spawn fails on an image
        # that doesn't have this user and this home.
        user = data.get("user", CONTAINER_USER)
        if not isinstance(user, str) or not is_valid_id(user):
            raise ConfigError("project.config", f"Invalid user '{user}' in {path}")
        home = data.get("home", "/root" if user == "root" else f"/home/{user}")
        if not isinstance(home, str) or not is_valid_home(home):
            raise ConfigError("project.config", f"Invalid home '{home}' (an absolute path) in {path}")

        return cls(
            name=name,
            image=image,
            repos=repos,
            ssh_key=ssh_key,
            env_files=env_files,
            shortcuts=shortcuts,
            env=env,
            secrets=secrets,
            user=user,
            home=home.rstrip("/"),
        )


def read_configs(directory: Path, parse: Callable[[Path], object]) -> tuple[dict, list[Issue]]:
    """Every <name>.toml of a config dir, parsed: the ones that are usable by name, and an error-level issue for
    each one that isn't. One broken file never hides the others."""
    configs: dict = {}
    problems: list[Issue] = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.toml")):
            try:
                config = parse(path)
            except ConfigError as e:
                problems.append(Issue(ERROR, e.cat, str(e), f"edit {path.name}"))
            else:
                configs[config.name] = config
    return configs, problems


def read_projects() -> tuple[dict[str, ProjectConfig], list[Issue]]:
    """~/.config/containerctl/projects/: the projects, and what is wrong with the files that aren't one.
    (Shortcut collisions are an issue of the loaded set: see shortcut_issues.)"""
    return read_configs(PROJECTS_DIR, ProjectConfig.from_toml)


def load_projects() -> dict[str, ProjectConfig]:
    """The projects, for a one-shot caller: broken files are logged"""
    projects, problems = read_projects()
    log_issues(problems)
    return projects


def resolve_enter_target(name: str, projects: dict[str, ProjectConfig]) -> tuple[ProjectConfig, str | None] | None:
    """What `enter <name>` means: a project (land in the default dir), else a shortcut that
    exactly one project declares (land in that dir). None when unknown or ambiguous."""
    if name in projects:
        return projects[name], None
    hits = [(p, rel) for p in projects.values() for alias, rel in p.shortcuts.items() if alias == name]
    if len(hits) != 1:
        return None
    project, rel_dir = hits[0]
    return project, f"{project.workspace}/{rel_dir}"


# =============================================================================
# Secrets: ~/.config/containerctl/secrets.toml
# =============================================================================


def load_secrets() -> dict[str, str]:
    """secrets.toml: flat `key = "value"`. Values are never logged; a malformed entry is skipped, a malformed
    file yields nothing. A file readable by others is reported: a token is only as private as the file's mode."""
    if not SECRETS_FILE.is_file():
        return {}
    try:
        with open(SECRETS_FILE, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        log_error("env.secret", f"Failed to load {SECRETS_FILE}: {e}")
        return {}
    if issue := secrets_mode_issue():  # a standing issue too; echoed here so `enter` shows it when it matters
        log_error(issue.cat, f"{issue.msg}: {issue.hint}")
    secrets: dict[str, str] = {}
    for key, value in data.items():
        if not is_valid_env_name(key):
            log_error("env.secret", f"Invalid key '{key}' in {SECRETS_FILE}")
            continue
        if not isinstance(value, str) or not value or "\x00" in value:
            log_error("env.secret", f"Invalid value for '{key}' in {SECRETS_FILE}: must be a non-empty string")
            continue
        secrets[key] = value
    return secrets


def resolve_env(project: "ProjectConfig") -> dict[str, str]:
    """What `enter` injects: the project's [env], plus each [secrets] entry found in secrets.toml. A missing
    secret is recorded under env.secret and left out; the rest still goes in. secrets.toml is only read when
    the project names a secret, and only the keys it names are used."""
    env = dict(project.env)
    if not project.secrets:
        return env
    secrets = load_secrets()
    for name, key in project.secrets.items():
        if key not in secrets:
            log_error("env.secret", f"{name}: no '{key}' in {SECRETS_FILE}")
            continue
        env[name] = secrets[key]
    return env


VALID_RESTART_POLICIES = {"no", "always", "unless-stopped", "on-failure"}
VALID_VOLUME_SUFFIX = {"ro", "rw"}
VALID_TMPFS_OPTION_CHARS = VALID_ID_CHARS + ",="  # e.g. "rw,noexec,nosuid,size=64m,mode=1777,uid=1000"


def is_valid_tmpfs_spec(spec_str: str) -> bool:
    """`/path` or `/path:opt,opt=value,...` for `--tmpfs`: absolute path, charset-checked options"""
    path, _, options = spec_str.partition(":")
    if not path.startswith("/") or not is_valid_path(path):
        return False
    return not options or all(c in VALID_TMPFS_OPTION_CHARS for c in options)


# What services still want before they run for months on a server (one line each; drop the line when done):
# TODO(services): log limits: `--log-opt max-size/max-file`; json-file grows until the disk is full
# TODO(services): drift: an issue when a running container no longer matches its TOML (label a config hash at run)
# TODO(services): secrets: `env` values sit in argv and `docker inspect`; give services a [secrets] like projects
# TODO(services): limits and health: memory / cpus, healthcheck, `--user`, `--security-opt no-new-privileges`
# TODO(services): updates: say when a pulled image is behind its tag; `recreate` is downtime with no way back


@dataclass
class ServiceConfig:
    """Service configuration loaded from TOML"""

    name: str
    image: str
    restart: str = "unless-stopped"
    ports: list[str] = field(default_factory=list)
    volumes: list[str] = field(default_factory=list)
    scripts: dict[str, list[str]] = field(default_factory=dict)
    env: list[str] = field(default_factory=list)
    command: list[str] = field(default_factory=list)  # argv after the image; empty = the image's own CMD
    networks: list[str] = field(default_factory=list)  # created if missing; `up` connects an existing container
    # Hardening (all optional, off by default): --read-only, --tmpfs, --cap-drop/--cap-add, --pids-limit
    read_only: bool = False
    tmpfs: list[str] = field(default_factory=list)
    cap_drop: list[str] = field(default_factory=list)
    cap_add: list[str] = field(default_factory=list)
    pids_limit: int = 0  # 0 = docker default (unlimited)

    @property
    def container_name(self) -> str:
        return f"{SERVICE_PREFIX}{self.name}"

    @classmethod
    def from_toml(cls, path: Path) -> "ServiceConfig":
        """Parse a service config from a TOML file; ConfigError when it can't be used"""
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as e:
            raise ConfigError("service.config", f"Failed to load {path}: {e}") from None

        name = path.stem
        if not is_valid_id(name):
            raise ConfigError(
                "service.config", f"Invalid service name '{name}': must be a-zA-Z0-9-_. (not starting with - or .)"
            )

        image = data.get("image", "")
        if not image:
            raise ConfigError("service.config", f"Missing 'image' in {path}")
        if not is_valid_image_ref(image):
            raise ConfigError("service.config", f"Invalid image ref '{image}' in {path}")

        restart = data.get("restart", "unless-stopped")
        if restart not in VALID_RESTART_POLICIES:
            raise ConfigError("service.config", f"Invalid restart policy '{restart}' in {path}")

        ports = []
        for p in data.get("ports", []):
            if not isinstance(p, str) or not is_valid_port_spec(p):
                raise ConfigError("service.config", f"Invalid port spec '{p}' in {path}")
            ports.append(p)

        volumes = []
        for v in data.get("volumes", []):
            if not isinstance(v, str):
                raise ConfigError("service.config", f"Invalid volume '{v}' in {path}")
            # Split volume spec: host:container or host:container:mode
            parts = v.split(":")
            if len(parts) < 2 or len(parts) > 3:
                raise ConfigError("service.config", f"Invalid volume spec '{v}' in {path}")
            host_path = str(Path(parts[0]).expanduser())
            if not is_valid_path(host_path):
                raise ConfigError("service.config", f"Invalid volume host path '{parts[0]}' in {path}")
            if len(parts) == 3 and parts[2] not in VALID_VOLUME_SUFFIX:
                raise ConfigError("service.config", f"Invalid volume mode '{parts[2]}' in {path}")
            # Rebuild with expanded host path
            expanded = f"{host_path}:{':'.join(parts[1:])}"
            volumes.append(expanded)

        env_list = []
        for e in data.get("env", []):
            if not isinstance(e, str) or "=" not in e:
                raise ConfigError("service.config", f"Invalid env entry '{e}' in {path}")
            env_list.append(e)

        networks = []
        for n in data.get("networks", []):
            if not isinstance(n, str) or not n or not is_valid_id(n):
                raise ConfigError("service.config", f"Invalid network name '{n}' in {path}")
            networks.append(n)

        read_only = data.get("read_only", False)
        if not isinstance(read_only, bool):
            raise ConfigError("service.config", f"Invalid 'read_only' (must be true/false) in {path}")

        tmpfs = []
        for t in data.get("tmpfs", []):
            if not isinstance(t, str) or not is_valid_tmpfs_spec(t):
                raise ConfigError("service.config", f"Invalid tmpfs spec '{t}' in {path}")
            tmpfs.append(t)

        caps: dict[str, list[str]] = {"cap_drop": [], "cap_add": []}
        for key in caps:
            for c in data.get(key, []):
                if not isinstance(c, str) or not c or not is_valid_id(c):
                    raise ConfigError("service.config", f"Invalid capability '{c}' in '{key}' in {path}")
                caps[key].append(c)

        pids_limit = data.get("pids_limit", 0)
        if isinstance(pids_limit, bool) or not isinstance(pids_limit, int) or pids_limit < 0:
            raise ConfigError("service.config", f"Invalid 'pids_limit' (must be a positive integer) in {path}")

        command = data.get("command", [])
        if not isinstance(command, list) or not all(isinstance(a, str) and a for a in command):
            raise ConfigError("service.config", f"Invalid 'command' (must be a list of non-empty strings) in {path}")

        scripts: dict[str, list[str]] = {}
        scripts_data = data.get("scripts", {})
        if not isinstance(scripts_data, dict):
            raise ConfigError("service.config", f"Invalid 'scripts' (must be a table) in {path}")
        for script_name, argv in scripts_data.items():
            if not script_name or not is_valid_id(script_name):
                raise ConfigError("service.config", f"Invalid script name '{script_name}' in {path}")
            if not isinstance(argv, list) or not argv:
                raise ConfigError("service.config", f"Script '{script_name}' must be a non-empty list in {path}")
            if not all(isinstance(a, str) and a for a in argv):
                raise ConfigError(
                    "service.config", f"Script '{script_name}' must be a list of non-empty strings in {path}"
                )
            scripts[script_name] = list(argv)

        return cls(
            name=name,
            image=image,
            restart=restart,
            ports=ports,
            volumes=volumes,
            scripts=scripts,
            env=env_list,
            command=list(command),
            networks=networks,
            read_only=read_only,
            tmpfs=tmpfs,
            cap_drop=caps["cap_drop"],
            cap_add=caps["cap_add"],
            pids_limit=pids_limit,
        )


def read_services() -> tuple[dict[str, ServiceConfig], list[Issue]]:
    """~/.config/containerctl/services/: the services, and what is wrong with the files that aren't one"""
    return read_configs(SERVICES_DIR, ServiceConfig.from_toml)


def load_services() -> dict[str, ServiceConfig]:
    """The services, for a one-shot caller: broken files are logged"""
    services, problems = read_services()
    log_issues(problems)
    return services


# =============================================================================
# Image recipes: ~/.config/containerctl/images/<name>.toml + <name>.Dockerfile [+ <name>/ context]
# =============================================================================

VALID_URL_CHARS = VALID_ID_CHARS + "/:.@~+%"


def is_valid_https_url(url: str) -> bool:
    """https only, conservative charset: an upstream Dockerfile URL, nothing fancier"""
    return url.startswith("https://") and len(url) > 8 and all(c in VALID_URL_CHARS for c in url)


def normalize_image_ref(ref: str) -> str:
    """`hermes` -> `hermes:latest`, so a project's `image` matches a recipe's `tag` however either was written"""
    if "@" in ref or ":" in ref.rsplit("/", 1)[-1]:
        return ref
    return f"{ref}:latest"


@dataclass
class ImageConfig:
    """An image recipe: `<name>.toml` + `<name>.Dockerfile`, with `<name>/` as the build context when it exists"""

    name: str
    dockerfile: Path
    context: Path  # only used when it is a directory; otherwise the Dockerfile is built with no context at all
    tag: str = ""  # defaults to <name>:latest
    upstream: str = ""  # https URL of the canonical Dockerfile; `image check` diffs the local one against it

    def __post_init__(self):
        self.tag = normalize_image_ref(self.tag or self.name)

    @property
    def repository(self) -> str:
        """`tag` without its tag part: where the cctl-local-NNNNNN builds live"""
        return self.tag.rsplit(":", 1)[0]

    def version_tag(self, n: int) -> str:
        return f"{self.repository}:{LOCAL_VERSION_PREFIX}{n:06d}"

    @property
    def has_context(self) -> bool:
        return self.context.is_dir()

    def recipe_hash(self) -> str | None:
        """sha256 over the Dockerfile and every context file, by sorted relative path. Stored as a label on the
        built image so `check` can tell a stale build from a current one. None (after logging) when unreadable."""
        files = [(self.dockerfile.name, self.dockerfile)]
        if self.has_context:
            files += [
                (str(p.relative_to(self.context)), p)
                for p in sorted(self.context.rglob("*"))
                if p.is_file() and ".git" not in p.relative_to(self.context).parts  # a clone's history is not recipe
            ]
        digest = hashlib.sha256()
        for rel, path in files:
            try:
                data = path.read_bytes()
            except OSError as e:
                log_error("image.recipe", f"Cannot read {path}: {e}")
                return None
            digest.update(f"{rel}\0{len(data)}\0".encode())
            digest.update(data)
        return digest.hexdigest()

    @classmethod
    def from_toml(cls, path: Path) -> "ImageConfig":
        """Parse an image recipe from a TOML file; the Dockerfile must exist beside it. ConfigError otherwise."""
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as e:
            raise ConfigError("image.config", f"Failed to load {path}: {e}") from None
        name = path.stem
        if not is_valid_id(name):
            raise ConfigError(
                "image.config", f"Invalid image name '{name}': must be a-zA-Z0-9-_. (not starting with - or .)"
            )
        tag = data.get("tag", "")
        if not isinstance(tag, str) or (tag and (not is_valid_image_ref(tag) or "@" in tag)):
            raise ConfigError("image.config", f"Invalid tag '{tag}' in {path}: name[:tag], no digest")
        upstream = data.get("upstream", "")
        if not isinstance(upstream, str) or (upstream and not is_valid_https_url(upstream)):
            raise ConfigError("image.config", f"Invalid upstream '{upstream}' in {path}: must be an https URL")
        dockerfile = path.with_suffix(".Dockerfile")
        if not dockerfile.is_file():
            raise ConfigError("image.config", f"Missing {dockerfile.name} beside {path}")
        return cls(name=name, dockerfile=dockerfile, context=path.with_suffix(""), tag=tag, upstream=upstream)


def read_images() -> tuple[dict[str, ImageConfig], list[Issue]]:
    """~/.config/containerctl/images/: the recipes, and what is wrong with the files that aren't one"""
    return read_configs(IMAGES_DIR, ImageConfig.from_toml)


def load_images() -> dict[str, ImageConfig]:
    """The recipes, for a one-shot caller: broken files are logged"""
    images, problems = read_images()
    log_issues(problems)
    return images


def find_image_recipe(ref: str, images: dict[str, ImageConfig]) -> ImageConfig | None:
    """The recipe whose tag is a project's or service's `image`, if any"""
    want = normalize_image_ref(ref)
    return next((image for image in images.values() if image.tag == want), None)


def fetch_upstream(url: str, limit: int = 1 << 20) -> str | None:
    """GET an https URL (CA-verified; nothing is sent but the request) and return its text, None after logging
    on failure. The result is only ever diffed and shown: never written to the config dir, never built."""
    if not is_valid_https_url(url):
        log_error("image.upstream", f"Invalid upstream URL: {url}")
        return None
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = resp.read(limit + 1)
    except (OSError, ValueError) as e:  # URLError and HTTPError are OSErrors
        log_error("image.upstream", f"Failed to fetch {url}: {e}")
        return None
    if len(data) > limit:
        log_error("image.upstream", f"{url}: larger than {limit} bytes, not a Dockerfile")
        return None
    return data.decode("utf-8", errors="replace")


# =============================================================================
# Detected issues: what wants an action right now, from state already in hand (no docker calls)
# =============================================================================


def shortcut_issues(projects: dict[str, ProjectConfig]) -> list[Issue]:
    """Shortcuts resolve globally: the two collisions that make `enter <name>` surprising"""
    owners: dict[str, list[str]] = {}
    for project in projects.values():
        for alias in project.shortcuts:
            owners.setdefault(alias, []).append(project.name)
    issues = []
    for alias, names in sorted(owners.items()):
        who = ", ".join(names)
        if alias in projects:
            msg = f"Shortcut '{alias}' ({who}) is shadowed by project '{alias}'"
            hint = f"rename it in {', '.join(f'{n}.toml' for n in names)}"
            issues.append(Issue(WARNING, "project.shortcut", msg, hint))
        elif len(names) > 1:
            msg = f"Shortcut '{alias}' is defined by several projects: {who}"
            issues.append(Issue(WARNING, "project.shortcut", msg, "`enter` needs it in exactly one"))
    return issues


def secrets_mode_issue() -> Issue | None:
    """A token is only as private as its file: secrets.toml readable by others"""
    try:
        mode = stat.S_IMODE(SECRETS_FILE.stat().st_mode)
    except OSError:  # absent: nothing to protect
        return None
    if mode & 0o077:
        msg = f"{SECRETS_FILE} is mode {mode:04o}, readable by others"
        return Issue(WARNING, "env.secret", msg, f"chmod 600 {SECRETS_FILE}")
    return None


def local_versions(image: ImageConfig, rows: list[Image]) -> tuple[list[int], int | None, str | None]:
    """(versions, current, tag_id) from `docker images` rows: the repository's cctl-local-N builds, sorted; the one
    the recipe tag points at (None when it points elsewhere or nowhere); the image ID the recipe tag points at
    (None when the tag doesn't exist)"""
    ids = {r.Tag: r.ID for r in rows if r.Repository == image.repository}
    versions: dict[int, str] = {}
    for tag, id_ in ids.items():
        n = tag.removeprefix(LOCAL_VERSION_PREFIX)
        if n != tag and n.isdigit():
            versions[int(n)] = id_
    tag_id = ids.get(image.tag.rsplit(":", 1)[1])
    current = max((n for n, id_ in versions.items() if id_ == tag_id), default=None)
    return sorted(versions), current, tag_id


def image_version_issue(name: str, image: ImageConfig, versions: list[int], current: int | None) -> Issue | None:
    """Builds pile up: past IMAGE_VERSIONS_WARN local versions, name the ones that are safe to try removing"""
    if len(versions) <= IMAGE_VERSIONS_WARN:
        return None
    old = " ".join(image.version_tag(n) for n in versions if n != current)
    return Issue(WARNING, "image.versions", f"{name} has {len(versions)} local versions", f"docker rmi {old}")


def failure_issues(failures: dict[str, Failure]) -> list[Issue]:
    """Failed listings, one issue per distinct reason naming the listings it took down: a daemon that is down is
    one line, not four, and a lone `volume ls` that fails is exactly that"""
    by_reason: dict[tuple[str, str], list[str]] = {}
    for key, failure in failures.items():
        by_reason.setdefault((failure.msg, failure.hint), []).append(key)
    return [
        Issue(ERROR, "docker", f"{msg} ({', '.join(keys)})", hint, "\n".join(failures[key].detail for key in keys))
        for (msg, hint), keys in by_reason.items()
    ]


def orphan_issues(names: list[str], projects: set[str], services: set[str]) -> list[Issue]:
    """cctl- containers nothing in the config dirs accounts for anymore. A TOML that is there but broken still
    owns its container (it is not in the loaded set, hence the look at the files): that one is a config issue.
    The hint leads with the file: a project container holds a workspace, `rm -f` takes it along."""
    issues = []
    for prefix, directory, known, kind in (
        (CONTAINER_PREFIX, PROJECTS_DIR, projects, "project"),
        (SERVICE_PREFIX, SERVICES_DIR, services, "service"),
    ):
        for container in names:
            name = container.removeprefix(prefix)
            if name == container or name in known or (directory / f"{name}.toml").is_file():
                continue
            msg = f"Container {container} has no {kind} config"
            issues.append(Issue(WARNING, f"{kind}.orphan", msg, f"restore {name}.toml, or docker rm -f {container}"))
    return issues


def leftover_note(container: Container) -> str:
    """What a question says of a container still under its spawning name, before removing it"""
    return f"{container.Names} ({created_ago(container)}: a spawn still running, or one that died)"


def created_ago(container: Container) -> str:
    """ "created 3 minutes ago", in docker's own words (`RunningFor` is the time since creation, running or not)"""
    return f"created {container.RunningFor}" if container.RunningFor else "created at an unknown time"


def container_age(container: Container) -> float | None:
    """Seconds since the container was created, from `docker ps`'s CreatedAt ("2026-11-11 12:27:03 +0100 CET");
    None when that is not a date we can read"""
    try:
        created = datetime.datetime.strptime(" ".join(container.CreatedAt.split()[:3]), "%Y-%m-%d %H:%M:%S %z")
    except ValueError:
        return None
    return (datetime.datetime.now(datetime.timezone.utc) - created).total_seconds()


def spawning_issues(containers: list[Container]) -> list[Issue]:
    """Containers still under their spawning name after SPAWN_WARN_AFTER seconds: a spawn that takes its time in
    another terminal (a big clone), one that is stuck, or one that died. Nothing here can tell which, and the
    message says so. Younger ones are a spawn doing its job: no warning. An age we can't read counts as old."""
    issues = []
    for c in containers:
        if not c.Names.startswith(SPAWNING_PREFIX):
            continue
        age = container_age(c)
        if age is not None and age < SPAWN_WARN_AFTER:
            continue
        msg = f"Container {c.Names} is still spawning ({created_ago(c)}): maybe just slow, maybe stuck or dead"
        project = c.Names.removeprefix(SPAWNING_PREFIX)
        hint = f"no spawn of it running? `destroy {project}` asks, then removes it; or docker rm -f {c.Names}"
        issues.append(Issue(WARNING, "project.spawning", msg, hint))
    return issues


def detect_issues(
    projects: dict[str, ProjectConfig],
    services: dict[str, "ServiceConfig"],
    images: dict[str, ImageConfig],
    listings: dict[str, Listing],
) -> list[Issue]:
    """Every issue detectable from loaded config and the docker listings the fetcher already holds. What a failed
    listing would have told is skipped, not guessed: the failure is the issue."""
    issues = failure_issues({key: listing.failure for key, listing in listings.items() if listing.failure})
    issues += shortcut_issues(projects)
    if secrets := secrets_mode_issue():
        issues.append(secrets)
    for key, listing in listings.items():
        if count := len(listing.skipped):
            msg = f"{count} unreadable row{'s' if count > 1 else ''} left out of the {key} listing"
            issues.append(Issue(WARNING, "docker.rows", msg, "its details have them", "\n".join(listing.skipped)))
    if not listings["images"].failure:
        for name, image in images.items():
            versions, current, _ = local_versions(image, listings["images"].rows)
            if issue := image_version_issue(name, image, versions, current):
                issues.append(issue)
    if not listings["containers"].failure:
        rows = listings["containers"].rows
        issues += orphan_issues([c.Names for c in rows], set(projects), set(services)) + spawning_issues(rows)
    return issues


# =============================================================================
# Workspace cleanliness check
# =============================================================================
#
# "Is it safe to destroy this container?" — before you tear a project down, make
# sure nothing in its workspace would be lost: no uncommitted/unpushed work, no
# stashes, no detached HEAD, and no directory you cloned by hand but forgot to
# add to the project's TOML.
#
# The logic is written once against two injected callables so it runs both
# host-side (over `docker exec`) and inside the container (`check-local`, over
# plain `git`) with zero duplication:
#   - list_dirs() -> list[str]           names of directories in the workspace
#   - run_git(dirname, args) -> (rc, out)  run `git -C <workspace>/<dirname> ...`


@dataclass
class RepoStatus:
    """Cleanliness of a single directory in the workspace."""

    name: str
    is_git: bool = True
    uncommitted: bool = False  # staged or modified tracked files
    untracked: bool = False  # files git doesn't know about (ignored ones don't count: they said they can go)
    detached: bool = False  # HEAD not on a branch
    unpushed: list[str] = field(default_factory=list)  # one line per local branch not provably on its remote
    tagged: int = 0  # commits reachable from a tag and from no remote ref
    stashes: int = 0
    worktrees: list[str] = field(default_factory=list)  # linked worktrees, by path: work may sit in each
    unknown: list[str] = field(default_factory=list)  # the git calls that failed: what they'd have said, nobody knows
    # Each checked-out submodule, at any depth, by path: a repo of its own, with its own branches, stashes and
    # remote. The superproject's `status` only says whether it sits on the recorded commit, not whether that
    # commit, or anything else in there, is on the submodule's remote.
    submodules: list["RepoStatus"] = field(default_factory=list)
    head_unpushed: int = 0  # of a submodule: commits on its HEAD and on none of its remote's refs
    # Whether this repo keeps the remote's tags apart from its own (see UPSTREAM_TAGS). Asked only when a count
    # above depends on it: without, a tag the remote has counts like one made here, and the line says how to tell.
    tags_mirrored: bool = True
    head_under_tag: bool = False  # no mirror, and some of `head_unpushed` sit under a tag: maybe the remote's

    @property
    def clean(self) -> bool:
        """A repo is clean only if everything is committed *and* verifiably pushed, on every branch."""
        return not self.problems()

    def problems(self) -> list[str]:
        """Human-readable reasons this dir is not clean (empty when clean)."""
        if not self.is_git:
            return ["not a git repository"]
        out = []
        if self.uncommitted:
            out.append("uncommitted changes")
        if self.untracked:
            out.append("untracked files")
        if self.detached:
            out.append("detached HEAD")
        out += self.unpushed
        if self.tagged:
            theirs = "" if self.tags_mirrored else MAYBE_THE_REMOTES
            out.append(f"{self.tagged} unpushed commit{'s' if self.tagged != 1 else ''} held by tags only{theirs}")
        if self.stashes:
            out.append(f"{self.stashes} stash{'es' if self.stashes != 1 else ''}")
        if self.head_unpushed:
            theirs = MAYBE_THE_REMOTES if self.head_under_tag else ""
            out.append(f"{self.head_unpushed} unpushed commit{'s' if self.head_unpushed != 1 else ''} on HEAD{theirs}")
        out += [f"linked worktree {path}" for path in self.worktrees]
        out += [f"`git {what}` failed, could not check" for what in self.unknown]
        out += [f"submodule {sub.name}: {problem}" for sub in self.submodules for problem in sub.problems()]
        return out


@dataclass
class WorkspaceReport:
    """Result of checking a whole workspace against a project's expected repos."""

    statuses: list[RepoStatus] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)  # dirs present but not in the config
    missing: list[str] = field(default_factory=list)  # dirs in the config but absent
    loose: list[str] = field(default_factory=list)  # what sits in the workspace and is not a dir: in no repo
    changed: list[str] = field(default_factory=list)  # those of `loose` we injected, no longer as the host has them

    @property
    def clean(self) -> bool:
        return all(s.clean for s in self.statuses) and not self.extra and not self.missing and not self.loose

    @property
    def dirty(self) -> list[str]:
        """The names behind a DIRTY verdict: unclean or unconfigured dirs, loose files, then configured-but-absent"""
        dirs = [s.name for s in self.statuses if not s.clean or s.name in self.extra]
        return dirs + list(self.loose) + list(self.missing)


# One line per local branch: name, upstream ref, how it stands against it ("[ahead 2]", "[gone]", "" when level).
# Tab-separated (%09): a ref name holds neither tabs nor spaces.
BRANCH_FORMAT = "--format=%(refname:short)%09%(upstream)%09%(upstream:track)"

# Where spawn's clone (and every fetch after it) mirrors the remote's tags: the remote-tracking refs git has for
# branches and not for tags. Outside refs/remotes, so `git branch -r` doesn't list them. See inspect_repo.
UPSTREAM_TAGS = "refs/upstream-tags"
MIRROR_REFSPEC = f"+refs/tags/*:{UPSTREAM_TAGS}/*"
CLONE_MIRRORS_TAGS = ["-c", f"remote.origin.fetch={MIRROR_REFSPEC}"]  # `git clone` takes it as is
# A repo without the mirror (a submodule, a clone made by hand or by an older spawn): said after a count that may
# then hold the remote's own tagged commits, with what settles it, in that repo, for this fetch and the next ones
MAYBE_THE_REMOTES = (
    f" (or the remote's own tags: `git config --add remote.origin.fetch '{MIRROR_REFSPEC}' && git fetch` tells)"
)


def unpushed_branches(listing: str) -> list[str]:
    """From `git for-each-ref BRANCH_FORMAT refs/heads`: a line for each branch whose commits are not provably
    on a remote. Every local branch counts, not just the one checked out: they all die with the container.
    (Against the remote-tracking refs as they are: no fetch, a check never talks to the network.)"""
    out = []
    for line in listing.splitlines():
        branch, _, rest = line.partition("\t")
        upstream, _, track = rest.partition("\t")
        if not branch:
            continue
        if not upstream:
            out.append(f"branch {branch}: no upstream")
        elif not upstream.startswith("refs/remotes/"):  # tracking another local branch proves nothing is pushed
            out.append(f"branch {branch}: upstream {upstream.removeprefix('refs/heads/')} is local, not a remote")
        elif "gone" in track:
            out.append(f"branch {branch}: upstream is gone")
        elif "ahead" in track:
            count = track.split("ahead", 1)[1].split(",")[0].strip(" ]")  # "[ahead 2, behind 1]" -> "2"
            out.append(f"branch {branch}: {count} unpushed commit{'s' if count != '1' else ''}")
    return out


# Prints the path of every checked-out submodule, nested ones included, relative to the repo it is asked in. Git
# runs the quoted part with sh, once per submodule; one that was never initialized holds nothing and is not listed.
LIST_SUBMODULES = ["submodule", "foreach", "--quiet", "--recursive", 'echo "$displaypath"']


def inspect_repo(name: str, run_git: Callable[[str, list[str]], tuple[int, str]], submodule_of: str = "") -> RepoStatus:
    """Build a RepoStatus for one workspace directory using an injected git runner. A git call that fails is not
    read as "nothing to report": it lands in `unknown`, and the dir is not clean.

    Its submodules get the same inspection, each as the repo it is (`submodule_of` is then the workspace dir it
    was found in, `name` its path there), with one difference: a detached HEAD is where git leaves a submodule,
    so what is asked instead is whether the commit it sits on is on its remote."""
    status = RepoStatus(name=name)
    name = f"{submodule_of}/{name}" if submodule_of else name  # where git runs; status.name is what gets shown

    rc, _ = run_git(name, ["rev-parse", "--git-dir"])
    if rc != 0:
        status.is_git = False
        return status

    def ask(args: list[str]) -> str | None:
        rc, out = run_git(name, args)
        if rc != 0:
            status.unknown.append(args[0])
            return None
        return out

    # --untracked-files: said here, so a `status.showUntrackedFiles = no` in some gitconfig can't hide them
    if (porcelain := ask(["status", "--porcelain", "--untracked-files=normal"])) is not None:
        lines = [ln for ln in porcelain.splitlines() if ln.strip()]
        status.untracked = any(ln.startswith("??") for ln in lines)
        status.uncommitted = any(not ln.startswith("??") for ln in lines)

    def count(args: list[str]) -> int:
        """What a `rev-list --count` says; 0 and a line in `unknown` when it is not a count: not an answer"""
        if (out := ask(args)) is None:
            return 0
        if not out.strip().isdigit():
            status.unknown.append(args[0])
            return 0
        return int(out.strip())

    upstream_tags = f"--glob={UPSTREAM_TAGS}/*"
    if submodule_of:
        status.head_unpushed = count(["rev-list", "--count", "HEAD", "--not", "--remotes", upstream_tags])
    else:
        rc, _ = run_git(name, ["symbolic-ref", "-q", "HEAD"])
        status.detached = rc != 0

    if (branches := ask(["for-each-ref", BRANCH_FORMAT, "refs/heads"])) is not None:
        status.unpushed = unpushed_branches(branches)

    # Commits only a tag holds on to (tagged on a detached HEAD, then moved on): on no branch, so not seen above.
    # A tag on a pushed commit is not looked at: whether the tag itself was pushed, only the network knows.
    # Git keeps no word of where a tag came from, so spawn's clone mirrors the remote's own (UPSTREAM_TAGS): what
    # those hold is on the remote, whatever branch it is on or off. A repo without the mirror (a submodule, one
    # cloned by hand or by an older spawn) counts every tag off the remote branches: more DIRTY than it should be,
    # never less. So when a count is not zero, ask whether the mirror is there; `problems` says so when it is not.
    status.tagged = count(["rev-list", "--count", "--tags", "--not", "--remotes", upstream_tags])
    if status.tagged or status.head_unpushed:
        _, refspecs = run_git(name, ["config", "--get-regexp", r"^remote\..*\.fetch$"])  # exits 1 on no match
        status.tags_mirrored = UPSTREAM_TAGS in refspecs
        if status.head_unpushed and not status.tags_mirrored:  # only worth saying if a tag could explain any
            untagged = count(["rev-list", "--count", "HEAD", "--not", "--remotes", "--tags"])
            status.head_under_tag = untagged < status.head_unpushed

    if (stash := ask(["stash", "list"])) is not None:
        status.stashes = len([ln for ln in stash.splitlines() if ln.strip()])

    # The first `worktree` line is the main one (this dir, or the repo this dir is a worktree of); every other is
    # a linked worktree, wherever it lives, prunable ones included: `git worktree remove`/`prune` settles each.
    if (worktrees := ask(["worktree", "list", "--porcelain"])) is not None:
        paths = [ln.removeprefix("worktree ") for ln in worktrees.splitlines() if ln.startswith("worktree ")]
        status.worktrees = paths[1:]

    # --recursive lists them at every depth from here, so a submodule is not asked for its own
    if not submodule_of and (listed := ask(LIST_SUBMODULES)) is not None:
        status.submodules = [inspect_repo(path, run_git, submodule_of=name) for path in listed.splitlines() if path]

    return status


def check_workspace(
    expected: list[str] | None,
    list_entries: Callable[[], dict[str, bool]],
    run_git: Callable[[str, list[str]], tuple[int, str]],
    injected: dict[str, str] | None = None,
    digest: Callable[[str], str | None] = lambda name: None,
) -> WorkspaceReport:
    """Compare the workspace on disk against the configured repos.

    ``list_entries`` gives everything directly in the workspace, hidden names included: name -> is it a
    directory. Every directory must be a clean git repo; anything else is loose (in no repo, so nowhere else).

    ``expected`` is the list of directory names the config says should be there;
    pass ``None`` to skip extra/missing detection (e.g. no manifest available)
    and only report per-repo git cleanliness (and loose files: no repo list ever expects those).

    ``injected`` is the one exception to loose: the files containerctl itself put in the workspace (see
    injected_files), name -> sha256 of the host's copy. One that ``digest`` (name -> sha256 of what is there, None
    when that can't be said) finds unchanged exists on the host too; changed, it is loose like any other.
    """
    entries = list_entries()
    present = sorted(name for name, is_dir in entries.items() if is_dir)
    report = WorkspaceReport()
    for name in sorted(name for name, is_dir in entries.items() if not is_dir):
        if injected and name in injected:
            if digest(name) == injected[name]:
                continue
            report.changed.append(name)
        report.loose.append(name)

    for name in present:
        report.statuses.append(inspect_repo(name, run_git))

    if expected is not None:
        expected_set = set(expected)
        present_set = set(present)
        report.extra = sorted(present_set - expected_set)
        report.missing = sorted(expected_set - present_set)

    return report


def print_report(label: str, report: WorkspaceReport) -> bool:
    """Print a workspace report; return True when clean."""
    print(f"{label}:")
    if not report.statuses and not report.missing and not report.loose:
        print("  (workspace is empty)")
    for status in report.statuses:
        tag = "extra" if status.name in report.extra else None
        if status.clean and not tag:
            print(f"  ✓ {status.name}")
            continue
        reasons = status.problems()
        if tag:
            reasons = ["not in config", *reasons] if reasons else ["not in config"]
        print(f"  ✗ {status.name} — {', '.join(reasons)}")
    for name in report.loose:
        why = "injected, but no longer as the host has it" if name in report.changed else "not a directory, in no repo"
        print(f"  ✗ {name} — {why}")
    for name in report.missing:
        print(f"  ✗ {name} — configured but absent")
    print("  clean" if report.clean else "  DIRTY")
    return report.clean


# --- host-side adapters (run over `docker exec`) --------------------------------


def _host_list_entries(container: str, workspace: str, user: str | None = None) -> dict[str, bool]:
    """Everything directly under the container's workspace: name -> is a directory (`ls -p` marks those with a
    trailing slash). No workspace at all is an answer (nothing there). A listing that failed is not: DockerError
    (a stopped container, most often), where "empty" would have been a guess that reads as "clean"."""
    script = '[ -d "$1" ] || exit 0; ls -A1p -- "$1"'  # the workspace rides in as $1: data, never code
    command = ["sh", "-c", script, "sh", workspace]
    result = Docker.exec_noninteractive(container, command, user=user)
    if result.returncode != 0:  # docker's failure (stopped, daemon down) or ls's own (unreadable): said as which
        raise DockerError(docker_failure(["exec", container, *command], result, inside="ls"))
    return {ln.rstrip("/"): ln.endswith("/") for ln in result.stdout.splitlines() if ln.strip("/")}


def _host_git(container: str, workspace: str, user: str | None = None) -> Callable[[str, list[str]], tuple[int, str]]:
    def run_git(name: str, args: list[str]) -> tuple[int, str]:
        result = Docker.exec_noninteractive(container, ["git", "-C", f"{workspace}/{name}", *args], user=user)
        return result.returncode, result.stdout

    return run_git


def injected_files(project: "ProjectConfig") -> dict[str, str]:
    """The env files spawn puts directly in the workspace (in no repo, so `check` would call them loose): name ->
    sha256 of the host's copy as it is now. One the host can't read is not listed: nothing to match."""
    out = {}
    for ef in project.env_files:
        if os.path.normpath(os.path.dirname(ef.dest)) != os.path.normpath(project.workspace):
            continue
        try:
            with open(ef.src, "rb") as f:
                out[os.path.basename(ef.dest)] = hashlib.file_digest(f, "sha256").hexdigest()
        except OSError:
            pass
    return out


def _host_digest(container: str, workspace: str, user: str | None = None) -> Callable[[str], str | None]:
    def digest(name: str) -> str | None:
        result = Docker.exec_noninteractive(container, ["sha256sum", "--", f"{workspace}/{name}"], user=user)
        return result.stdout.split()[0] if result.returncode == 0 and result.stdout.split() else None

    return digest


def project_report(project: "ProjectConfig") -> WorkspaceReport:
    """Check a project's workspace from the host, over `docker exec`; the container must exist, and run:
    DockerError when its workspace can't be listed. `check` prints the report, `destroy` asks with it in hand."""
    name, workspace, user = project.container_name, project.workspace, project.user  # as its user: git's own
    expected = [r.name for r in project.repos]  # ownership check refuses a repo that is somebody else's
    return check_workspace(
        expected,
        lambda: _host_list_entries(name, workspace, user),
        _host_git(name, workspace, user),
        injected_files(project),
        _host_digest(name, workspace, user),
    )


@dataclass
class DestroyQuestion:
    """What `destroy` would remove, and what must be known before saying yes. Built once, for the CLI and the
    TUI alike: what one discloses, the other does."""

    names: list[str] = field(default_factory=list)  # the containers the question is about: only those may go
    verdict: str = ""  # of the workspace check, when it is not "clean"
    report: WorkspaceReport | None = None  # the check itself, when it ran
    text: str = ""  # the question; empty when there is nothing to destroy


def destroy_question(project: "ProjectConfig") -> DestroyQuestion:
    """Look at what a project has (one `ps`), `check` the workspace of its container, and word the question: the
    verdict (DIRTY, or a check that could not run) is in it, and so is what an unfinished spawn left (never
    entered, so nothing to check, but it may be a spawn still running elsewhere)."""
    container, leftover = Docker.project_containers(project)
    question = DestroyQuestion(names=[c.Names for c in (container, leftover) if c])
    what = []
    if container:
        try:
            question.report = project_report(project)
        except DockerError as e:  # stopped, most often: nothing says the workspace is clean; still the user's call
            log_error("docker.command", e.failure.detail)
            question.verdict = f"its workspace could NOT be checked ({e})"
        else:
            if not question.report.clean:
                question.verdict = f"its workspace is DIRTY ({', '.join(question.report.dirty)})"
        what.append(f"{project.container_name}{', though ' + question.verdict if question.verdict else ''}")
    if leftover:
        what.append(leftover_note(leftover))
    if what:
        question.text = f"Destroy {' and '.join(what)}?"
    return question


# --- local adapters (run inside the container, plain git) -----------------------


def _local_workspace() -> Path:
    return Path.home() / "workspace"


def _local_list_entries() -> dict[str, bool]:
    """Same answer as _host_list_entries, from inside: hidden names included, a symlink is not a directory"""
    ws = _local_workspace()
    if not ws.is_dir():
        return {}
    return {p.name: p.is_dir() and not p.is_symlink() for p in ws.iterdir()}


def _local_git(name: str, args: list[str]) -> tuple[int, str]:
    result = subprocess.run(
        ["git", "-C", str(_local_workspace() / name), *args],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout


def _local_digest(name: str) -> str | None:
    try:
        with open(_local_workspace() / name, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    except OSError:
        return None


def read_injected() -> dict[str, str]:
    """Read, inside a container, what install_self said of the files injected in the workspace (see
    injected_files): as the host had them at the last `enter`. Nothing readable is nothing injected."""
    path = Path.home() / ".config" / "containerctl" / "injected.json"
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(data, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        return data
    return {}


def read_manifest() -> list[str] | None:
    """Read the injected expected-repos manifest inside a container, if present."""
    path = Path.home() / ".config" / "containerctl" / "expected.json"
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        return data
    return None


def install_self(
    container: str,
    expected: list[str],
    user: str = CONTAINER_USER,
    home: str = CONTAINER_HOME,
    injected: dict[str, str] | None = None,
) -> bool:
    """Copy containerctl + the expected-repos manifest into a container, under `home`, for `user` (the
    project's: see ProjectConfig), and beside the manifest what `injected_files` says: `check-local` has no
    host to ask.

    Makes ``containerctl check-local`` available on that user's PATH. False
    when a copy failed (logged): that fails a spawn, `enter` goes on without.

    Runs on every `enter`, so it must only touch what it owns: directories are
    created as the user, and the three files `docker cp` writes (as root, always) are
    chowned by name, as root. Nothing recursive, nothing else in the home is affected.
    """
    self_path = Path(__file__).resolve()
    ok = True
    bin_dir, config_dir = f"{home}/.local/bin", f"{home}/.config/containerctl"
    self_dest, manifest, injected_dest = (
        f"{bin_dir}/containerctl",
        f"{config_dir}/expected.json",
        f"{config_dir}/injected.json",
    )

    Docker.exec_noninteractive(container, ["mkdir", "-p", bin_dir, config_dir], user=user)
    if not Docker.cp(str(self_path), container, self_dest):
        return False
    Docker.exec_noninteractive(container, ["chmod", "+x", self_dest], user=ROOT)

    # Write each to a temp file, whole, then copy it in.
    for content, dest in ((expected, manifest), (injected or {}, injected_dest)):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        try:
            json.dump(content, tmp)
            tmp.flush()
            tmp.close()
            if not Docker.cp(tmp.name, container, dest):
                ok = False
        finally:
            os.unlink(tmp.name)

    # "user:" = its own group
    Docker.exec_noninteractive(container, ["chown", f"{user}:", self_dest, manifest, injected_dest], user=ROOT)
    # Ensure ~/.local/bin is on PATH for future interactive shells (idempotent, as the user so a
    # missing .bashrc is created with the right owner).
    # Shell required here: the conditional append (`grep -qs ... || echo ... >>`) has
    # no argv equivalent. The home was validated when the config was parsed, but
    # charset-check both values at the call site anyway (defense in depth) since they land
    # in a shell string; the single-quote check guards the `'...'` quoting below.
    bashrc = f"{home}/.bashrc"
    path_line = 'export PATH="$HOME/.local/bin:$PATH"'
    if all(c in VALID_PATH_CHARS for c in bashrc) and "'" not in bashrc + path_line:
        Docker.exec_noninteractive(
            container,
            ["sh", "-c", f"grep -qs '.local/bin' '{bashrc}' || echo '{path_line}' >> '{bashrc}'"],
            user=user,
        )
    else:
        log_error("install_self.unsafe_shell_value", f"Refusing shell interpolation for {bashrc!r}")
    return ok


# =============================================================================
# Terminal Primitives
# =============================================================================


class Term:
    """ANSI escape code utilities for terminal manipulation"""

    # Screen control
    CLEAR = "\033[2J\033[H"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    INVERT = "\033[7m"
    HIDE_CURSOR = "\033[?25l"
    SHOW_CURSOR = "\033[?25h"
    ALT_SCREEN = "\033[?1049h"
    MAIN_SCREEN = "\033[?1049l"
    CLEAR_LINE = "\033[2K"

    # Colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    # Box drawing characters
    BOX_H = "─"
    BOX_V = "│"
    BOX_TL = "┌"
    BOX_TR = "┐"
    BOX_BL = "└"
    BOX_BR = "┘"
    BOX_T = "┬"
    BOX_B = "┴"
    BOX_L = "├"
    BOX_R = "┤"
    BOX_X = "┼"

    # Key codes
    KEY_UP = "\x1b[A"
    KEY_DOWN = "\x1b[B"
    KEY_RIGHT = "\x1b[C"
    KEY_LEFT = "\x1b[D"
    KEY_ENTER = "\r"
    _cooked: list | None = None  # the terminal's own settings while we hold it raw: see raw_on
    KEY_ESC = "\x1b"
    KEY_TAB = "\t"
    KEY_BACKSPACE = "\x7f"

    @staticmethod
    def size() -> tuple[int, int]:
        """Get terminal size (rows, cols)"""
        cols, rows = os.get_terminal_size()
        return rows, cols

    @staticmethod
    def move(row: int, col: int) -> None:
        """Move cursor to position (1-indexed)"""
        print(f"\033[{row};{col}H", end="", flush=True)

    @staticmethod
    def clear() -> None:
        """Clear screen and move cursor to top-left"""
        print("\033[2J\033[H", end="", flush=True)

    @staticmethod
    def write(text: str) -> None:
        """Write text at current cursor position"""
        print(text, end="", flush=True)

    @staticmethod
    def writeln(row: int, col: int, text: str) -> None:
        """Write text at specific position"""
        Term.move(row, col)
        Term.write(text)

    @staticmethod
    def raw_on() -> None:
        """Take the keyboard for a session: every key at once, none echoed, until raw_off. Held from one end of
        the TUI to the other, not per key read: a mode that flips back between two reads leaves gaps in which a
        typed character is echoed over the screen. No-op when already on, or when stdin isn't a terminal.

        tty.setraw, less two things. Output processing stays as it was (a "\n" is still a new line). Ctrl-C
        still interrupts, a docker call that hangs included; Ctrl-Z and Ctrl-Backslash are plain keys: suspending or
        dumping core here would leave the shell with this mode. Set without setraw's default flush (TCSAFLUSH),
        which drops whatever was typed since the last read."""
        if Term._cooked is not None or not sys.stdin.isatty():
            return
        fd = sys.stdin.fileno()
        Term._cooked = termios.tcgetattr(fd)
        tty.setraw(fd, termios.TCSANOW)
        mode = termios.tcgetattr(fd)
        mode[1] |= Term._cooked[1] & termios.OPOST
        mode[3] |= termios.ISIG
        off = os.fpathconf(fd, "PC_VDISABLE")
        mode[6][termios.VSUSP] = mode[6][termios.VQUIT] = bytes([off])
        termios.tcsetattr(fd, termios.TCSANOW, mode)

    @staticmethod
    def raw_off() -> None:
        """Give the keyboard back as raw_on found it (to a shell, a docker exec, an input()). No-op when off."""
        if Term._cooked is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, Term._cooked)
            Term._cooked = None

    @staticmethod
    def getch(timeout: float | None = None) -> str:
        """Read single keypress; with a `timeout` (seconds), "" when none came in time. Inside a raw_on session
        the mode is left alone; outside one, it is raw for this read only.

        Bytes are read off the fd itself, never through sys.stdin: its buffer takes every byte that is there on
        the first read, and select, which only sees the fd, then reports nothing left while the rest of an arrow
        key or of a burst sits in Python."""
        fd = sys.stdin.fileno()
        ours = Term._cooked is None
        if ours:
            Term.raw_on()

        def ready(seconds: float) -> bool:
            return bool(select.select([fd], [], [], seconds)[0])

        def read() -> str:
            return os.read(fd, 1).decode(errors="replace")

        try:
            if timeout is not None and not ready(timeout):
                return ""
            ch = read()
            # Handle escape sequences: the rest follows at once, a lone Esc is followed by nothing
            if ch == "\x1b" and ready(0.1):
                ch += read()
                if ch[-1] == "[" and ready(0.1):
                    ch += read()
            return ch
        finally:
            if ours:
                Term.raw_off()

    @staticmethod
    def style(text: str, *styles: str) -> str:
        """Apply styles to text"""
        return "".join(styles) + text + Term.RESET


class Screen:
    """Buffered screen rendering - builds frame in memory, flushes once"""

    def __init__(self):
        self.buf: list[str] = []

    def clear(self) -> None:
        """Reset buffer"""
        self.buf = []

    def move(self, row: int, col: int) -> None:
        """Add cursor move to buffer"""
        self.buf.append(f"\033[{row};{col}H")

    def write(self, text: str) -> None:
        """Add text to buffer"""
        self.buf.append(text)

    def clear_line(self) -> None:
        """Add clear line to buffer"""
        self.buf.append(Term.CLEAR_LINE)

    def writeln(self, row: int, col: int, text: str) -> None:
        """Add positioned text to buffer"""
        self.move(row, col)
        self.write(text)

    def flush(self) -> None:
        """Write entire buffer to stdout at once"""
        sys.stdout.write("".join(self.buf))
        sys.stdout.flush()
        self.buf = []


# =============================================================================
# Docker Backend
# =============================================================================


REFUSED = 2  # exit code of a call we never made: an argument failed validation (what, is in the log)


class Docker:
    """Docker CLI wrapper using subprocess"""

    @staticmethod
    def _run(
        args: list[str],
        capture: bool = True,
        stdin=None,
        input: str | None = None,
        timeout: float | None = None,
        own_group: bool = False,
    ) -> subprocess.CompletedProcess:
        """Run docker command; `input` feeds stdin from a string, `stdin` from a file (never both). With a
        `timeout` (seconds; see QUERY_TIMEOUT), a call that hasn't answered by then is killed.

        `own_group` takes the call out of the terminal's foreground process group, and off its stdin: a Ctrl-C
        goes to that whole group, whichever thread started the process and whatever it reads from, and a listing
        the fetcher is waiting on must not die of a Ctrl-C meant for an action (see App.handle_key). Out of the
        group, nothing the terminal sends reaches it (Ctrl-C, the hangup of a closed window), so we end it
        ourselves: see _run_own_group. Always captured, `stdin` and `input` don't apply."""
        cmd = ["docker"] + args
        try:
            if own_group:
                return Docker._run_own_group(cmd, timeout)
            if input is not None:
                return subprocess.run(cmd, capture_output=capture, text=True, input=input, timeout=timeout)
            return subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout, stdin=stdin)
        except FileNotFoundError:  # no docker on PATH: reads as a failed call, which every caller already reports
            return subprocess.CompletedProcess(cmd, 127, "", "docker: command not found on PATH")
        except subprocess.TimeoutExpired:  # so does a daemon that hangs
            return subprocess.CompletedProcess(cmd, TIMED_OUT, "", f"{NO_ANSWER} {timeout}s")

    _own_group: set[subprocess.Popen] = set()  # the own_group calls in flight, whichever thread waits on them
    _own_group_lock = threading.Lock()
    _leaving = False  # set at exit (see kill_own_group): no own_group call starts after it

    @staticmethod
    def _run_own_group(cmd: list[str], timeout: float | None) -> subprocess.CompletedProcess:
        """subprocess.run for a call in its own process group (see _run), which we keep track of while it runs:
        whoever waits on it kills it when its time is up or the wait is interrupted, and kill_own_group gets the
        ones nobody waits on any more (the fetcher's, a daemon thread, when we exit)."""
        pipe = subprocess.PIPE
        with Docker._own_group_lock:  # started and noted as one step: kill_own_group misses none, and none starts after
            if Docker._leaving:
                return subprocess.CompletedProcess(cmd, 1, "", "containerctl: leaving, not asked")
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=pipe, stderr=pipe, text=True, process_group=0)
            Docker._own_group.add(proc)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except BaseException:  # its time is up, Ctrl-C, an exit: it does not outlive the wait
            proc.kill()
            proc.communicate()
            raise
        finally:
            with Docker._own_group_lock:
                Docker._own_group.discard(proc)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)

    @staticmethod
    def kill_own_group(leaving: bool = False) -> None:
        """Kill the own_group calls still in flight: a `docker ps` waiting on a hung daemon must not stay behind
        us. When the TUI stops its fetcher, and with `leaving` at exit (see leave_cleanly), after which none may
        start: the fetcher is a daemon thread, still running while we go. What can't be covered is our own
        SIGKILL, or the power: such a client is then left waiting, harmless, until the daemon answers or restarts."""
        with Docker._own_group_lock:
            Docker._leaving = Docker._leaving or leaving
            procs = list(Docker._own_group)
        for proc in procs:
            try:
                proc.kill()
            except OSError:  # already gone
                pass

    @staticmethod
    def _absent(args: list[str], result: subprocess.CompletedProcess) -> bool:
        """For a call asking "does it exist?": False when it does, True when docker said there is no such thing,
        DockerError for any other failure (no answer in time, daemon down, ...). Not knowing is neither yes nor
        no, and a caller that read it as "no" would create, pull or report over something that is there."""
        if result.returncode == 0:
            return False
        failure, text = docker_failure(args, result), (result.stderr or "").lower()
        # A failure we know first: "docker: command not found" is not docker saying the image was not found
        if failure.msg == UNKNOWN_DOCKER_FAILURE[0] and any(words in text for words in NO_SUCH):
            return True
        raise DockerError(failure)

    @staticmethod
    def _stream(args: list[str], env: dict[str, str] | None = None) -> int:
        """Run a docker command on our own terminal (logs, exec: nothing captured) and hand back its exit code.
        `env` extends the client's environment for this one call (see exec)."""
        kwargs = {"env": {**os.environ, **env}} if env else {}
        try:
            return subprocess.run(["docker"] + args, **kwargs).returncode
        except FileNotFoundError:
            log_error("docker.command", "docker: command not found on PATH")
            return 127

    @staticmethod
    def _list(args: list[str], parse: Callable[[dict], object], timeout: float = QUERY_TIMEOUT) -> Listing:
        """Run a listing whose output is one JSON object per line (JSON Lines: `--format '{{json .}}'`), so each
        line stands alone: one that isn't JSON, or isn't a row we can use, is skipped and the others still count.
        Nothing is logged here: the failure or the skipped lines are handed back."""
        result = Docker._run(args, timeout=timeout, own_group=True)
        if result.returncode != 0:
            return Listing(failure=docker_failure(args, result))
        listing = Listing()
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            try:
                listing.rows.append(parse(json.loads(line)))
            except ValueError as e:  # not JSON (JSONDecodeError is one), or not a row
                listing.skipped.append(f"docker {' '.join(args)}: {e} in {line[:200]}")
        return listing

    # A listing hands back its outcome, and the caller says what a failure is to it: the fetcher reads `.failure`
    # (an issue), a one-shot caller takes `.logged()` (a log line, no rows) or `.required()` (DockerError).

    @staticmethod
    def list_containers(
        all_containers: bool = True, include_size: bool = False, timeout: float = QUERY_TIMEOUT
    ) -> Listing:
        """`docker ps`.

        `--size=false` matters: with the `{{json .}}` template the docker CLI otherwise computes every
        container's disk usage, which walks their filesystems and turns a 75ms `ps` into seconds: asked for,
        it gets SIZE_TIMEOUT whatever `timeout` says."""
        args = ["ps", "--format", "{{json .}}"]
        if all_containers:
            args.insert(1, "-a")
        if not include_size:
            args.insert(1, "--size=false")
        return Docker._list(args, Container.from_dict, SIZE_TIMEOUT if include_size else timeout)

    @staticmethod
    def list_images(timeout: float = QUERY_TIMEOUT) -> Listing:
        return Docker._list(["images", "--format", "{{json .}}"], Image.from_dict, timeout)

    @staticmethod
    def list_volumes(timeout: float = QUERY_TIMEOUT) -> Listing:
        return Docker._list(["volume", "ls", "--format", "{{json .}}"], Volume.from_dict, timeout)

    @staticmethod
    def list_networks(timeout: float = QUERY_TIMEOUT) -> Listing:
        return Docker._list(["network", "ls", "--format", "{{json .}}"], Network.from_dict, timeout)

    @staticmethod
    def listings(timeout: float = QUERY_TIMEOUT) -> dict[str, Listing]:
        """The four listings, by tab key (see LISTINGS). Each is its own call with its own outcome, a timeout
        included: one failing says nothing about the next, so all four always run."""
        return {
            "containers": Docker.list_containers(timeout=timeout),
            "images": Docker.list_images(timeout),
            "volumes": Docker.list_volumes(timeout),
            "networks": Docker.list_networks(timeout),
        }

    @staticmethod
    def start(container_id: str) -> bool:
        """Start a container"""
        if not is_valid_id(container_id):
            log_error("docker.invalid_id", f"Invalid container ID: {container_id}")
            return False
        result = Docker._run(["start", container_id])
        return result.returncode == 0

    @staticmethod
    def stop(container_id: str) -> bool:
        """Stop a container"""
        if not is_valid_id(container_id):
            log_error("docker.invalid_id", f"Invalid container ID: {container_id}")
            return False
        result = Docker._run(["stop", container_id])
        return result.returncode == 0

    @staticmethod
    def restart(container_id: str) -> bool:
        """Restart a container"""
        if not is_valid_id(container_id):
            log_error("docker.invalid_id", f"Invalid container ID: {container_id}")
            return False
        result = Docker._run(["restart", container_id])
        return result.returncode == 0

    @staticmethod
    def remove_container(container_id: str, force: bool = False) -> bool:
        """Remove a container"""
        if not is_valid_id(container_id):
            log_error("docker.invalid_id", f"Invalid container ID: {container_id}")
            return False
        args = ["rm", container_id]
        if force:
            args.insert(1, "-f")
        result = Docker._run(args)
        return result.returncode == 0

    @staticmethod
    def remove_image(image_id: str, force: bool = False) -> bool:
        """Remove an image"""
        if not is_valid_image_ref(image_id):
            log_error("docker.invalid_id", f"Invalid image ID: {image_id}")
            return False
        args = ["rmi", image_id]
        if force:
            args.insert(1, "-f")
        result = Docker._run(args)
        return result.returncode == 0

    @staticmethod
    def image_exists(ref: str) -> bool:
        """Is the image present locally?"""
        if not is_valid_image_ref(ref):
            log_error("docker.invalid_id", f"Invalid image ref: {ref}")
            return False
        args = ["image", "inspect", ref]
        # DockerError when docker could not say: ensure_image would build or pull over an image that is there
        return not Docker._absent(args, Docker._run(args, timeout=QUERY_TIMEOUT))

    @staticmethod
    def build_image(image: "ImageConfig") -> bool:
        """`docker build` a recipe as the next <repo>:cctl-local-NNNNNN and re-point the recipe tag at it, streaming
        the output. The context dir is sent when it exists; otherwise the Dockerfile goes in on stdin, so a recipe
        that copies nothing sends nothing."""
        recipe = image.recipe_hash()
        if recipe is None:
            return False
        # Fresh: another terminal may just have built one. DockerError when the listing failed: numbering from "no
        # versions" would re-point cctl-local-000000 at this build.
        versions, _, _ = local_versions(image, Docker.list_images().required())
        version = image.version_tag(max(versions, default=-1) + 1)
        labels = ["--label", f"{LABEL_IMAGE}={image.name}", "--label", f"{LABEL_RECIPE}={recipe}"]
        args = ["build", "-t", version, "-t", image.tag, *labels]
        if image.has_context:
            result = Docker._run(args + ["-f", str(image.dockerfile), str(image.context)], capture=False)
        else:
            with open(image.dockerfile, "rb") as f:
                result = Docker._run(args + ["-"], capture=False, stdin=f)
        if result.returncode != 0:
            log_error("image.build", f"docker build failed for {version} (exit {result.returncode})")
            return False
        print(f"Built {version}, {image.tag} now points at it")
        return True

    @staticmethod
    def pull_image(ref: str) -> bool:
        """`docker pull`, streaming the progress"""
        if not is_valid_image_ref(ref):
            log_error("docker.invalid_id", f"Invalid image ref: {ref}")
            return False
        return Docker._run(["pull", ref], capture=False).returncode == 0

    @staticmethod
    def ensure_image(ref: str) -> bool:
        """Before `docker run`: get the image when it is absent. Built from its recipe when one claims the tag,
        pulled otherwise: here rather than left to `docker run`, whose output we capture, so the progress shows.
        Never rebuilds or re-pulls an existing image: `image build` is the explicit way."""
        if not is_valid_image_ref(ref):  # image_exists says False to that too, and False here means "get it"
            log_error("docker.invalid_id", f"Invalid image ref: {ref}")
            return False
        if Docker.image_exists(ref):
            return True
        image = find_image_recipe(ref, load_images())
        if image is None:
            print(f"Pulling {ref}...")
            return Docker.pull_image(ref)
        print(f"Building {image.tag} from {image.dockerfile}...")
        return Docker.build_image(image)

    @staticmethod
    def remove_volume(volume_name: str) -> bool:
        """Remove a volume"""
        if not is_valid_id(volume_name):
            log_error("docker.invalid_id", f"Invalid volume name: {volume_name}")
            return False
        result = Docker._run(["volume", "rm", volume_name])
        return result.returncode == 0

    @staticmethod
    def remove_network(network_id: str) -> bool:
        """Remove a network"""
        if not is_valid_id(network_id):
            log_error("docker.invalid_id", f"Invalid network ID: {network_id}")
            return False
        result = Docker._run(["network", "rm", network_id])
        return result.returncode == 0

    @staticmethod
    def ensure_network(name: str) -> bool:
        """Ensure a Docker network exists (create if missing)"""
        if not is_valid_id(name):
            log_error("docker.invalid_id", f"Invalid network name: {name}")
            return False
        args = ["network", "inspect", name]
        # DockerError when docker could not say: the create below would fail on "already exists"
        if not Docker._absent(args, Docker._run(args, timeout=QUERY_TIMEOUT)):
            return True
        result = Docker._run(["network", "create", name])
        if result.returncode != 0:
            log_error("docker.network_create", f"Failed to create network {name}: {result.stderr.strip()}")
            return False
        return True

    @staticmethod
    def connect_network(network: str, container: str) -> bool:
        """Connect a container to a network"""
        if not is_valid_id(network) or not is_valid_id(container):
            log_error("docker.invalid_id", f"Invalid network/container: {network}/{container}")
            return False
        result = Docker._run(["network", "connect", network, container])
        return result.returncode == 0

    @staticmethod
    def disconnect_network(network: str, container: str) -> bool:
        """Disconnect a container from a network"""
        if not is_valid_id(network) or not is_valid_id(container):
            log_error("docker.invalid_id", f"Invalid network/container: {network}/{container}")
            return False
        result = Docker._run(["network", "disconnect", network, container])
        return result.returncode == 0

    @staticmethod
    def logs(container_id: str, follow: bool = False, tail: int | None = None) -> int:
        """Show container logs (streams to stdout); docker's exit code, 2 when we refused the call"""
        if not is_valid_id(container_id):
            log_error("docker.invalid_id", f"Invalid container ID: {container_id}")
            return REFUSED
        args = ["logs"]
        if follow:
            args.append("-f")
        if tail:
            args.extend(["--tail", str(tail)])
        args.append(container_id)
        return Docker._stream(args)

    @staticmethod
    def exec(
        container_id: str,
        command: list[str],
        interactive: bool = True,
        user: str | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> int:
        """Execute command in container; its exit code (docker hands the command's through), 2 when we refused
        the call.

        `env` is passed as bare `-e NAME` flags: docker fills each value from the client's environment,
        which we extend for this one call, so a secret never appears in argv (`ps`) or on any disk."""
        if not is_valid_id(container_id):
            log_error("docker.invalid_id", f"Invalid container ID: {container_id}")
            return REFUSED
        if user is not None and (not user or not is_valid_id(user)):
            log_error("docker.invalid_user", f"Invalid user: {user}")
            return REFUSED
        if workdir is not None and (not workdir or not all(c in VALID_PATH_CHARS for c in workdir)):
            log_error("docker.invalid_workdir", f"Invalid workdir: {workdir}")
            return REFUSED
        for name in env or {}:
            if not is_valid_env_name(name):
                log_error("docker.invalid_env", f"Invalid env name: {name}")
                return REFUSED
        args = ["exec"]
        if interactive:
            # -i: our stdin is theirs. -t: a terminal, when we have one to lend. Without (a pipe, cron, CI) docker
            # would refuse -t ("the input device is not a TTY"); the output comes through either way.
            args.append("-it" if stdin_is_tty() else "-i")
        if user is not None:
            args.extend(["-u", user])
        if workdir is not None:
            args.extend(["-w", workdir])
        for name in env or {}:
            args.extend(["-e", name])
        args.append(container_id)
        args.extend(command)
        return Docker._stream(args, env=env)

    @staticmethod
    def shell(container_id: str, shell: str = "/bin/sh") -> int:
        """Open interactive shell in container; see exec for what comes back"""
        return Docker.exec(container_id, [shell], interactive=True)

    @staticmethod
    def inspect(cls: type[Inspected], ref: str):
        """`docker inspect` one object as `cls` (ContainerInspect, ImageInspect, ...); None when there is no such
        object, or (after logging) when `ref` or the answer is not something we can use. DockerError when docker
        could not say (no answer, daemon down, ...): that is not "no such object", and every caller reads None
        as one. `--type` keeps a container and a network of the same name apart."""
        if not cls.VALID(ref):
            log_error("docker.invalid_id", f"Invalid {cls.TYPE}: {ref}")
            return None
        args = ["inspect", "--type", cls.TYPE, ref]
        result = Docker._run(args, timeout=QUERY_TIMEOUT)
        if Docker._absent(args, result):
            return None
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            log_error(cls.CAT, f"docker {' '.join(args)}: not JSON", {"data": result.stdout[:200]})
            return None
        return cls.from_dict(data[0]) if data else None

    @staticmethod
    def investigate_source(container_id: str) -> dict:
        """Investigate container source (compose, systemd, devcontainer, etc.)"""
        if not is_valid_id(container_id):
            log_error("docker.invalid_id", f"Invalid container ID: {container_id}")
            return {"source": "error", "details": ["Invalid container ID"]}
        result = {"source": "cli", "details": []}

        # Get container info
        info = Docker.inspect(ContainerInspect, container_id)
        if not info:
            return result

        container_name = info.name_clean
        labels = info.labels

        # Check Docker Compose
        if "com.docker.compose.project" in labels:
            result["source"] = "docker-compose"
            result["details"].append(f"Project: {labels.get('com.docker.compose.project')}")
            result["details"].append(f"Service: {labels.get('com.docker.compose.service')}")
            if "com.docker.compose.project.working_dir" in labels:
                result["details"].append(f"Working dir: {labels.get('com.docker.compose.project.working_dir')}")
            if "com.docker.compose.project.config_files" in labels:
                for f in labels.get("com.docker.compose.project.config_files", "").split(","):
                    if f and not f.startswith("/tmp/"):
                        result["details"].append(f"Config: {f}")

        # Check Devcontainer
        if "devcontainer.config_file" in labels:
            result["source"] = "devcontainer"
            result["details"].append(f"Config: {labels.get('devcontainer.config_file')}")
            if "devcontainer.local_folder" in labels:
                result["details"].append(f"Folder: {labels.get('devcontainer.local_folder')}")

        # Check Kubernetes
        if "io.kubernetes.pod.name" in labels:
            result["source"] = "kubernetes"
            result["details"].append(f"Pod: {labels.get('io.kubernetes.pod.name')}")
            result["details"].append(f"Namespace: {labels.get('io.kubernetes.pod.namespace')}")

        # Check Portainer
        if "io.portainer.accesscontrol.public" in labels or any(k.startswith("io.portainer") for k in labels):
            result["source"] = "portainer"
            result["details"].append("Managed by Portainer")

        # Systemd heuristic scan (only if no other source found)
        if result["source"] == "cli":
            systemd_paths = ["/etc/systemd/system", os.path.expanduser("~/.config/systemd/user")]
            for spath in systemd_paths:
                if not os.path.isdir(spath):
                    continue
                try:
                    for fname in os.listdir(spath):
                        if not fname.endswith(".service"):
                            continue
                        fpath = os.path.join(spath, fname)
                        try:
                            with open(fpath, "r") as f:
                                content = f.read()
                                # Look for container name in docker run --name or docker start
                                if f"--name {container_name}" in content or f"--name={container_name}" in content:
                                    result["source"] = "systemd"
                                    result["details"].append(f"Unit: {fpath}")
                                    break
                                if f"docker start {container_name}" in content:
                                    result["source"] = "systemd"
                                    result["details"].append(f"Unit: {fpath}")
                                    break
                        except (IOError, PermissionError) as e:
                            log_error("source.systemd_read", str(e), {"path": fpath})
                    if result["source"] == "systemd":
                        break
                except (IOError, PermissionError) as e:
                    log_error("source.systemd_list", str(e), {"path": spath})

        # If still cli, add a note
        if result["source"] == "cli" and not result["details"]:
            result["details"].append("No managed source detected")
            result["details"].append("Likely started via: docker run ...")

        return result

    @staticmethod
    def containers_by_name() -> dict[str, Container]:
        """One `docker ps -a`, keyed by container name. DockerError when it failed: a caller asking whether a
        container exists must not be told "no" by a call that never got an answer."""
        return {c.Names: c for c in Docker.list_containers().required()}

    @staticmethod
    def find_project_container(project_name: str) -> Container | None:
        """Find a container belonging to a project by its cctl- name; None when there is none, DockerError
        when docker couldn't say"""
        return Docker.containers_by_name().get(f"{CONTAINER_PREFIX}{project_name}")

    @staticmethod
    def project_containers(project: "ProjectConfig") -> tuple[Container | None, Container | None]:
        """(the project's container, what an unfinished spawn of it left), each None when there is none; one `ps`"""
        existing = Docker.containers_by_name()
        return existing.get(project.container_name), existing.get(project.spawning_name)

    @staticmethod
    def create_project_container(project: "ProjectConfig", name: str | None = None) -> str | None:
        """Create and start a container for a project, as `name` (spawn's temporary one) when given. Its ID, as
        `docker run -d` prints it (its name, should docker print nothing usable); None when it failed."""
        network_name = f"{NETWORK_PREFIX}{project.name}"
        if not Docker.ensure_network(network_name):
            log_error("project.create", f"Failed to ensure network {network_name}", echo=True)
            return None
        if not Docker.ensure_image(project.image):
            log_error("project.create", f"Failed to build or pull image {project.image}", echo=True)
            return None
        args = [
            "run",
            "-d",
            "--init",  # PID 1 is `sleep infinity`, which never reaps the children `docker exec` leaves behind
            "--name",
            name or project.container_name,
            "--network",
            network_name,
            "--label",
            f"{LABEL_PROJECT}={project.name}",
        ]
        if project.repos:
            args.extend(["--label", f"{LABEL_REPO}={','.join(r.url for r in project.repos)}"])
        args.extend([project.image, "sleep", "infinity"])
        result = Docker._run(args)
        if result.returncode != 0:
            log_error("project.create", f"Failed to create container: {result.stderr.strip()}", echo=True)
            return None
        container_id = result.stdout.strip()
        return container_id if is_valid_id(container_id) else name or project.container_name

    @staticmethod
    def cp(src: str, container_id: str, dest: str) -> bool:
        """Copy file into container"""
        result = Docker._run(["cp", src, f"{container_id}:{dest}"])
        if result.returncode != 0:
            log_error("docker.cp", f"Failed to copy {src} to {container_id}:{dest}: {result.stderr.strip()}")
            return False
        return True

    @staticmethod
    def exec_noninteractive(
        container_id: str,
        command: list[str],
        workdir: str | None = None,
        user: str | None = None,
        input: str | None = None,
        capture: bool = True,
    ) -> subprocess.CompletedProcess:
        """Execute command in container non-interactively; `input` is fed to its stdin. Output is captured,
        unless `capture` is off: it then goes to our terminal as it comes (a clone's progress).

        Like `exec`, checks what becomes a docker option or name here too, wherever the values came from: a
        refused call is a result with the REFUSED exit code, and the reason in the log."""
        refused = None
        if not is_valid_id(container_id):
            refused = ("docker.invalid_id", f"Invalid container ID: {container_id}")
        elif user is not None and not is_valid_id(user):
            refused = ("docker.invalid_user", f"Invalid user: {user}")
        elif workdir is not None and not is_valid_home(workdir):
            refused = ("docker.invalid_workdir", f"Invalid workdir: {workdir}")
        if refused:
            log_error(*refused)
            return subprocess.CompletedProcess(["docker", "exec"], REFUSED, "", refused[1])
        args = ["exec"]
        if input is not None:
            args.append("-i")
        if user is not None:
            args.extend(["-u", user])
        if workdir is not None:
            args.extend(["-w", workdir])
        args.append(container_id)
        return Docker._run(args + command, input=input, capture=capture)

    @staticmethod
    def spawn_project(project: "ProjectConfig") -> bool:
        """Full spawn flow: create the container, put everything in it (see _provision), and only then give it
        its project name. Nothing here cleans up after an interruption: an unfinished spawn keeps its spawning
        name, which is all the next one needs to know (see SPAWNING_PREFIX)."""
        name = project.spawning_name
        # Before anything is created: finding out after the clones would cost a whole spawn
        wanted = ([project.ssh_key] if project.ssh_key else []) + [ef.src for ef in project.env_files]
        if missing := [path for path in wanted if not os.path.isfile(path)]:
            log_error("project.spawn", f"not found on this host: {', '.join(missing)}. Nothing spawned.", echo=True)
            return False
        if leftover := Docker.containers_by_name().get(name):
            # It may also be a spawn still running in another terminal: only the user knows, so the user says
            print(
                f"{name} exists ({created_ago(leftover)}): a spawn still running elsewhere, or one that never finished."
            )
            if not confirm_cli("Remove it and start over?"):
                print(f"Not removed: nothing spawned (to remove it yourself: docker rm -f {name})", file=sys.stderr)
                return False
            if not Docker.remove_container(leftover.ID, force=True):  # by ID: the one the question was about
                log_error("project.spawn", f"Could not remove {name}", echo=True)
                return False
        print(f"Creating container {project.container_name}...")
        # From here on by ID: the spawning name may change hands (another terminal said yes to "start over"), and
        # what we put in, rename or clean up must be the container we made
        container_id = Docker.create_project_container(project, name=name)
        if container_id is None:
            return False
        if not Docker._provision(project, container_id):
            return False
        result = Docker._run(["rename", container_id, project.container_name])
        if result.returncode != 0:  # it stays under its spawning name, complete: the Issues tab says what to do
            msg = f"Could not rename {name} to {project.container_name}: {result.stderr.strip()}"
            log_error("project.spawn", msg, echo=True)
            return False
        print(f"Container {project.container_name} ready!")
        return True

    @staticmethod
    def _provision(project: "ProjectConfig", name: str) -> bool:
        """What spawn puts in the fresh container `name` (its ID: see spawn_project): SSH key, forge host keys,
        clones, env files, containerctl. Any of it failing fails the spawn, and the container is removed: the
        project name is only given to a container that has all of it."""

        def fail(msg: str) -> bool:
            log_error("project.spawn", f"{project.name}: {msg}", echo=True)
            print(f"Cleaning up container {project.spawning_name}...", file=sys.stderr)
            Docker.remove_container(name, force=True)
            return False

        # Copy SSH key if configured
        if project.ssh_key:
            if not os.path.isfile(project.ssh_key):
                return fail(f"SSH key not found: {project.ssh_key}")
            print("Injecting SSH key...")
            ssh_dir, user = f"{project.home}/.ssh", project.user
            key = f"{ssh_dir}/id_ed25519"
            Docker.exec_noninteractive(name, ["mkdir", "-p", ssh_dir], user=user)
            if not Docker.cp(project.ssh_key, name, key):
                return fail("failed to copy SSH key into container")
            Docker.exec_noninteractive(name, ["chmod", "600", key], user=ROOT)
            Docker.exec_noninteractive(name, ["chown", f"{user}:", key], user=ROOT)  # "user:" = its own group
            # Verify the key is there for who will use it: this is also where a wrong `user` or `home` shows
            result = Docker.exec_noninteractive(name, ["test", "-r", key], user=user)
            if result.returncode != 0:
                return fail(f"SSH key not readable by {user} inside container: {result.stderr.strip()}")
            # Pinned forge host keys (see KNOWN_HOSTS), appended as the user so ssh can keep adding to the file
            result = Docker.exec_noninteractive(
                name, ["tee", "-a", f"{ssh_dir}/known_hosts"], user=project.user, input=KNOWN_HOSTS
            )
            if result.returncode != 0:
                return fail(f"could not write known_hosts: {result.stderr.strip()}")

        # Clone repos if configured (each into its own dir under the workspace)
        if project.repos:
            Docker.exec_noninteractive(name, ["mkdir", "-p", project.workspace], user=project.user)
            for repo in project.repos:
                print(f"Cloning {repo.url}...")
                clone = ["git", "clone", "--progress"]  # not captured: a long clone shows it is alive
                clone += CLONE_MIRRORS_TAGS
                if repo.branch:
                    clone += ["-b", repo.branch]
                # `--` ends option parsing so a URL starting with `-` can't be read as a git option
                clone.append("--")
                clone.append(repo.url)
                if repo.dir:
                    clone.append(repo.dir)
                result = Docker.exec_noninteractive(
                    name, clone, workdir=project.workspace, user=project.user, capture=False
                )
                if result.returncode != 0:
                    return fail(f"git clone failed for {repo.url}")  # git just said why

        # Copy env files
        for ef in project.env_files:
            if os.path.isfile(ef.src):
                print(f"Injecting {ef.dest}...")
                # Ensure destination directory exists
                dest_dir = os.path.dirname(ef.dest)
                # Under the home it is the user's, like the rest; anywhere else (/etc/...) it is root's business
                theirs = ef.dest.startswith(f"{project.home}/")
                Docker.exec_noninteractive(name, ["mkdir", "-p", dest_dir], user=project.user if theirs else ROOT)
                if not Docker.cp(ef.src, name, ef.dest):
                    return fail(f"failed to copy {ef.src} to {ef.dest}")
                if theirs:
                    Docker.exec_noninteractive(name, ["chown", f"{project.user}:", ef.dest], user=ROOT)
            else:  # it was there when the spawn started
                return fail(f"env file {ef.src} not found")

        # Install containerctl + the expected-repos manifest so `containerctl
        # check-local` is available from inside the container.
        repo_names = [r.name for r in project.repos]
        if not install_self(name, repo_names, project.user, project.home, injected_files(project)):
            return fail(
                f"failed to install containerctl under {project.home} as {project.user}: when the image has"
                " another user or home, say so in the project file (`user`, `home`)"
            )

        return True

    @staticmethod
    def enter_project(project: "ProjectConfig", user: str | None = None, workdir: str | None = None) -> int | None:
        """Enter a project: spawn if needed, start if stopped, then attach a shell. The shell's exit code (see
        exec), or None when no shell was entered: the spawn failed or was declined, and has said why. What to
        do with that is the caller's: the CLI exits with it, the TUI stays up.

        `workdir` (a shortcut target) overrides the default landing directory; `user` who the shell is for
        (the project's user unless said: never the image's default, see CONTAINER_USER)."""
        container = Docker.find_project_container(project.name)

        if container is None:
            # No container — spawn it
            if not Docker.spawn_project(project):
                print(f"Failed to spawn project {project.name}", file=sys.stderr)
                return None
        else:
            state = container.State.lower()
            if state != "running":
                print(f"Starting {project.container_name}...")
                if not Docker.start(project.container_name):
                    log_error("project.enter", f"Could not start {project.container_name}", echo=True)
                    return None
            # Keep the in-container checker + manifest fresh (config may have changed,
            # or the container predates this feature). Best-effort.
            repo_names = [r.name for r in project.repos]
            install_self(project.container_name, repo_names, project.user, project.home, injected_files(project))

        # Pick the landing directory:
        #   - exactly one repo  -> enter that repo's dir
        #   - multiple repos     -> enter the shared workspace root
        #   - no repo            -> home
        if workdir is None:
            workdir = project.home
            if len(project.repos) == 1:
                repo_name = project.repos[0].name
                if repo_name and all(c in VALID_REPO_NAME_CHARS for c in repo_name):
                    workdir = f"{project.workspace}/{repo_name}"
            elif len(project.repos) > 1:
                workdir = project.workspace
        # [env] values and [secrets] looked up in secrets.toml ride into this shell's process environment
        # only. Their problems are echoed right before the prompt, where they matter (they stay in the log too).
        seen = len(get_errors())
        env = resolve_env(project)
        for problem in get_errors()[seen:]:  # what resolve_env just logged: this run's only logger is this thread
            if problem.cat.startswith("env."):
                print(f"env: {problem.msg}", file=sys.stderr)
        # cd into the landing dir if it exists, else fall back home - done atomically inside the
        # container (no TOCTOU, unlike a separate `test -d` probe then `-w`). workdir and home ride in
        # as $1 and $2 (data, never interpolated as code); `--` stops cd option-parsing a leading dash.
        return Docker.exec(
            project.container_name,
            ["bash", "-c", 'cd -- "$1" 2>/dev/null || cd -- "$2"; exec bash', "bash", workdir, project.home],
            user=user or project.user,
            env=env,
        )

    @staticmethod
    def destroy_project(project: "ProjectConfig", names: list[str] | None = None) -> bool:
        """Stop and remove a project's containers. `names` are the ones the user was asked about: only those go,
        by name, whatever appeared since (a spawn started meanwhile was in no question; one that finished
        meanwhile no longer has the name that was agreed to). Unset: all that exist now, an unfinished spawn's
        leftover included."""
        if names is None:
            names = [c.Names for c in Docker.project_containers(project) if c]
        if not names:
            print(f"No container found for project {project.name}")
            return False
        for name in names:
            print(f"Destroying {name}...")
            if not Docker.remove_container(name, force=True):
                return False
        # Best-effort network cleanup
        network_name = f"{NETWORK_PREFIX}{project.name}"
        Docker.remove_network(network_name)
        return True

    @staticmethod
    def find_service_container(service_name: str) -> Container | None:
        """Find a container belonging to a service by its cctl-svc- name; None when there is none, DockerError
        when docker couldn't say"""
        return Docker.containers_by_name().get(f"{SERVICE_PREFIX}{service_name}")

    @staticmethod
    def create_service_container(service: "ServiceConfig") -> bool:
        """Create and start a container for a service"""
        # `docker run` takes a single --network; the rest are connected right after creation
        for network in service.networks:
            if not Docker.ensure_network(network):
                log_error("service.create", f"Failed to ensure network {network}")
                return False
        if not Docker.ensure_image(service.image):
            log_error("service.create", f"Failed to build or pull image {service.image}")
            return False
        args = [
            "run",
            "-d",
            "--name",
            service.container_name,
            "--restart",
            service.restart,
            "--label",
            f"{LABEL_SERVICE}={service.name}",
        ]
        if service.networks:
            args.extend(["--network", service.networks[0]])
        if service.read_only:
            args.append("--read-only")
        for t in service.tmpfs:
            args.extend(["--tmpfs", t])
        for c in service.cap_drop:
            args.extend(["--cap-drop", c])
        for c in service.cap_add:
            args.extend(["--cap-add", c])
        if service.pids_limit:
            args.extend(["--pids-limit", str(service.pids_limit)])
        for port in service.ports:
            args.extend(["-p", port])
        for vol in service.volumes:
            args.extend(["-v", vol])
        for e in service.env:
            args.extend(["-e", e])
        args.append(service.image)
        args.extend(service.command)
        result = Docker._run(args)
        if result.returncode != 0:
            log_error(
                "service.create",
                f"Failed to create container: {result.stderr.strip()}",
            )
            return False
        return Docker.apply_service_networks(service, already=set(service.networks[:1]))

    @staticmethod
    def apply_service_networks(service: "ServiceConfig", already: set[str] | None = None) -> bool:
        """Connect the service container to every configured network it is not on yet.

        `already` skips the inspect (used right after `run --network`). Networks the container
        is on but the config no longer lists are left alone: `recreate` is the way to drop them."""
        if already is None:
            info = Docker.inspect(ContainerInspect, service.container_name)
            if info is None:
                return False
            already = set(((info.NetworkSettings or {}).get("Networks") or {}).keys())
        ok = True
        for network in service.networks:
            if network in already:
                continue
            if not Docker.ensure_network(network) or not Docker.connect_network(network, service.container_name):
                log_error("service.network", f"Failed to connect {service.container_name} to {network}")
                ok = False
        return ok

    @staticmethod
    def service_up(service: "ServiceConfig") -> bool:
        """Bring a service up: create if missing, start if stopped"""
        container = Docker.find_service_container(service.name)
        if container is None:
            print(f"Creating service {service.container_name}...")
            return Docker.create_service_container(service)
        state = container.State.lower()
        if state != "running":
            print(f"Starting {service.container_name}...")
            if not Docker.start(service.container_name):
                return False
        else:
            print(f"Service {service.container_name} is already running")
        # Networks added to the config since creation get connected; the hardening flags can't
        # be changed on a live container, use `service recreate` for those.
        return Docker.apply_service_networks(service)

    @staticmethod
    def service_down(service: "ServiceConfig") -> bool:
        """Stop and remove a service container"""
        container = Docker.find_service_container(service.name)
        if container is None:
            print(f"No container found for service {service.name}")
            return False
        print(f"Removing {service.container_name}...")
        return Docker.remove_container(service.container_name, force=True)

    @staticmethod
    def service_recreate(service: "ServiceConfig") -> bool:
        """Recreate a service container (down + up)"""
        container = Docker.find_service_container(service.name)
        if container is not None:
            print(f"Removing {service.container_name}...")
            if not Docker.remove_container(service.container_name, force=True):
                return False
        print(f"Creating {service.container_name}...")
        return Docker.create_service_container(service)


# =============================================================================
# Background Data Fetcher
# =============================================================================


@dataclass
class ProjectRow:
    """A line of the Projects tab"""

    Name: str
    Status: str
    Image: str
    Repo: str


@dataclass
class ServiceRow:
    """A line of the Services tab"""

    Name: str
    Status: str
    Image: str
    Ports: str


def container_status(container: Container | None) -> str:
    return (container.State or container.Status) if container else "not created"


def status_lookup(ps: Listing) -> Callable[[str], str]:
    """Container name -> what a STATUS column says, from one `docker ps` outcome. When it failed nobody knows:
    "unknown" for every name, never "not created"."""
    if ps.failure:
        return lambda name: "unknown"
    by_name = {c.Names: c for c in ps.rows}
    return lambda name: container_status(by_name.get(name))


class DataFetcher:
    """Background thread that fetches Docker data every N seconds"""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self.lock = threading.Lock()
        self.data: dict[str, list] = {
            "containers": [],
            "images": [],
            "volumes": [],
            "networks": [],
            "projects": [],
            "services": [],
            "issues": [],  # list[Issue]: what the last fetch detected, errors first
        }
        self.version = 0  # how many fetches landed: the TUI redraws when it moves
        self.failures: dict[str, str] = {}  # tab key -> why the last fetch has nothing to show there (our words)
        self.notes: dict[str, str] = {}  # tab key -> why the rows listed there may not be all of them
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start background fetching"""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Initial fetch (blocking) so we have data immediately
        self._fetch_all()

    def stop(self) -> None:
        """Stop background fetching"""
        self._stop_event.set()
        Docker.kill_own_group()  # the listing it may be waiting on: quitting doesn't wait for a docker that hangs
        if self._thread:
            self._thread.join(timeout=1.0)

    def get(self, key: str) -> list:
        """Get cached data for a key (thread-safe)"""
        with self.lock:
            return self.data.get(key, []).copy()

    def get_failures(self) -> dict[str, str]:
        """Tab key -> why its listing failed, as of the last fetch (thread-safe)"""
        with self.lock:
            return dict(self.failures)

    def get_notes(self) -> dict[str, str]:
        """Tab key -> what is missing from the rows it lists, as of the last fetch (thread-safe)"""
        with self.lock:
            return dict(self.notes)

    def _fetch_all(self) -> None:
        """One fetch, as a unit: rows, failures and issues are swapped in together. What it finds wrong (docker
        down, a failed listing, unreadable rows, a broken config) is state: issues and panels show it while it
        lasts, and none of it goes to the error log. A crash is the one thing it logs, every tick it happens: that
        is not a backend being unreliable, it is a bug of ours, and a bug should be loud until it is fixed. Every
        panel says so too, until a fetch succeeds."""
        notes: dict[str, str] = {}
        try:
            data, listings = self._fetch()
            shown = {key: listing.failure.msg for key, listing in listings.items() if listing.failure}
            for key, listing in listings.items():
                if count := len(listing.skipped):
                    notes[key] = f"{count} unreadable row{'s' if count > 1 else ''} left out: see Issues"
        except Exception as e:  # a crashed fetch must not kill the thread, nor hide behind a view that looks fine
            crash = f"{type(e).__name__}: {e}"
            log_error("fetcher", crash)
            data = {key: [] for key in self.data}
            data["issues"] = [Issue(ERROR, "fetcher", f"Fetch crashed: {crash}", "a bug of ours: please report it")]
            shown = {key: f"Fetch crashed: {crash}" for key in self.data if key != "issues"}
        with self.lock:
            self.data.update(data)
            self.failures = shown
            self.notes = notes
            self.version += 1

    def _fetch(self) -> tuple[dict[str, list], dict[str, Listing]]:
        """Fetch all data from Docker and the config files, and detect the issues in them"""
        listings = Docker.listings(timeout=POLL_TIMEOUT)
        projects_config, broken_projects = read_projects()
        services_config, broken_services = read_services()
        images_config, broken_images = read_images()
        issues = broken_projects + broken_services + broken_images
        issues += detect_issues(projects_config, services_config, images_config, listings)

        # Projects and services come from config, so they list whatever docker does; only their status needs `ps`
        status = status_lookup(listings["containers"])
        projects_data = [
            ProjectRow(name, status(p.container_name), p.image, p.repos_display) for name, p in projects_config.items()
        ]
        services_data = []
        for name, service in services_config.items():
            ports = ", ".join(service.ports[:3])
            if len(service.ports) > 3:
                ports += f" +{len(service.ports) - 3}"
            services_data.append(ServiceRow(name, status(service.container_name), service.image, ports))

        data = {key: listing.rows for key, listing in listings.items()}
        data.update(projects=projects_data, services=services_data, issues=by_severity(issues))
        return data, listings

    def _run(self) -> None:
        """Background thread loop"""
        while not self._stop_event.is_set():
            self._stop_event.wait(self.interval)
            if not self._stop_event.is_set():
                self._fetch_all()


# =============================================================================
# UI Components
# =============================================================================


@dataclass
class Tab:
    """Tab definition"""

    name: str
    key: str
    columns: list[tuple[str, str, int]]  # (key, header, width)
    id_key: str
    empty: str = ""  # what the list area says when there are no rows; "No <name> found" when unset
    anchor: tuple[str, ...] = ()  # the fields that name a row from one fetch to the next; (id_key,) when unset


class UI:
    """UI rendering components - all methods write to a Screen buffer"""

    @staticmethod
    def tab_bar(
        scr: Screen, tabs: list[Tab], selected: int, width: int, row: int, alerts: dict[str, str] | None = None
    ) -> None:
        """Render tab bar; `alerts` colors the tabs it names (by key) with the given Term color"""
        scr.move(row, 1)
        scr.clear_line()

        x = 1
        for i, tab in enumerate(tabs):
            if color := (alerts or {}).get(tab.key):
                if i == selected:
                    text = Term.style(f" {tab.name} ", Term.BOLD, Term.INVERT, color)
                else:
                    text = Term.style(f" {tab.name} ", color)
            elif i == selected:
                text = Term.style(f" {tab.name} ", Term.BOLD, Term.INVERT)
            else:
                text = Term.style(f" {tab.name} ", Term.DIM)
            scr.move(row, x)
            scr.write(text)
            x += len(tab.name) + 2

    @staticmethod
    def divider(scr: Screen, row: int, width: int, char: str = "─") -> None:
        """Render horizontal divider"""
        scr.writeln(row, 1, Term.style(char * width, Term.DIM))

    @staticmethod
    def list_header(scr: Screen, tab: Tab, row: int, width: int) -> None:
        """Render list header"""
        scr.move(row, 1)
        scr.clear_line()

        line = ""
        remaining_width = width - 2
        for key, header, col_width in tab.columns:
            if col_width == 0:  # Flex column: whatever is left, none on a terminal too narrow for the fixed ones
                col_width = max(remaining_width, 0)
            text = header[:col_width].ljust(col_width)
            line += text + " "
            remaining_width -= col_width + 1

        scr.move(row, 2)
        scr.write(Term.style(line, Term.BOLD, Term.CYAN))

    @staticmethod
    def list_row(
        scr: Screen,
        tab: Tab,
        item: object,
        row: int,
        width: int,
        selected: bool,
        status_colors: dict[str, str] | None = None,
    ) -> None:
        """Render a single list row"""
        scr.move(row, 1)
        scr.clear_line()

        line = ""
        remaining_width = width - 2

        for key, header, col_width in tab.columns:
            if col_width == 0:  # Flex column: whatever is left, none on a terminal too narrow for the fixed ones
                col_width = max(remaining_width, 0)

            value = str(getattr(item, key))
            text = value[:col_width].ljust(col_width)

            # Apply status colors (the map is keyed by status substring; it applies to the Status column)
            if status_colors and key == "Status":
                status = value.lower()
                for status_key, color in status_colors.items():
                    if status_key in status:
                        text = color + text + Term.RESET
                        break

            line += text + " "
            remaining_width -= col_width + 1

        scr.move(row, 2)
        if selected:
            scr.write(Term.style(line.rstrip(), Term.INVERT))
        else:
            scr.write(line.rstrip())

    @staticmethod
    def status_bar(scr: Screen, text: str, row: int, width: int) -> None:
        """Render status bar"""
        scr.move(row, 1)
        scr.clear_line()
        scr.write(Term.style(text[:width].ljust(width), Term.DIM, Term.INVERT))

    @staticmethod
    def message(scr: Screen, text: str, row: int, width: int, style: str = "") -> None:
        """Render a message line"""
        scr.move(row, 1)
        scr.clear_line()
        if style:
            scr.write(style + text[:width] + Term.RESET)
        else:
            scr.write(text[:width])

    @staticmethod
    def confirm(prompt: str, row: int, cols: int | None = None) -> bool:
        """Show confirmation prompt and get y/n response (flushes immediately). On line `row`; one that would not
        fit in `cols` (a DIRTY list, a leftover) is asked in a box instead, wrapped, and scrolling when tall."""
        if cols is not None and len(prompt) + len(" [y/N] ") > cols:
            lines = textwrap.wrap(prompt, max(20, cols - 8))
            return bool(UI.info_box("Confirm", lines, row + 1, cols, answer=True))
        sys.stdout.write(f"\033[{row};1H" + Term.CLEAR_LINE)
        sys.stdout.write(Term.style(f"{prompt} [y/N] ", Term.YELLOW))
        sys.stdout.flush()
        try:
            return UI._yes()
        except KeyboardInterrupt:  # Ctrl-C answers no: it closes the question, not the TUI
            return False

    @staticmethod
    def _yes() -> bool:
        """Read keys until one answers the y/N on screen"""
        while True:
            key = Term.getch()
            if key.lower() == "y":
                return True
            if key.lower() == "n" or key == Term.KEY_ENTER or key == Term.KEY_ESC:
                return False

    @staticmethod
    def info_box(
        title: str, lines: list[str], rows: int, cols: int, ok: bool = False, answer: bool = False
    ) -> bool | None:
        """Display an info box modal and wait for a keypress to dismiss it: any key, or with `ok` Enter alone (a
        warning must be read, and a key that was already on its way must not be what closes it). Content taller
        than the screen scrolls: i / k (or the arrows) move it a line up / down, and are then not keys that close.
        With `answer` it asks: y is True; n, Enter or Esc False (see confirm). Ctrl-C closes any of them, an `ok`
        one included, and answers no: it ends the modal that is open, not the TUI under it."""
        content_width = max(len(title), max((len(line) for line in lines), default=20)) + 4
        box_width = min(content_width, cols - 4)
        view = min(len(lines), max(1, rows - 6))  # content lines on screen at once: frame and hint must fit too
        tall = len(lines) > view
        start_row = max(1, (rows - (view + 4)) // 2)
        start_col = max(1, (cols - box_width) // 2)
        closing = "[Press Enter: ok]" if ok else "[Press any key to close]"
        if answer:
            closing = "[y: yes] [n / Enter: no]"
        out = sys.stdout.write

        def at(row: int) -> str:
            return f"\033[{start_row + row};{start_col}H"

        top = 0
        while True:
            out(at(0) + Term.style("┌" + "─" * (box_width - 2) + "┐", Term.CYAN))
            out(
                at(1)
                + Term.style("│ ", Term.CYAN)
                + Term.style(title[: box_width - 4].center(box_width - 4), Term.BOLD)
                + Term.style(" │", Term.CYAN)
            )
            out(at(2) + Term.style("├" + "─" * (box_width - 2) + "┤", Term.CYAN))
            for i, line in enumerate(lines[top : top + view]):
                display_line = line[: box_width - 4].ljust(box_width - 4)
                out(at(3 + i) + Term.style("│ ", Term.CYAN) + display_line + Term.style(" │", Term.CYAN))
            out(at(3 + view) + Term.style("└" + "─" * (box_width - 2) + "┘", Term.CYAN))
            hint = f"[i/k: scroll, {top + 1}-{top + view} of {len(lines)}] {closing}" if tall else closing
            out(at(4 + view) + Term.DIM + hint.center(box_width)[: cols - start_col + 1] + Term.RESET)
            sys.stdout.flush()

            try:
                key = Term.getch()
            except KeyboardInterrupt:
                return False if answer else None
            if tall and key in ("i", Term.KEY_UP):
                top = max(0, top - 1)
            elif tall and key in ("k", Term.KEY_DOWN):
                top = min(len(lines) - view, top + 1)
            elif answer:
                if key.lower() == "y":
                    return True
                if key.lower() == "n" or key in (Term.KEY_ENTER, Term.KEY_ESC):
                    return False
            elif not ok or key == Term.KEY_ENTER:
                return None


# =============================================================================
# Application
# =============================================================================


class App:
    """Main TUI application"""

    # When its row disappears, the cursor falls onto a neighbour by itself. A key that lands within this many
    # seconds may have been on its way to the row that was there (a finger is slower than a redraw): it does
    # nothing, a box says so, and it is for the user to press it again. Moving the cursor yourself, or the box's
    # ok, ends the wait.
    SWITCH_GRACE = 2.0
    # busy() invites a Ctrl-C; one pressed as the action ends lands in the main loop, where it would quit. For this
    # long after such an action, a Ctrl-C is taken as meant for it: said, and nothing more. The next one quits.
    LATE_CTRL_C = 1.0
    REDRAW_POLL = 0.2  # seconds between two looks at whether a fetch landed, while no key comes
    MOVE_KEYS = ("i", "k", "g", "G", "j", "l", *"123456789")
    MOVE_KEYS += (Term.KEY_UP, Term.KEY_DOWN, Term.KEY_LEFT, Term.KEY_RIGHT, Term.KEY_TAB)

    def __init__(self):
        self.running = True
        self.invited_ctrl_c = False  # busy() said "Ctrl-C to stop waiting" during the action in progress
        self.spare_until = 0.0  # time.monotonic() until which a Ctrl-C in the main loop is a late one: LATE_CTRL_C
        self.current_tab = 0
        self.selected_index = 0
        self.scroll_offset = 0
        self.rendered: tuple | None = None  # (tab key, index, anchor) of the selection as last drawn: see locate
        self.switched_at: float | None = None  # when the cursor last fell onto another row by itself: see locate
        self.resized = False  # the terminal changed size and the screen doesn't know yet: see run
        self.message_text = ""
        self.message_style = ""
        self.scr = Screen()  # Buffered screen for flicker-free rendering
        self.fetcher = DataFetcher(interval=2.0)  # Background data fetcher

        # Status colors for containers and projects
        self.container_status_colors = {
            "up": Term.GREEN,
            "running": Term.GREEN,
            "exited": Term.RED,
            "created": Term.YELLOW,
            "paused": Term.YELLOW,
        }
        self.project_status_colors = {
            "running": Term.GREEN,
            "exited": Term.RED,
            "not created": Term.DIM,
        }

        # Define tabs
        self.tabs = [
            Tab(
                name="Containers",
                key="containers",
                columns=[
                    ("Names", "NAME", 20),
                    ("Project", "PROJECT", 15),
                    ("Status", "STATUS", 18),
                    ("Image", "IMAGE", 25),
                    ("Ports", "PORTS", 0),
                ],
                id_key="ID",
            ),
            Tab(
                name="Images",
                key="images",
                columns=[
                    ("Repository", "REPOSITORY", 35),
                    ("Tag", "TAG", 20),
                    ("Size", "SIZE", 15),
                    ("ID", "ID", 0),
                ],
                id_key="ID",
                anchor=("ID", "Repository", "Tag"),  # one image, several tags: several rows with one ID
            ),
            Tab(
                name="Volumes",
                key="volumes",
                columns=[
                    ("Name", "NAME", 50),
                    ("Driver", "DRIVER", 15),
                    ("Mountpoint", "MOUNTPOINT", 0),
                ],
                id_key="Name",
            ),
            Tab(
                name="Networks",
                key="networks",
                columns=[
                    ("Name", "NAME", 30),
                    ("Driver", "DRIVER", 15),
                    ("Scope", "SCOPE", 10),
                    ("ID", "ID", 0),
                ],
                id_key="ID",
            ),
            Tab(
                name="Projects",
                key="projects",
                columns=[
                    ("Name", "PROJECT", 20),
                    ("Status", "STATUS", 15),
                    ("Image", "IMAGE", 30),
                    ("Repo", "REPO", 0),
                ],
                id_key="Name",
                empty=f"No projects configured: add TOML to {PROJECTS_DIR}/",
            ),
            Tab(
                name="Services",
                key="services",
                columns=[
                    ("Name", "SERVICE", 20),
                    ("Status", "STATUS", 15),
                    ("Image", "IMAGE", 30),
                    ("Ports", "PORTS", 0),
                ],
                id_key="Name",
                empty=f"No services configured: add TOML to {SERVICES_DIR}/",
            ),
            Tab(
                name="Issues",
                key="issues",
                columns=[
                    ("level", "LEVEL", 8),
                    ("cat", "CATEGORY", 18),
                    ("msg", "MESSAGE", 56),
                    ("hint", "HINT", 0),
                ],
                id_key="cat",
                empty="No issues",
                anchor=("level", "cat", "msg"),
            ),
            Tab(
                name="Errors",
                key="errors",
                columns=[
                    ("time", "TIME", 8),
                    ("cat", "CATEGORY", 24),
                    ("msg", "MESSAGE", 0),
                ],
                id_key="cat",
                empty="No errors",
                anchor=("ts", "cat", "msg"),
            ),
        ]

    def current(self) -> Tab:
        """Get current tab"""
        return self.tabs[self.current_tab]

    def items(self) -> list:
        """Get items for current tab from cache. Only real rows: what an empty list says is render's business."""
        if self.current().key == "errors":
            errors = get_errors()
            errors.reverse()  # newest first
            return errors
        return self.fetcher.get(self.current().key)

    def anchor_of(self, item: object) -> tuple:
        """What names this row from one fetch to the next"""
        tab = self.current()
        return tuple(getattr(item, name) for name in tab.anchor or (tab.id_key,))

    def locate(self, items: list) -> tuple[object | None, bool]:
        """(the selected row, whether it is gone) in a fresh snapshot. What is selected is a row, named by its
        anchor as last drawn; the index is only where it was. The fetcher swaps the lists every tick, so the
        selection follows its row wherever it is now, and when the row is gone it falls to the closest line: the
        one that took its place, or the last. A redraw just shows that; a key meant for the gone row acts on
        nothing (its neighbour was never on screen under the cursor). Before anything was drawn for this tab and
        index (a tab switch, a jump), the index is all there is."""
        index = self.selected_index
        here = items[index] if 0 <= index < len(items) else None
        if self.rendered is None or self.rendered[:2] != (self.current().key, index):
            return here, False
        anchor = self.rendered[2]
        if here is not None and self.anchor_of(here) == anchor:
            return here, False
        for found, item in enumerate(items):
            if self.anchor_of(item) == anchor:
                self.scroll_offset = max(0, self.scroll_offset + found - index)  # same line on screen
                self.selected_index = found
                return item, False
        self.selected_index = max(0, min(index, len(items) - 1))
        if items:  # the cursor now sits on a row nobody put it on: see SWITCH_GRACE
            self.switched_at = time.monotonic()
        return None, bool(items) or anchor is not None

    def just_switched(self) -> bool:
        """Whether the cursor fell onto its row by itself less than SWITCH_GRACE seconds ago"""
        return self.switched_at is not None and time.monotonic() - self.switched_at < self.SWITCH_GRACE

    def selected_item(self) -> object | None:
        """Get currently selected item"""
        items = self.items()
        if 0 <= self.selected_index < len(items):
            return items[self.selected_index]
        return None

    def selected_id(self) -> str | None:
        """Get ID of selected item"""
        item = self.selected_item()
        if item:
            return getattr(item, self.current().id_key)
        return None

    def tab_alerts(self) -> dict[str, str]:
        """Issues is red while anything detected is error-level, yellow on warnings only; Errors is red while the
        log holds anything"""
        alerts = {}
        if issues := self.fetcher.get("issues"):
            alerts["issues"] = Term.RED if any(i.level == ERROR for i in issues) else Term.YELLOW
        if _errors:  # a len read, not a walk: see the lock's comment
            alerts["errors"] = Term.RED
        return alerts

    def banner(self) -> str:
        """The one thing worth saying on every tab: why docker is unreachable (a daemon that hangs included: each
        listing gives up after POLL_TIMEOUT and says so), when all four listings failed for the same reason. A
        partial failure is its panel's and the Issues tab's business."""
        failures = self.fetcher.get_failures()
        reasons = {failures.get(key, "") for key in LISTINGS}
        return reasons.pop() if len(reasons) == 1 else ""

    def busy(self, text: str) -> None:
        """Say what is about to block the loop, on the message line, before it does: a `docker stop` takes its
        ten seconds, and a screen that just sits there says nothing. The next render replaces it."""
        rows, cols = Term.size()
        UI.message(self.scr, f"{text}... (Ctrl-C to stop waiting)", rows - 1, cols, Term.YELLOW)
        self.scr.flush()
        self.invited_ctrl_c = True

    @staticmethod
    def pause() -> None:
        """On the main screen, before going back to ours: time to read what was just printed there"""
        try:
            input("\nPress Enter to return...")
        except EOFError:  # Ctrl-D is an answer too
            pass

    def shell_ended(self, where: str, code: int, container: str) -> None:
        """Back from a shell: say a non-zero exit code, which the main screen showed for an instant at best.
        The code alone can't tell whose it is: a shell hands back that of the last command typed in it (1, 130:
        an ordinary session), and docker its own when it never ran the shell (125-127, but also 1 for a container
        that is not running). So docker is asked: when `container` no longer runs, or on 125-127, it goes to the
        log too, with what is known."""
        if not code:
            return
        self.set_message(f"Shell in {where} exited with code {code}", Term.YELLOW)
        try:
            info = Docker.inspect(ContainerInspect, container)
        except DockerError as e:  # docker could not say: that is not "it no longer exists"
            log_error("docker.exec", f"Shell in {where} exited with code {code}; then: {e.failure.detail}")
            self.set_message(
                f"Shell in {where} exited with code {code}, and {container} could not be read: {e}", Term.RED
            )
            return
        if info is None or not info.State.get("Running"):
            why = "no longer exists" if info is None else f"is not running ({info.State.get('Status', 'unknown')})"
            log_error("docker.exec", f"Shell in {where} exited with code {code}: {container} {why}")
            self.set_message(f"Shell in {where} exited with code {code}: {container} {why}", Term.RED)
        elif code in (125, 126, 127):
            log_error("docker.exec", f"Shell in {where} exited with code {code}: docker may not have run it at all")

    def set_message(self, text: str, style: str = "") -> None:
        """Set status message"""
        self.message_text = text
        self.message_style = style

    def render(self) -> None:
        """Render the entire UI (buffered - single flush)"""
        rows, cols = Term.size()
        scr = self.scr
        items = self.items()  # Get from cache once

        # The selection is a row, not a position: follow it where it went, else fall to the row that took its place
        self.locate(items)
        if self.selected_index >= len(items):
            self.selected_index = max(0, len(items) - 1)
        anchor = self.anchor_of(items[self.selected_index]) if items else None
        self.rendered = (self.current().key, self.selected_index, anchor)  # what a key will act on: see locate

        # Layout calculations
        tab_row = 1
        header_row = 3
        list_start = 4
        list_height = rows - 6  # Leave room for header, status, etc.
        status_row = rows

        # Reset buffer and position cursor at top
        scr.clear()
        scr.move(1, 1)

        # Tab bar
        UI.tab_bar(scr, self.tabs, self.current_tab, cols, tab_row, alerts=self.tab_alerts())

        # Divider; in its place, what took docker down (on every tab), or what this tab's rows are missing
        if banner := self.banner():
            scr.writeln(2, 1, Term.style(f"── {banner} ".ljust(cols, "─")[:cols], Term.BOLD, Term.RED))
        elif note := self.fetcher.get_notes().get(self.current().key):
            scr.writeln(2, 1, Term.style(f"── {note} ".ljust(cols, "─")[:cols], Term.YELLOW))
        else:
            UI.divider(scr, 2, cols)

        # List header
        UI.list_header(scr, self.current(), header_row, cols)

        # List items
        visible_items = items[self.scroll_offset : self.scroll_offset + list_height]
        for i, item in enumerate(visible_items):
            actual_index = self.scroll_offset + i
            is_selected = actual_index == self.selected_index

            # Apply status colors for containers and projects tabs
            status_colors = None
            if self.current().key == "containers":
                status_colors = self.container_status_colors
            elif self.current().key == "projects":
                status_colors = self.project_status_colors
            elif self.current().key == "services":
                status_colors = self.project_status_colors

            UI.list_row(
                scr,
                self.current(),
                item,
                list_start + i,
                cols,
                is_selected,
                status_colors,
            )

        # Clear remaining lines in list area (avoid leftover content)
        for i in range(len(visible_items), list_height):
            scr.move(list_start + i, 1)
            scr.clear_line()

        # Nothing to list: because the listing failed (say why), or because there is nothing
        if not items:
            failure = self.fetcher.get_failures().get(self.current().key)
            text = failure or self.current().empty or f"No {self.current().name.lower()} found"
            UI.message(scr, f"  {text}", list_start, cols, Term.RED if failure else Term.DIM)

        # The row between the list and the message line holds nothing of ours, but a tall box draws on it
        scr.move(rows - 2, 1)
        scr.clear_line()

        # Clear message line if no message
        if self.message_text:
            UI.message(scr, self.message_text, rows - 1, cols, self.message_style)
        else:
            scr.move(rows - 1, 1)
            scr.clear_line()

        # Status bar with keybindings
        if self.current().key == "containers":
            keys = "[s]tart [S]top [r]estart [L]ogs [e]xec [d]elete [o]rigin [q]uit"
        elif self.current().key == "projects":
            keys = "[e]nter [d]estroy [q]uit"
        elif self.current().key == "services":
            keys = "[u]p [d]own [r]ecreate [q]uit"
        elif self.current().key == "networks":
            keys = "[e]dit [d]elete [q]uit"
        elif self.current().key == "issues":
            keys = "[enter] details [q]uit"
        elif self.current().key == "errors":
            keys = "[enter] details [c]lear [q]uit"
        else:
            keys = "[d]elete [q]uit"
        keys += f"  │  {len(items)} items"
        UI.status_bar(scr, keys, status_row, cols)

        # Single flush - write everything at once
        scr.flush()

    def handle_key(self, key: str) -> None:
        """Handle keypress. An action that needed an answer docker didn't give stops there and says why. Ctrl-C
        while an action runs (or asks) gives up on that action, not on the TUI: the docker client we were waiting
        on dies with it, what the daemon had already been told may still happen, and the next fetch shows it."""
        try:
            self._handle_key(key)
        except KeyboardInterrupt:
            self.enter_tui()  # the action may have left for the main screen
            self.set_message("Interrupted: what docker was already doing may still finish", Term.YELLOW)
        except DockerError as e:
            log_error("docker.command", e.failure.detail)
            self.enter_tui()  # the action may have left for the main screen before it failed
            self.set_message(f"{e}: {e.failure.hint}", Term.RED)
        finally:
            if self.invited_ctrl_c:
                self.invited_ctrl_c = False
                self.spare_until = time.monotonic() + self.LATE_CTRL_C

    def _handle_key(self, key: str) -> None:
        rows, cols = Term.size()
        list_height = rows - 6

        # Clear message on any key
        self.message_text = ""

        # Quit (check first, before any other handlers)
        if key == "q":
            self.running = False
            return

        # One snapshot for the whole keypress: the fetcher swaps the lists under us, and a row looked up twice may
        # be two different rows (the prompt naming one image, the delete taking another), or gone the second time.
        # In it, the row the key is for is the one that was on screen (see locate), not whatever sits at its index.
        # The branches below are one elif chain: the ones that move the selection never reach the ones that use it.
        items = self.items()
        selected, gone = self.locate(items)
        selected_id = getattr(selected, self.current().id_key) if selected else None
        if key in self.MOVE_KEYS:  # moved by hand: wherever the cursor ends up is the user's choice
            self.switched_at = None

        # Navigation (jikl: j=left, i=up, k=down, l=right)

        if key in ("k", Term.KEY_DOWN):
            if self.selected_index < len(items) - 1:
                self.selected_index += 1
                # Scroll down if needed
                if self.selected_index >= self.scroll_offset + list_height:
                    self.scroll_offset += 1

        elif key in ("i", Term.KEY_UP):
            if self.selected_index > 0:
                self.selected_index -= 1
                # Scroll up if needed
                if self.selected_index < self.scroll_offset:
                    self.scroll_offset = self.selected_index

        elif key == "g":  # Go to top
            self.selected_index = 0
            self.scroll_offset = 0

        elif key == "G":  # Go to bottom
            self.selected_index = max(0, len(items) - 1)
            self.scroll_offset = max(0, len(items) - list_height)

        # Tab switching (no refresh needed - background thread handles it)
        elif key == Term.KEY_TAB or key == "l" or key == Term.KEY_RIGHT:
            self.current_tab = (self.current_tab + 1) % len(self.tabs)
            self.selected_index = 0
            self.scroll_offset = 0

        elif key == "j" or key == Term.KEY_LEFT:
            self.current_tab = (self.current_tab - 1) % len(self.tabs)
            self.selected_index = 0
            self.scroll_offset = 0

        # Number keys for tabs
        elif key and key in "123456789":
            idx = int(key) - 1
            if idx < len(self.tabs):
                self.current_tab = idx
                self.selected_index = 0
                self.scroll_offset = 0

        # Not a key that moves around: it was for a row that isn't there anymore, or the cursor only just fell
        # onto this one. Either way nothing is done and nothing is retried: a box to ok, then the key is the
        # user's to press again, for the row then on screen.
        elif gone or (selected_id and self.just_switched()):
            self.switched_at = None
            self.rendered = None  # oked: the redraw that follows starts from where the cursor is, not a new fall
            what = "the row that key was for is gone" if gone else f"the selection just moved to '{selected_id}'"
            lines = [f"The list changed: {what}.", "Nothing was done: press the key again if you still mean it."]
            UI.info_box("Selection changed", lines, rows, cols, ok=True)

        # Container-specific actions
        elif self.current().key == "containers" and selected_id:
            container_id = selected_id

            if key == "s":  # Start
                self.busy(f"Starting {container_id[:12]}")
                if Docker.start(container_id):
                    self.set_message(f"Started {container_id[:12]}", Term.GREEN)
                else:
                    self.set_message(f"Failed to start {container_id[:12]}", Term.RED)

            elif key == "S":  # Stop
                self.busy(f"Stopping {container_id[:12]}")
                if Docker.stop(container_id):
                    self.set_message(f"Stopped {container_id[:12]}", Term.GREEN)
                else:
                    self.set_message(f"Failed to stop {container_id[:12]}", Term.RED)

            elif key == "r":  # Restart
                self.busy(f"Restarting {container_id[:12]}")
                if Docker.restart(container_id):
                    self.set_message(f"Restarted {container_id[:12]}", Term.GREEN)
                else:
                    self.set_message(f"Failed to restart {container_id[:12]}", Term.RED)

            elif key == "L":  # Logs
                self.exit_tui_temporarily()
                Docker.logs(container_id, tail=100)
                self.pause()
                self.enter_tui()

            elif key == "e":  # Exec
                self.exit_tui_temporarily()
                code = Docker.shell(container_id)
                self.enter_tui()
                self.shell_ended(selected.Names, code, container_id)

            elif key == "d":  # Delete
                name = selected.Names
                if UI.confirm(f"Delete container '{name}'?", rows - 1, cols):
                    self.busy(f"Deleting {name}")
                    if Docker.remove_container(container_id, force=True):
                        self.set_message(f"Deleted {name}", Term.GREEN)
                    else:
                        self.set_message(f"Failed to delete {name}", Term.RED)

            elif key == "o":  # Origin/source
                name = selected.Names
                result = Docker.investigate_source(container_id)
                title = f"Source: {result['source']}"
                lines = [f"Container: {name}"] + result["details"]
                UI.info_box(title, lines, rows, cols)

        # Image deletion
        elif self.current().key == "images" and key == "d" and selected_id:
            image_id = selected_id
            repo = selected.Repository
            tag = selected.Tag
            name = f"{repo}:{tag}" if tag else repo

            if UI.confirm(f"Delete image '{name}'?", rows - 1, cols):
                self.busy(f"Deleting {name}")
                if Docker.remove_image(image_id, force=True):
                    self.set_message(f"Deleted {name}", Term.GREEN)
                else:
                    self.set_message(f"Failed to delete {name}", Term.RED)

        # Volume deletion
        elif self.current().key == "volumes" and key == "d" and selected_id:
            volume_name = selected_id

            if UI.confirm(f"Delete volume '{volume_name}'?", rows - 1, cols):
                if Docker.remove_volume(volume_name):
                    self.set_message(f"Deleted {volume_name}", Term.GREEN)
                else:
                    self.set_message(f"Failed to delete {volume_name}", Term.RED)

        # Network actions
        elif self.current().key == "networks" and selected_id:
            network_id = selected_id
            name = selected.Name

            if key == "d":  # Delete
                if UI.confirm(f"Delete network '{name}'?", rows - 1, cols):
                    if Docker.remove_network(network_id):
                        self.set_message(f"Deleted {name}", Term.GREEN)
                    else:
                        self.set_message(f"Failed to delete {name}", Term.RED)

            elif key == "e":  # Edit — show connected containers
                info = Docker.inspect(NetworkInspect, network_id)
                if info:
                    containers = info.Containers
                    if containers:
                        lines = [v.get("Name", k[:12]) for k, v in containers.items()]
                    else:
                        lines = ["No containers connected"]
                    UI.info_box(f"Network: {info.Name}", lines, rows, cols)
                else:
                    self.set_message(f"Failed to inspect {name}", Term.RED)

        # Project actions
        elif self.current().key == "projects" and selected_id:
            project_name = selected_id
            projects, _ = read_projects()  # a broken file is already on the Issues tab
            project = projects.get(project_name)

            if project and key == "e":  # Enter
                self.exit_tui_temporarily()
                code = Docker.enter_project(project)
                if code is None:  # why is on the main screen (and in the log): leave time to read it
                    self.pause()
                self.enter_tui()
                if code is None:
                    self.set_message(f"{project_name} was not entered: see Errors", Term.RED)
                else:
                    self.shell_ended(project_name, code, project.container_name)

            elif project and key == "d":  # Destroy, the prompt carrying `check`'s verdict when it is DIRTY
                self.busy(f"Checking the workspace of {project_name}")
                question = destroy_question(project)
                if question.names:
                    if UI.confirm(question.text, rows - 1, cols):
                        self.busy(f"Destroying {project_name}")
                        # Only what the prompt named: what appeared while it was up was never agreed to
                        if Docker.destroy_project(project, question.names):
                            self.set_message(f"Destroyed {project_name}", Term.GREEN)
                        else:
                            self.set_message(f"Failed to destroy {project_name}", Term.RED)
                else:
                    self.set_message(f"No container for {project_name}", Term.YELLOW)

        # Service actions
        elif self.current().key == "services" and selected_id:
            service_name = selected_id
            services, _ = read_services()  # a broken file is already on the Issues tab
            service = services.get(service_name)

            if service and key == "u":  # Up
                self.exit_tui_temporarily()
                Docker.service_up(service)
                self.pause()
                self.enter_tui()

            elif service and key == "d":  # Down
                if Docker.find_service_container(service_name):
                    if UI.confirm(f"Stop service '{service_name}'?", rows - 1, cols):
                        self.busy(f"Stopping {service_name}")
                        if Docker.service_down(service):
                            self.set_message(f"Stopped {service_name}", Term.GREEN)
                        else:
                            self.set_message(f"Failed to stop {service_name}", Term.RED)
                else:
                    self.set_message(f"No container for {service_name}", Term.YELLOW)

            elif service and key == "r":  # Recreate
                if UI.confirm(f"Recreate service '{service_name}'?", rows - 1, cols):
                    self.exit_tui_temporarily()
                    if Docker.service_recreate(service):
                        print(f"Recreated {service_name}")
                    else:
                        print(f"Failed to recreate {service_name}")
                    self.pause()
                    self.enter_tui()

        # What a row clips, in full: an issue's message and hint, a log entry's message and context
        elif self.current().key == "issues" and key == Term.KEY_ENTER and selected:
            issue = selected
            lines = textwrap.wrap(issue.msg, max(cols - 8, 20)) or [""]
            if issue.hint:
                lines += [""] + textwrap.wrap(issue.hint, max(cols - 8, 20))
            for raw in issue.detail.splitlines():
                lines += [""] + textwrap.wrap(raw, max(cols - 8, 20))
            UI.info_box(f"{issue.level}  {issue.cat}", lines, rows, cols)

        elif self.current().key == "errors" and key == Term.KEY_ENTER and selected:
            event = selected
            lines = textwrap.wrap(event.msg, max(cols - 8, 20)) or [""]
            if event.ctx:
                lines += [""] + textwrap.wrap(str(event.ctx), max(cols - 8, 20))
            UI.info_box(f"{event.time}  {event.cat}", lines, rows, cols)

        elif self.current().key == "errors" and key == "c" and items:
            if UI.confirm("Clear the error log?", rows - 1, cols):
                self.set_message(f"Cleared {clear_errors()} errors", Term.GREEN)

    def enter_tui(self) -> None:
        """Enter TUI mode: our screen, and the keyboard raw for as long as we stay"""
        Term.raw_on()
        print(Term.ALT_SCREEN + Term.HIDE_CURSOR, end="", flush=True)

    def exit_tui_temporarily(self) -> None:
        """Exit TUI mode temporarily (for exec/logs): what runs next gets the terminal as we found it"""
        print(Term.SHOW_CURSOR + Term.MAIN_SCREEN, end="", flush=True)
        Term.raw_off()

    def exit_tui(self) -> None:
        """Exit TUI mode completely"""
        print(Term.SHOW_CURSOR + Term.MAIN_SCREEN, end="", flush=True)
        Term.raw_off()

    def run(self) -> None:
        """Main TUI loop"""

        # A resize is only noted here, and drawn by the loop below. A handler runs wherever the main thread
        # happens to be: drawing from it paints the list over the shell or the logs we handed the terminal to
        # (the loop doesn't turn while one of those runs), or lands in the middle of a frame being built.
        def handle_resize(signum, frame):
            self.resized = True

        signal.signal(signal.SIGWINCH, handle_resize)

        # Start background data fetcher
        self.fetcher.start()

        self.enter_tui()
        try:
            while self.running:
                try:
                    # Read before drawing: what changes meanwhile is drawn next
                    self.resized = False
                    version = self.fetcher.version
                    self.render()
                    key = ""
                    # A key, a fetch to show, a new size: whichever first
                    while not key and not self.resized and self.fetcher.version == version:
                        key = Term.getch(timeout=self.REDRAW_POLL)
                    if key:
                        self.handle_key(key)
                except KeyboardInterrupt:
                    if time.monotonic() >= self.spare_until:
                        raise
                    self.spare_until = 0.0  # one late Ctrl-C is spared, not every one for a second
                    self.set_message("Too late to interrupt: that had just finished", Term.YELLOW)
        except KeyboardInterrupt:  # Ctrl-C quits from the list; during an action it only ends that (handle_key)
            pass
        finally:
            self.fetcher.stop()
            self.exit_tui()


# =============================================================================
# CLI
# =============================================================================


TABLE_MIN_COL = 8


def fit_widths(headers: list[str], rows: list[list[str]], width: int | None, shrink: list[int]) -> list[int]:
    """Column widths for a table: each column's content width, then, given a `width`, the columns
    listed in `shrink` give up space until the widest line fits: always the widest of them first
    (ties: earlier in `shrink`), one character at a time, so long columns end up balanced instead
    of one being crushed. None goes under TABLE_MIN_COL."""
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    if width is not None:
        while sum(widths) + len(widths) - 1 > width:
            candidates = [i for i in shrink if widths[i] > TABLE_MIN_COL]
            if not candidates:
                break
            widths[max(candidates, key=lambda i: widths[i])] -= 1
    return widths


def format_table(headers: list[str], rows: list[list[str]], width: int | None, shrink: list[int]) -> list[str]:
    """Render left-aligned columns sized by `fit_widths`; clipped cells end with '…'.
    `width=None` never clips (piped output, `--wide`)."""
    widths = fit_widths(headers, rows, width, shrink)

    def cell(text: str, w: int) -> str:
        return text if len(text) <= w else text[: w - 1] + "…"

    lines = [" ".join(f"{cell(h, w):<{w}}" for h, w in zip(headers, widths)).rstrip()]
    lines.append("-" * (sum(widths) + len(widths) - 1))
    for row in rows:
        lines.append(" ".join(f"{cell(c, w):<{w}}" for c, w in zip(row, widths)).rstrip())
    return lines


def _table_width(wide: bool) -> int | None:
    """Terminal width to fit tables into; None (no clipping) when asked or when not on a tty"""
    if wide or not sys.stdout.isatty():
        return None
    return shutil.get_terminal_size().columns


def cli_ps(args: argparse.Namespace) -> None:
    """List containers"""
    containers = Docker.list_containers(all_containers=not args.running).required()

    if args.quiet:
        for c in containers:
            print(c.ID)
        return

    headers = ["ID", "NAME", "STATUS", "IMAGE", "PORTS", "NETWORKS"]
    rows = [
        [
            c.ID[:12],
            c.Names,
            c.Status,
            c.Image,
            c.Ports,
            c.Networks,
        ]
        for c in containers
    ]
    # Give up NETWORKS first, then PORTS, IMAGE, NAME, STATUS; the ID is never clipped
    print("\n".join(format_table(headers, rows, _table_width(args.wide), shrink=[5, 4, 3, 1, 2])))


def cli_images(args: argparse.Namespace) -> None:
    """List images"""
    images = Docker.list_images().required()

    if args.quiet:
        for img in images:
            print(img.ID)
        return

    print(f"{'REPOSITORY':<35} {'TAG':<20} {'SIZE':<15} {'ID'}")
    print("-" * 80)
    for img in images:
        repo = img.Repository[:34]
        tag = img.Tag[:19]
        size = img.Size[:14]
        id_ = img.ID[:12]
        print(f"{repo:<35} {tag:<20} {size:<15} {id_}")


def cli_volumes(args: argparse.Namespace) -> None:
    """List volumes"""
    volumes = Docker.list_volumes().required()

    if args.quiet:
        for v in volumes:
            print(v.Name)
        return

    print(f"{'NAME':<50} {'DRIVER':<15}")
    print("-" * 65)
    for v in volumes:
        name = v.Name[:49]
        driver = v.Driver[:14]
        print(f"{name:<50} {driver:<15}")


def cli_networks(args: argparse.Namespace) -> None:
    """List networks"""
    networks = Docker.list_networks().required()

    if args.quiet:
        for n in networks:
            print(n.ID)
        return

    print(f"{'NAME':<30} {'DRIVER':<15} {'SCOPE':<10} {'ID'}")
    print("-" * 70)
    for n in networks:
        name = n.Name[:29]
        driver = n.Driver[:14]
        scope = n.Scope[:9]
        id_ = n.ID[:12]
        print(f"{name:<30} {driver:<15} {scope:<10} {id_}")


def cli_start(args: argparse.Namespace) -> None:
    """Start container"""
    if Docker.start(args.container):
        print(f"Started {args.container}")
    else:
        print(f"Failed to start {args.container}", file=sys.stderr)
        sys.exit(1)


def cli_stop(args: argparse.Namespace) -> None:
    """Stop container"""
    if Docker.stop(args.container):
        print(f"Stopped {args.container}")
    else:
        print(f"Failed to stop {args.container}", file=sys.stderr)
        sys.exit(1)


def cli_restart(args: argparse.Namespace) -> None:
    """Restart container"""
    if Docker.restart(args.container):
        print(f"Restarted {args.container}")
    else:
        print(f"Failed to restart {args.container}", file=sys.stderr)
        sys.exit(1)


def cli_logs(args: argparse.Namespace) -> None:
    """Show container logs"""
    sys.exit(Docker.logs(args.container, follow=args.follow, tail=args.tail))


def cli_exec(args: argparse.Namespace) -> None:
    """Execute command in container. Everything after the container is the command, dashes included
    (`exec <id> ls -la`); a leading `--` is accepted and dropped."""
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print("Usage: containerctl exec <id> <cmd...>", file=sys.stderr)
        sys.exit(2)
    sys.exit(Docker.exec(args.container, command))


def cli_shell(args: argparse.Namespace) -> None:
    """Open shell in container"""
    sys.exit(Docker.shell(args.container, args.shell))


def cli_enter(args: argparse.Namespace) -> None:
    """Enter a project (spawn or attach)"""
    projects = load_projects()
    target = resolve_enter_target(args.project, projects)
    if target is None:
        shortcuts = sorted({alias for p in projects.values() for alias in p.shortcuts})
        print(f"Unknown project or shortcut: {args.project}", file=sys.stderr)
        print(f"Available projects: {', '.join(projects.keys()) or 'none'}", file=sys.stderr)
        print(f"Available shortcuts: {', '.join(shortcuts) or 'none'}", file=sys.stderr)
        print(f"Add project configs to {PROJECTS_DIR}/ (see: containerctl help config)", file=sys.stderr)
        sys.exit(1)
    project, workdir = target
    user = None
    for option in args.options:
        key, sep, value = option.partition("=")
        if not sep or key != "user":
            print(f"Unknown option: {option} (supported: user=<user>)", file=sys.stderr)
            sys.exit(1)
        if not value or not is_valid_id(value):
            print(f"Invalid user: {value}", file=sys.stderr)
            sys.exit(1)
        user = value
    code = Docker.enter_project(project, user=user, workdir=workdir)
    if code != 0:  # no shell (None), or the shell's own exit code, as `shell` and `exec` hand it back
        sys.exit(1 if code is None else code)


def cli_spawn(args: argparse.Namespace) -> None:
    """Spawn a new project container"""
    projects = load_projects()
    if args.project not in projects:
        print(f"Unknown project: {args.project}", file=sys.stderr)
        sys.exit(1)
    project = projects[args.project]
    if Docker.find_project_container(project.name):
        print(f"Container {project.container_name} already exists. Use 'enter' or 'destroy' first.", file=sys.stderr)
        sys.exit(1)
    if not Docker.spawn_project(project):
        sys.exit(1)


def stdin_is_tty() -> bool:
    """Is somebody at a terminal on the other end of our stdin?"""
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except ValueError:  # closed after startup; closed before it (`<&-`), there is no sys.stdin at all
        return False


def confirm_cli(prompt: str) -> bool:
    """y/N on the terminal. No when there is nobody to ask: a script, a pipe, a closed stdin."""
    if not stdin_is_tty():
        return False
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def cli_destroy(args: argparse.Namespace) -> None:
    """Destroy a project container, after a `check` of its workspace and a yes: the verdict (DIRTY, or a check
    that could not run) is in the question, and so is what an unfinished spawn left (it may still be running).
    `--force` skips the question, not the check: the verdict is still printed, as a warning."""
    projects = load_projects()
    if args.project not in projects:
        print(f"Unknown project: {args.project}", file=sys.stderr)
        sys.exit(1)
    project = projects[args.project]
    question = destroy_question(project)
    if question.report and not question.report.clean:
        print_report(f"{project.name} ({project.container_name})", question.report)
    if args.force:
        if question.verdict:
            print(f"warning: {project.name}: {question.verdict}, destroying anyway (--force)", file=sys.stderr)
    elif question.text and not confirm_cli(question.text):
        nobody = "" if stdin_is_tty() else ": no terminal to ask on, use --force"
        print(f"Not destroyed{nobody}", file=sys.stderr)
        sys.exit(1)
    if not Docker.destroy_project(project, question.names):  # what was asked about
        sys.exit(1)
    print(f"Destroyed {args.project}")


def cli_projects(args: argparse.Namespace) -> None:
    """List configured projects and their container status"""
    projects = load_projects()
    if not projects:
        print(f"No projects configured. Add TOML files to {PROJECTS_DIR}/ (see: containerctl help config)")
        return

    headers = ["PROJECT", "STATUS", "IMAGE", "REPO", "SHORTCUTS"]
    ps = Docker.list_containers()  # once, not one `docker ps` per project
    ps.logged()
    status = status_lookup(ps)
    rows = []
    for name, project in projects.items():
        shortcuts = ", ".join(project.shortcuts) or "-"
        rows.append([name, status(project.container_name), project.image, project.repos_display, shortcuts])
    width = _table_width(args.wide)
    shrink = [3, 2, 0, 1]  # REPO, IMAGE, PROJECT, STATUS
    if width is not None:
        # A shortcut list longer than the balanced share the fitter would give that column collapses
        # to its count rather than being clipped mid-name (the full list is `projects -w`). The
        # second fit below then hands the freed space back to the other columns.
        share = fit_widths(headers, rows, width, shrink + [4])[4]
        for row, project in zip(rows, projects.values()):
            n = len(project.shortcuts)
            if n and len(row[-1]) > share:
                row[-1] = f"({n} shortcut{'s' if n > 1 else ''})"
    print("\n".join(format_table(headers, rows, width, shrink)))


def _check_one(project: "ProjectConfig") -> bool | None:
    """Check a single project's workspace over docker exec.

    Returns True if clean, False if dirty or impossible to check, None if there is no container.
    """
    container = Docker.find_project_container(project.name)
    if container is None:
        print(f"{project.name}: no container (nothing to check)")
        return None
    try:
        report = project_report(project)
    except DockerError as e:  # no verdict is not a clean one; with --all, the others still get theirs
        log_error("docker.command", e.failure.detail)
        print(f"{project.name} ({project.container_name}):\n  could NOT be checked: {e} ({e.failure.hint})")
        return False
    return print_report(f"{project.name} ({project.container_name})", report)


def cli_check(args: argparse.Namespace) -> None:
    """Check whether project workspaces are safe to destroy (host-side)."""
    projects = load_projects()
    if args.all:
        targets = list(projects.values())
        if not targets:
            print(f"No projects configured. Add TOML files to {PROJECTS_DIR}/ (see: containerctl help config)")
            return
    else:
        if args.project is None:
            print("Usage: containerctl check <project> | containerctl check --all", file=sys.stderr)
            sys.exit(2)
        if args.project not in projects:
            print(f"Unknown project: {args.project}", file=sys.stderr)
            sys.exit(1)
        targets = [projects[args.project]]

    dirty = False
    for i, project in enumerate(targets):
        if i:
            print()
        result = _check_one(project)
        if result is False:
            dirty = True
    if dirty:
        sys.exit(1)


def current_issues() -> list[Issue]:
    """The issues as the TUI detects them, for one CLI run: nothing is logged, what is broken is in the list"""
    projects, broken_projects = read_projects()
    services, broken_services = read_services()
    images, broken_images = read_images()
    issues = broken_projects + broken_services + broken_images
    return by_severity(issues + detect_issues(projects, services, images, Docker.listings()))


def print_issues(issues: list[Issue], file=None) -> None:
    file = file or sys.stdout
    if not issues:
        print("No issues", file=file)
        return
    rows = [[i.level, i.cat, i.msg, i.hint] for i in issues]
    print("\n".join(format_table(["LEVEL", "CATEGORY", "MESSAGE", "HINT"], rows, None, [])), file=file)
    for issue in issues:  # the raw material the TUI shows on Enter
        if issue.detail:
            print(f"\n{issue.cat}: {issue.msg}", file=file)
            for raw in issue.detail.splitlines():
                print(f"  {raw}", file=file)


def print_errors(errors: list[Event], file=None) -> None:
    file = file or sys.stderr
    rows = [[e.cat, e.msg] for e in errors]
    print("\n".join(format_table(["ERROR", "MESSAGE"], rows, None, [])), file=file)


def cli_issues(args: argparse.Namespace) -> None:
    """What wants an action: the Issues tab for one run. Exit 1 when anything is error-level (docker unreachable,
    a config file that doesn't parse) or the run logged errors (printed by main); warnings alone exit 0."""
    issues = current_issues()
    print_issues(issues)
    if any(i.level == ERROR for i in issues) or get_errors():
        sys.exit(1)


def cli_check_local(args: argparse.Namespace) -> None:
    """Check the current workspace from inside a container (no docker)."""
    expected = read_manifest()
    if expected is None:
        print("Warning: no manifest found; skipping extra/missing detection", file=sys.stderr)
    try:
        report = check_workspace(expected, _local_list_entries, _local_git, read_injected(), _local_digest)
    except OSError as e:  # a workspace we can't list, or no git here: no verdict, and that is not "clean"
        print(f"error: could not check the workspace: {e}", file=sys.stderr)
        sys.exit(1)
    clean = print_report("workspace", report)
    if not clean:
        sys.exit(1)


def cli_service_up(args: argparse.Namespace) -> None:
    """Bring a service up"""
    services = load_services()
    if args.service not in services:
        print(f"Unknown service: {args.service}", file=sys.stderr)
        print(
            f"Available services: {', '.join(services.keys()) or 'none'}",
            file=sys.stderr,
        )
        print(f"Add service configs to {SERVICES_DIR}/", file=sys.stderr)
        sys.exit(1)
    if not Docker.service_up(services[args.service]):
        sys.exit(1)


def cli_service_down(args: argparse.Namespace) -> None:
    """Bring a service down"""
    services = load_services()
    if args.service not in services:
        print(f"Unknown service: {args.service}", file=sys.stderr)
        sys.exit(1)
    if not Docker.service_down(services[args.service]):
        sys.exit(1)


def cli_service_recreate(args: argparse.Namespace) -> None:
    """Recreate a service"""
    services = load_services()
    if args.service not in services:
        print(f"Unknown service: {args.service}", file=sys.stderr)
        sys.exit(1)
    if not Docker.service_recreate(services[args.service]):
        sys.exit(1)
    print(f"Recreated {args.service}")


def cli_service_script(args: argparse.Namespace) -> None:
    """Run a pre-registered script in a service container"""
    services = load_services()
    if args.service not in services:
        print(f"Unknown service: {args.service}", file=sys.stderr)
        print(f"Available services: {', '.join(services.keys()) or 'none'}", file=sys.stderr)
        sys.exit(1)
    service = services[args.service]
    if args.script not in service.scripts:
        print(f"Unknown script '{args.script}' for service {args.service}", file=sys.stderr)
        print(f"Available scripts: {', '.join(service.scripts.keys()) or 'none'}", file=sys.stderr)
        sys.exit(1)
    container = Docker.find_service_container(service.name)
    if container is None:
        print(f"No container found for service {service.name} (run 'service up' first)", file=sys.stderr)
        sys.exit(1)
    sys.exit(Docker.exec(service.container_name, service.scripts[args.script]))


def cli_service_scripts(args: argparse.Namespace) -> None:
    """List pre-registered scripts for a service"""
    services = load_services()
    if args.service not in services:
        print(f"Unknown service: {args.service}", file=sys.stderr)
        sys.exit(1)
    service = services[args.service]
    if not service.scripts:
        print(f"No scripts configured for service {service.name}")
        return
    width = max(len(n) for n in service.scripts) + 2
    for name, argv in service.scripts.items():
        print(f"{name:<{width}} {' '.join(argv)}")


def cli_services(args: argparse.Namespace) -> None:
    """List configured services and their container status"""
    services = load_services()
    if not services:
        print(f"No services configured. Add TOML files to {SERVICES_DIR}/ (see: containerctl help config)")
        return

    headers = ["SERVICE", "STATUS", "IMAGE", "PORTS", "NETWORKS"]
    ps = Docker.list_containers()  # once, not one `docker ps` per service
    ps.logged()
    status = status_lookup(ps)
    rows = []
    for name, service in services.items():
        ports = ", ".join(service.ports[:3])
        if len(service.ports) > 3:
            ports += f" +{len(service.ports) - 3}"
        rows.append([name, status(service.container_name), service.image, ports, ", ".join(service.networks) or "-"])
    print("\n".join(format_table(headers, rows, _table_width(args.wide), shrink=[4, 3, 2, 0, 1])))


def _image_or_exit(name: str, images: dict[str, "ImageConfig"]) -> "ImageConfig":
    if name not in images:
        print(f"Unknown image: {name}", file=sys.stderr)
        print(f"Available images: {', '.join(images) or 'none'}", file=sys.stderr)
        print(f"Add image recipes to {IMAGES_DIR}/ (see: containerctl help config)", file=sys.stderr)
        sys.exit(1)
    return images[name]


def cli_image_build(args: argparse.Namespace) -> None:
    """Build an image from its recipe (always; `up`/`spawn` only build when the image is missing)"""
    image = _image_or_exit(args.image, load_images())
    print(f"Building {image.tag} from {image.dockerfile}...")
    if not Docker.build_image(image):
        sys.exit(1)


@dataclass
class ImageState:
    """What `image ls` / `image check` know about one recipe from docker alone (no network)"""

    versions: list[int]  # every cctl-local-N of the repository
    current: int | None  # the one the recipe tag points at
    built: str  # up to date | stale | not built | built elsewhere | unreadable

    @property
    def current_display(self) -> str:
        return "-" if self.current is None else f"{LOCAL_VERSION_PREFIX}{self.current:06d}"


def image_state(image: "ImageConfig", rows: list[Image]) -> ImageState:
    """From the recipe files and the `docker images` rows (one listing serves every recipe)"""
    recipe = image.recipe_hash()
    versions, current, tag_id = local_versions(image, rows)
    if recipe is None:
        built = "unreadable"
    elif tag_id is None:
        built = "not built"
    else:
        info = Docker.inspect(ImageInspect, image.tag)
        label = ((info.Config or {}).get("Labels") or {}).get(LABEL_RECIPE) if info else None
        built = "built elsewhere" if label is None else ("up to date" if label == recipe else "stale")
    return ImageState(versions, current, built)


def upstream_status(image: "ImageConfig") -> tuple[str, str | None]:
    """(status, diff): - | in sync | differs | unreachable, plus the unified diff upstream -> local when they differ"""
    if not image.upstream:
        return "-", None
    remote = fetch_upstream(image.upstream)
    if remote is None:
        return "unreachable", None
    try:
        local = image.dockerfile.read_text(errors="replace")
    except OSError:
        return "unreachable", None
    if local == remote:
        return "in sync", None
    diff = difflib.unified_diff(
        remote.splitlines(True), local.splitlines(True), fromfile=image.upstream, tofile=str(image.dockerfile)
    )
    return "differs", "".join(diff)


def print_images(images: dict[str, "ImageConfig"], upstream: bool) -> bool:
    """The recipes table, with upstream diffs after it when asked. True when nothing is off."""
    if not images:
        print(f"No image recipes. Add <name>.toml + <name>.Dockerfile to {IMAGES_DIR}/ (see: containerctl help config)")
        return True
    headers = ["IMAGE", "TAG", "CURRENT", "VERSIONS", "BUILT"] + (["UPSTREAM"] if upstream else [])
    rows, notes, ok = [], [], True
    # Once, not one `docker images` per recipe. DockerError when it failed: "not built" would be a guess.
    listing = Docker.list_images().required()
    for name, image in images.items():
        state = image_state(image, listing)
        row = [name, image.tag, state.current_display, str(len(state.versions)), state.built]
        ok = ok and state.built == "up to date"
        if upstream:
            status, diff = upstream_status(image)
            row.append(status)
            ok = ok and status in ("-", "in sync")
            if diff:
                notes.append(diff)
        rows.append(row)
        if issue := image_version_issue(name, image, state.versions, state.current):
            notes.append(
                f"warning: {issue.msg}; docker refuses to remove an image a container still uses,\n"
                f"so this is safe to try:  {issue.hint}\n"
            )
    print("\n".join(format_table(headers, rows, None, [])))
    for note in notes:
        print()
        print(note, end="")
    return ok


def cli_image_ls(args: argparse.Namespace) -> None:
    """List recipes: current build, version count, built state. Docker only, never the network."""
    print_images(load_images(), upstream=False)


def cli_image_check(args: argparse.Namespace) -> None:
    """`image ls` plus local Dockerfile vs upstream, diffs included. Exit 1 when anything is off."""
    images = load_images()
    if args.image:
        images = {args.image: _image_or_exit(args.image, images)}
    if not print_images(images, upstream=True):
        sys.exit(1)


# `help full`: the why and the concepts, ahead of the commands. Written for me in two years, and for whoever reads
# the README, which is this output (a test holds them equal). Say what a thing is and why it is that way, not how
# to type it: the commands below it do that.
CONCEPTS = [
    "Why: I work in throwaway containers and keep a few services up. I want that in one file I can read,",
    "stdlib only, on top of the docker CLI I already know. No daemon of its own, no state of its own:",
    "what exists is what `docker ps` says, what should exist is a few TOML files. Not compose, not k8s.",
    "──────────────────────────────────────────────",
    "Concepts",
    "- project  ==> a throwaway dev container (cctl-prj-<name>, own network cctl-net-prj-<name>) from a TOML:",
    "               image, ssh key, repos cloned in ~/workspace, env files. `enter` spawns it when",
    "               missing, then drops you in a shell. Nothing is mounted from the host: what is not pushed",
    "               dies with the container. That is the point, and why `check` exists. It is made as",
    "               cctl-spawning-<name> and renamed once all of that is in: a spawn that died halfway is",
    "               not a project container: the next `enter` says how old it is, asks, and starts over.",
    "- check    ==> is it safe to destroy? Everything in the workspace must be a configured repo, and every",
    "               repo boring: nothing uncommitted or untracked, no stash, no detached HEAD, no linked",
    "               worktree, every local branch on its upstream. Anything else is DIRTY, what could not be",
    "               read included. Submodules, nested ones too, are repos of their own and checked as such.",
    "               No network: it trusts the remote refs as last fetched. `destroy` runs it first, then asks.",
    "- shortcut ==> a name for a dir of a project: `enter <shortcut>` lands there. Global: keep them unique.",
    "- env      ==> of a project: [env] plain values, [secrets] names of keys in secrets.toml. Both reach the",
    "               `enter` shell only: never argv, never the container's config, never a project file.",
    "- service  ==> a long-lived container (cctl-svc-<name>): `docker run` written as TOML, plus scripts to",
    "               exec in it. `up` creates or starts (and connects new networks), it never changes a",
    "               container: any other TOML edit needs `recreate`. Its `env` is plain `docker run -e`:",
    "               anyone who can `docker inspect` reads it.",
    "- image    ==> a recipe: a Dockerfile (+ context dir) beside a TOML. Each build is its own tag,",
    "               <repo>:cctl-local-NNNNNN, and the recipe tag is re-pointed at it: the previous build is",
    "               still there to go back to. `spawn` / `up` build a missing image, never rebuild one.",
    "               `image check` says when the build is stale, and diffs the Dockerfile against upstream.",
    "               A project image must have: the project's user and its home (`dev` and /home/dev,",
    "               unless its TOML says another `user` / `home`), bash, git (+ ssh for git@ repos),",
    "               coreutils. python3 too, for `check-local` to run inside. Its USER does not matter: the",
    "               clones, the checks and the `enter` shell all run as the project's user.",
    "- issue    ==> true right now and wants an action: docker down, a TOML that doesn't parse, a container",
    "               with no config left, ... Detected, never stored: fix it and it is gone.",
    "- error    ==> something that failed once: a log. A tab in the TUI; a CLI run prints its own at the end.",
    "- TUI      ==> no argument. One tab per kind of thing, refreshed every 2s. i/k up/down, j/l or 1-9 for",
    "               tabs, g/G top/bottom, q quits; the bottom line has the keys of the tab you are on.",
    "──────────────────────────────────────────────",
]


def usage(full: bool = False) -> int:
    """Display usage information: the commands, after the concepts when `full`"""
    output_lines = [
        "containerctl - KISS containers CLI/TUI manager",
        "──────────────────────────────────────────────",
        *(CONCEPTS if full else []),
        "- containerctl                          ==> launch TUI",
        "- containerctl help                     ==> show this help",
        "- containerctl help full                ==> the why and the concepts, then this help (= the README)",
        "- containerctl help config              ==> example project, service, image and secrets TOML",
        "──────────────────────────────────────────────",
        "- containerctl ps                       ==> list containers",
        "- containerctl ps -q                    ==> list container IDs only",
        "- containerctl ps -r                    ==> list running containers only",
        "- containerctl ps -w                    ==> wide: never clip columns to the terminal width",
        "- containerctl images                   ==> list images",
        "- containerctl images -q                ==> list image IDs only",
        "- containerctl volumes                  ==> list volumes",
        "- containerctl volumes -q               ==> list volume names only",
        "- containerctl networks                 ==> list networks",
        "- containerctl networks -q              ==> list network IDs only",
        "──────────────────────────────────────────────",
        "- containerctl start <id>               ==> start container",
        "- containerctl stop <id>                ==> stop container",
        "- containerctl restart <id>             ==> restart container",
        "──────────────────────────────────────────────",
        "- containerctl logs <id>                ==> show container logs",
        "- containerctl logs <id> -f             ==> follow container logs",
        "- containerctl logs <id> -n 100         ==> show last 100 lines",
        "──────────────────────────────────────────────",
        "- containerctl exec <id> <cmd...>       ==> execute command in container (exits with its code)",
        "- containerctl shell <id>               ==> open shell in container",
        "- containerctl shell <id> -s /bin/bash  ==> open bash in container",
        "──────────────────────────────────────────────",
        "- containerctl enter <project>          ==> enter project (spawn or attach)",
        "- containerctl enter <proj> user=root   ==> enter project as another user",
        "- containerctl enter <shortcut>         ==> enter project, landing in a [shortcuts] dir",
        "- containerctl spawn <project>          ==> force-create project container",
        "- containerctl destroy <project>        ==> stop and remove project container (checks, then asks)",
        "- containerctl destroy <project> -f     ==> same, without asking (--force; DIRTY is still said)",
        "- containerctl projects                 ==> list configured projects",
        "- containerctl projects -w              ==> wide: full shortcut lists, never clip",
        "──────────────────────────────────────────────",
        "- containerctl check <project>          ==> is the workspace safe to destroy? (exit 1 when DIRTY)",
        "- containerctl check --all              ==> check every configured project",
        "- containerctl check-local              ==> run the same check from inside a container",
        "- containerctl issues                   ==> what wants an action: docker down, broken config, shortcuts, ...",
        "──────────────────────────────────────────────",
        "- containerctl service up <name>               ==> create/start a service",
        "- containerctl service down <name>             ==> stop and remove a service",
        "- containerctl service recreate <name>         ==> recreate a service",
        "- containerctl service scripts <name>          ==> list pre-registered scripts",
        "- containerctl service script <name> <script>  ==> run a pre-registered script",
        "- containerctl services                        ==> list configured services",
        "- containerctl services -w                     ==> wide: never clip columns",
        "──────────────────────────────────────────────",
        "- containerctl image ls                        ==> list recipes: current build, versions (nags past 3)",
        "- containerctl image build <name>              ==> build <repo>:cctl-local-NNNNNN, re-point the recipe tag",
        "- containerctl image check [name]              ==> built image vs recipe, local Dockerfile vs upstream",
        "──────────────────────────────────────────────",
        f"Project configs: {PROJECTS_DIR}/<name>.toml",
        f"Service configs: {SERVICES_DIR}/<name>.toml",
        f"Image recipes:   {IMAGES_DIR}/<name>.toml + <name>.Dockerfile [+ <name>/ build context]",
        f"Secrets:         {SECRETS_FILE} (chmod 600; a project's [secrets] picks keys from it)",
    ]
    print("\n" + "\n".join(output_lines) + "\n")
    return 0


def config_usage() -> int:
    """Display example project and service configs (every supported field)"""
    output_lines = [
        f"# {PROJECTS_DIR}/<name>.toml  -  the file name is the project name",
        "# The image must have: the user and home below, bash, git (+ ssh for git@ repos), coreutils;",
        "# python3 too, for `containerctl check-local` to run inside. A spawn into an image without them fails.",
        'image = "ubuntu:24.04"                    # or pin by digest: "ubuntu@sha256:..."',
        'user = "ubuntu"                           # who clones, checks and enters; owns what follows (default: "dev")',
        'home = "/home/ubuntu"                     # its home there (default: /home/<user>, /root for root)',
        'ssh_key = "~/.ssh/id_ed25519"             # copied to <home>/.ssh/id_ed25519 (optional)',
        "repos = [                                 # cloned under <home>/workspace (optional)",
        '  "git@github.com:you/app.git",',
        '  { url = "https://github.com/you/lib.git", dir = "lib", branch = "main" },',
        "]",
        "env_files = [                             # copied into the container (optional)",
        '  { src = "~/.secrets/app.env", dest = "/home/ubuntu/workspace/app/.env" },',
        "]",
        "[shortcuts]                               # `containerctl enter <alias>` lands there (optional)",
        'app = "app"                               # relative to <home>/workspace',
        'lib = "lib/src"',
        "[env]                                     # plain values for the `enter` shell's environment (optional)",
        'TZ = "UTC"',
        "[secrets]                                 # env name = key in secrets.toml, looked up at `enter` (optional)",
        'CLAUDE_CODE_OAUTH_TOKEN = "claude_token"  # another project may expose the same key under another name',
        "",
        f"# {SERVICES_DIR}/<name>.toml  -  the file name is the service name",
        'image = "nginx:1.27"',
        'restart = "unless-stopped"                # no | always | unless-stopped | on-failure',
        'ports = ["127.0.0.1:8080:80"]             # [host_ip:]host:container[/udp]; no ip = all interfaces',
        'volumes = ["~/srv/www:/usr/share/nginx/html:ro"]',
        'env = ["TZ=UTC"]',
        'command = ["nginx", "-g", "daemon off;"]    # argv after the image (optional; default: the image CMD)',
        'networks = ["proxy"]                      # created if missing; `up` also connects an existing container',
        "read_only = true                          # read-only root fs: writable paths must be tmpfs or volumes",
        'tmpfs = ["/tmp", "/run:size=16m"]         # path[:options]',
        'cap_drop = ["ALL"]                        # hardening flags apply on create/recreate, not on `up`',
        'cap_add = ["NET_BIND_SERVICE"]',
        "pids_limit = 256",
        "[scripts]                                 # `containerctl service script <name> reload`",
        'reload = ["nginx", "-s", "reload"]',
        "",
        f"# {IMAGES_DIR}/<name>.toml  -  recipe: <name>.Dockerfile beside it; <name>/ is the build context if present",
        'tag = "hermes:latest"                     # newest build\'s alias, for `image =` (default <name>:latest)',
        'upstream = "https://raw.githubusercontent.com/you/images/main/hermes/Dockerfile"  # `image check` diffs it',
        "",
        f"# {SECRETS_FILE}  -  chmod 600, never versioned; only the keys a project names are read",
        "# Claude Code subscription token: `claude setup-token` on a trusted machine, valid ~1 year.",
        'claude_token = "sk-ant-oat01-..."',
    ]
    print("\n" + "\n".join(output_lines) + "\n")
    return 0


def leave_cleanly() -> None:
    """However we go, go the same way. A closed terminal (SIGHUP) or a `kill` (SIGTERM) becomes an ordinary exit,
    so every `finally` on the way out runs (the TUI gives the terminal back), and at any exit the docker calls
    still in flight in their own process group are killed: nothing else would (see Docker._run)."""

    def leave(signum, frame):
        sys.exit(128 + signum)

    for sig in (signal.SIGHUP, signal.SIGTERM):
        if signal.getsignal(sig) is not signal.SIG_IGN:  # ignored on purpose (`nohup`) stays ignored
            signal.signal(sig, leave)
    atexit.register(Docker.kill_own_group, leaving=True)


def main() -> None:
    """Main entry point"""
    leave_cleanly()
    # Check for help before argparse
    if len(sys.argv) > 1 and sys.argv[1] in ("help", "-h", "--help"):
        if sys.argv[2:] == ["config"]:
            sys.exit(config_usage())
        sys.exit(usage(full=sys.argv[2:] == ["full"]))

    parser = argparse.ArgumentParser(
        description="containerctl - containers CLI/TUI manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # ps
    ps_parser = subparsers.add_parser("ps", help="List containers")
    ps_parser.add_argument("-q", "--quiet", action="store_true", help="Only show IDs")
    ps_parser.add_argument("-r", "--running", action="store_true", help="Only running")
    ps_parser.add_argument("-w", "--wide", action="store_true", help="Never clip columns to the terminal width")
    ps_parser.set_defaults(func=cli_ps)

    # images
    images_parser = subparsers.add_parser("images", help="List images")
    images_parser.add_argument("-q", "--quiet", action="store_true", help="Only show IDs")
    images_parser.set_defaults(func=cli_images)

    # volumes
    volumes_parser = subparsers.add_parser("volumes", help="List volumes")
    volumes_parser.add_argument("-q", "--quiet", action="store_true", help="Only show names")
    volumes_parser.set_defaults(func=cli_volumes)

    # networks
    networks_parser = subparsers.add_parser("networks", help="List networks")
    networks_parser.add_argument("-q", "--quiet", action="store_true", help="Only show IDs")
    networks_parser.set_defaults(func=cli_networks)

    # start
    start_parser = subparsers.add_parser("start", help="Start container")
    start_parser.add_argument("container", help="Container ID or name")
    start_parser.set_defaults(func=cli_start)

    # stop
    stop_parser = subparsers.add_parser("stop", help="Stop container")
    stop_parser.add_argument("container", help="Container ID or name")
    stop_parser.set_defaults(func=cli_stop)

    # restart
    restart_parser = subparsers.add_parser("restart", help="Restart container")
    restart_parser.add_argument("container", help="Container ID or name")
    restart_parser.set_defaults(func=cli_restart)

    # logs
    logs_parser = subparsers.add_parser("logs", help="Show container logs")
    logs_parser.add_argument("container", help="Container ID or name")
    logs_parser.add_argument("-f", "--follow", action="store_true", help="Follow logs")
    logs_parser.add_argument("-n", "--tail", type=int, help="Number of lines")
    logs_parser.set_defaults(func=cli_logs)

    # exec
    exec_parser = subparsers.add_parser("exec", help="Execute command in container")
    exec_parser.add_argument("container", help="Container ID or name")
    exec_parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")
    exec_parser.set_defaults(func=cli_exec)

    # shell
    shell_parser = subparsers.add_parser("shell", help="Open shell in container")
    shell_parser.add_argument("container", help="Container ID or name")
    shell_parser.add_argument("-s", "--shell", default="/bin/sh", help="Shell to use")
    shell_parser.set_defaults(func=cli_shell)

    # enter
    enter_parser = subparsers.add_parser("enter", help="Enter a project (spawn or attach)")
    enter_parser.add_argument("project", help="Project name")
    enter_parser.add_argument("options", nargs="*", metavar="key=value", help="Modifiers: user=<user>")
    enter_parser.set_defaults(func=cli_enter)

    # spawn
    spawn_parser = subparsers.add_parser("spawn", help="Spawn a new project container")
    spawn_parser.add_argument("project", help="Project name")
    spawn_parser.set_defaults(func=cli_spawn)

    # destroy
    destroy_parser = subparsers.add_parser("destroy", help="Destroy a project container")
    destroy_parser.add_argument("project", help="Project name")
    destroy_parser.add_argument("-f", "--force", action="store_true", help="Don't ask, even when DIRTY")
    destroy_parser.set_defaults(func=cli_destroy)

    # check
    check_parser = subparsers.add_parser("check", help="Check if a project workspace is safe to destroy")
    check_parser.add_argument("project", nargs="?", help="Project name")
    check_parser.add_argument("-a", "--all", action="store_true", help="Check all configured projects")
    check_parser.set_defaults(func=cli_check)

    # check-local (runs inside a container; installed to ~/.local/bin on spawn)
    check_local_parser = subparsers.add_parser("check-local", help="Check current workspace (inside a container)")
    check_local_parser.set_defaults(func=cli_check_local)

    # issues
    issues_parser = subparsers.add_parser("issues", help="What wants an action, as the TUI's Issues tab detects it")
    issues_parser.set_defaults(func=cli_issues)

    # projects
    projects_parser = subparsers.add_parser("projects", help="List configured projects")
    projects_parser.add_argument("-w", "--wide", action="store_true", help="Full shortcut lists, never clip")
    projects_parser.set_defaults(func=cli_projects)

    # service (with sub-subcommands)
    service_parser = subparsers.add_parser("service", help="Manage services")
    service_sub = service_parser.add_subparsers(dest="service_command")

    svc_up = service_sub.add_parser("up", help="Create/start a service")
    svc_up.add_argument("service", help="Service name")
    svc_up.set_defaults(func=cli_service_up)

    svc_down = service_sub.add_parser("down", help="Stop and remove a service")
    svc_down.add_argument("service", help="Service name")
    svc_down.set_defaults(func=cli_service_down)

    svc_recreate = service_sub.add_parser("recreate", help="Recreate a service")
    svc_recreate.add_argument("service", help="Service name")
    svc_recreate.set_defaults(func=cli_service_recreate)

    svc_script = service_sub.add_parser("script", help="Run a pre-registered script in a service")
    svc_script.add_argument("service", help="Service name")
    svc_script.add_argument("script", help="Script name")
    svc_script.set_defaults(func=cli_service_script)

    svc_scripts = service_sub.add_parser("scripts", help="List pre-registered scripts for a service")
    svc_scripts.add_argument("service", help="Service name")
    svc_scripts.set_defaults(func=cli_service_scripts)

    # image (with sub-subcommands)
    image_parser = subparsers.add_parser("image", help="Manage image recipes")
    image_sub = image_parser.add_subparsers(dest="image_command")

    img_ls = image_sub.add_parser("ls", help="List recipes: current build, versions, built state")
    img_ls.set_defaults(func=cli_image_ls)

    img_build = image_sub.add_parser("build", help="Build the next <repo>:cctl-local-NNNNNN, re-point the recipe tag")
    img_build.add_argument("image", help="Image name")
    img_build.set_defaults(func=cli_image_build)

    img_check = image_sub.add_parser("check", help="Built image vs recipe, local Dockerfile vs upstream")
    img_check.add_argument("image", nargs="?", help="Image name (default: all)")
    img_check.set_defaults(func=cli_image_check)

    # services (list)
    services_parser = subparsers.add_parser("services", help="List configured services")
    services_parser.add_argument("-w", "--wide", action="store_true", help="Never clip columns")
    services_parser.set_defaults(func=cli_services)

    args = parser.parse_args()

    if args.command is None:
        # No command - launch TUI
        app = App()
        app.run()
    elif args.command == "service" and not hasattr(args, "func"):
        service_parser.print_help()
    elif args.command == "image" and not hasattr(args, "func"):
        image_parser.print_help()
    else:
        # One run, one report: whatever the command's loaders and docker calls logged, as a table on stderr once
        # it is done (a sys.exit inside the command still gets the report).
        try:
            args.func(args)
        except DockerError as e:  # the command needed an answer docker didn't give: it stopped, say why
            log_error("docker.command", e.failure.detail)
            print(f"error: {e} ({e.failure.hint})", file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:  # Ctrl-C: the docker client we were waiting on got it too; no traceback
            print("\ninterrupted", file=sys.stderr)
            sys.exit(130)
        finally:
            if errors := [e for e in get_errors() if not e.echoed]:
                print_errors(errors, file=sys.stderr)


if __name__ == "__main__":
    main()

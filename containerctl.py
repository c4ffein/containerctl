#!/usr/bin/env python3
"""
containerctl - KISS containers CLI/TUI manager
MIT License - Copyright (c) 2026 c4ffein
WARNING: I don't recommand using this as-is. This a PoC, and usable by me because I know what I want to do with it.
- You can use it if you feel that you can edit the code yourself and you can live with my future breaking changes.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass, field
from typing import Any, Callable


# =============================================================================
# Error Registry
# =============================================================================

_errors: list[dict] = []


def log_error(category: str, message: str, context: dict | None = None) -> None:
    """Log a non-fatal error to the global registry"""
    _errors.append({
        "ts": time.time(),
        "cat": category,
        "msg": message,
        "ctx": context or {},
    })


def get_errors() -> list[dict]:
    """Get all logged errors"""
    return _errors


def get_errors_by_category() -> dict[str, list[dict]]:
    """Get errors grouped by category"""
    grouped: dict[str, list[dict]] = {}
    for err in _errors:
        cat = err["cat"]
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append(err)
    return grouped


def clear_errors(category: str) -> int:
    """Clear all errors in a category, returns count cleared"""
    global _errors
    before = len(_errors)
    _errors = [e for e in _errors if e["cat"] != category]
    return before - len(_errors)


def fetch_errors() -> list[dict]:
    """Fetch errors formatted for the TUI tab"""
    grouped = get_errors_by_category()
    if not grouped:
        return [{"Category": "No errors", "Count": "", "Latest": "", "_category": ""}]
    result = []
    for cat, errs in sorted(grouped.items()):
        latest = errs[-1]
        result.append({
            "Category": cat,
            "Count": str(len(errs)),
            "Latest": latest["msg"][:50],
            "_category": cat,  # Hidden field for clearing
        })
    return result


# =============================================================================
# Docker Data Models
# =============================================================================

VALID_ID_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
VALID_IMAGE_REF_CHARS = VALID_ID_CHARS + "/:."  # Images can have repo/name:tag and registry.domain


def is_valid_id(id_str: str) -> bool:
    """Check if ID contains only valid characters (a-z, A-Z, 0-9, -, _)"""
    return all(c in VALID_ID_CHARS for c in id_str)


def is_valid_image_ref(ref_str: str) -> bool:
    """Check if image reference contains only valid characters (allows /, :, . for repo:tag)"""
    return all(c in VALID_IMAGE_REF_CHARS for c in ref_str)


@dataclass
class Container:
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
    Platform: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Container | None":
        """Create Container from dict, logging errors for missing required fields"""
        required = ["ID", "Names", "Image", "Status"]
        missing = [f for f in required if f not in data]
        if missing:
            log_error("docker.container_parse", f"Missing fields: {missing}", {"data": str(data)[:200]})
            return None
        if not is_valid_id(data["ID"]):
            log_error("docker.container_parse", f"Invalid ID: {data['ID']}", {"data": str(data)[:200]})
            return None
        try:
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except TypeError as e:
            log_error("docker.container_parse", str(e), {"data": str(data)[:200]})
            return None

    def to_dict(self) -> dict:
        """Convert to dict for UI compatibility"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Image:
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

    @classmethod
    def from_dict(cls, data: dict) -> "Image | None":
        """Create Image from dict, logging errors for missing required fields"""
        required = ["ID", "Repository", "Tag"]
        missing = [f for f in required if f not in data]
        if missing:
            log_error("docker.image_parse", f"Missing fields: {missing}", {"data": str(data)[:200]})
            return None
        if not is_valid_id(data["ID"]):
            log_error("docker.image_parse", f"Invalid ID: {data['ID']}", {"data": str(data)[:200]})
            return None
        try:
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except TypeError as e:
            log_error("docker.image_parse", str(e), {"data": str(data)[:200]})
            return None

    def to_dict(self) -> dict:
        """Convert to dict for UI compatibility"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Volume:
    """Docker volume data model"""
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

    @classmethod
    def from_dict(cls, data: dict) -> "Volume | None":
        """Create Volume from dict, logging errors for missing required fields"""
        required = ["Name", "Driver"]
        missing = [f for f in required if f not in data]
        if missing:
            log_error("docker.volume_parse", f"Missing fields: {missing}", {"data": str(data)[:200]})
            return None
        if not is_valid_id(data["Name"]):
            log_error("docker.volume_parse", f"Invalid Name: {data['Name']}", {"data": str(data)[:200]})
            return None
        try:
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except TypeError as e:
            log_error("docker.volume_parse", str(e), {"data": str(data)[:200]})
            return None

    def to_dict(self) -> dict:
        """Convert to dict for UI compatibility"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Network:
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

    @classmethod
    def from_dict(cls, data: dict) -> "Network | None":
        """Create Network from dict, logging errors for missing required fields"""
        required = ["ID", "Name", "Driver"]
        missing = [f for f in required if f not in data]
        if missing:
            log_error("docker.network_parse", f"Missing fields: {missing}", {"data": str(data)[:200]})
            return None
        if not is_valid_id(data["ID"]):
            log_error("docker.network_parse", f"Invalid ID: {data['ID']}", {"data": str(data)[:200]})
            return None
        try:
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except TypeError as e:
            log_error("docker.network_parse", str(e), {"data": str(data)[:200]})
            return None

    def to_dict(self) -> dict:
        """Convert to dict for UI compatibility"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


# =============================================================================
# Docker Inspect Data Models
# =============================================================================

@dataclass
class ContainerInspect:
    """Docker container inspect data model (from docker inspect)"""
    Id: str
    Name: str
    Created: str
    State: dict
    Image: str
    Config: dict
    Path: str = ""
    Args: list = field(default_factory=list)
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
    HostConfig: dict = field(default_factory=dict)
    GraphDriver: dict = field(default_factory=dict)
    Mounts: list = field(default_factory=list)
    NetworkSettings: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "ContainerInspect | None":
        """Create ContainerInspect from dict, logging errors for missing required fields"""
        required = ["Id", "Name", "Created", "State", "Image", "Config"]
        missing = [f for f in required if f not in data]
        if missing:
            log_error("docker.container_inspect_parse", f"Missing fields: {missing}", {"data": str(data)[:200]})
            return None
        if not is_valid_id(data["Id"]):
            log_error("docker.container_inspect_parse", f"Invalid Id: {data['Id']}", {"data": str(data)[:200]})
            return None
        try:
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except TypeError as e:
            log_error("docker.container_inspect_parse", str(e), {"data": str(data)[:200]})
            return None

    @property
    def labels(self) -> dict:
        """Get container labels from Config"""
        return self.Config.get("Labels") or {}

    @property
    def name_clean(self) -> str:
        """Get container name without leading slash"""
        return self.Name.lstrip("/")


@dataclass
class ImageInspect:
    """Docker image inspect data model (from docker image inspect)"""
    Id: str
    Created: str
    Architecture: str
    Os: str
    Size: int
    RepoTags: list = field(default_factory=list)
    RepoDigests: list = field(default_factory=list)
    Parent: str = ""
    Comment: str = ""
    DockerVersion: str = ""
    Author: str = ""
    Config: dict = field(default_factory=dict)
    GraphDriver: dict = field(default_factory=dict)
    RootFS: dict = field(default_factory=dict)
    Metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "ImageInspect | None":
        """Create ImageInspect from dict, logging errors for missing required fields"""
        required = ["Id", "Created", "Architecture", "Os", "Size"]
        missing = [f for f in required if f not in data]
        if missing:
            log_error("docker.image_inspect_parse", f"Missing fields: {missing}", {"data": str(data)[:200]})
            return None
        if not is_valid_image_ref(data["Id"]):
            log_error("docker.image_inspect_parse", f"Invalid Id: {data['Id']}", {"data": str(data)[:200]})
            return None
        try:
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except TypeError as e:
            log_error("docker.image_inspect_parse", str(e), {"data": str(data)[:200]})
            return None


@dataclass
class VolumeInspect:
    """Docker volume inspect data model (from docker volume inspect)"""
    Name: str
    Driver: str
    Mountpoint: str
    Scope: str
    CreatedAt: str = ""
    Labels: dict | None = None
    Options: dict | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "VolumeInspect | None":
        """Create VolumeInspect from dict, logging errors for missing required fields"""
        required = ["Name", "Driver", "Mountpoint", "Scope"]
        missing = [f for f in required if f not in data]
        if missing:
            log_error("docker.volume_inspect_parse", f"Missing fields: {missing}", {"data": str(data)[:200]})
            return None
        if not is_valid_id(data["Name"]):
            log_error("docker.volume_inspect_parse", f"Invalid Name: {data['Name']}", {"data": str(data)[:200]})
            return None
        try:
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except TypeError as e:
            log_error("docker.volume_inspect_parse", str(e), {"data": str(data)[:200]})
            return None


@dataclass
class NetworkInspect:
    """Docker network inspect data model (from docker network inspect)"""
    Id: str
    Name: str
    Created: str
    Scope: str
    Driver: str
    EnableIPv4: bool = True
    EnableIPv6: bool = False
    IPAM: dict = field(default_factory=dict)
    Internal: bool = False
    Attachable: bool = False
    Ingress: bool = False
    ConfigFrom: dict = field(default_factory=dict)
    ConfigOnly: bool = False
    Containers: dict = field(default_factory=dict)
    Options: dict = field(default_factory=dict)
    Labels: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "NetworkInspect | None":
        """Create NetworkInspect from dict, logging errors for missing required fields"""
        required = ["Id", "Name", "Created", "Scope", "Driver"]
        missing = [f for f in required if f not in data]
        if missing:
            log_error("docker.network_inspect_parse", f"Missing fields: {missing}", {"data": str(data)[:200]})
            return None
        if not is_valid_id(data["Id"]):
            log_error("docker.network_inspect_parse", f"Invalid Id: {data['Id']}", {"data": str(data)[:200]})
            return None
        try:
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except TypeError as e:
            log_error("docker.network_inspect_parse", str(e), {"data": str(data)[:200]})
            return None


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
    def getch() -> str:
        """Read single keypress in raw mode"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            # Handle escape sequences
            if ch == '\x1b':
                # Try to read more characters for escape sequences
                import select
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch += sys.stdin.read(1)
                    if ch[-1] == '[':
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            ch += sys.stdin.read(1)
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

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

class Docker:
    """Docker CLI wrapper using subprocess"""

    @staticmethod
    def _run(args: list[str], capture: bool = True) -> subprocess.CompletedProcess:
        """Run docker command"""
        cmd = ["docker"] + args
        return subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
        )

    @staticmethod
    def _run_json(args: list[str]) -> list[dict]:
        """Run docker command and parse JSON output"""
        result = Docker._run(args)
        if result.returncode != 0:
            log_error("docker.command", f"docker {' '.join(args)}", {
                "returncode": result.returncode,
                "stderr": result.stderr.strip() if result.stderr else "",
            })
            return []

        items = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log_error("docker.json_parse", str(e), {"line": line[:100]})
        return items

    @staticmethod
    def containers(all_containers: bool = True) -> list[dict]:
        """List containers"""
        args = ["ps", "--format", "{{json .}}"]
        if all_containers:
            args.insert(1, "-a")
        raw = Docker._run_json(args)
        # Claude did this, not me - we should directly use these objects in the long-term, but since it works...
        return [c.to_dict() for c in (Container.from_dict(r) for r in raw) if c is not None]

    @staticmethod
    def images() -> list[dict]:
        """List images"""
        raw = Docker._run_json(["images", "--format", "{{json .}}"])
        # Claude did this, not me - we should directly use these objects in the long-term, but since it works...
        return [i.to_dict() for i in (Image.from_dict(r) for r in raw) if i is not None]

    @staticmethod
    def volumes() -> list[dict]:
        """List volumes"""
        raw = Docker._run_json(["volume", "ls", "--format", "{{json .}}"])
        # Claude did this, not me - we should directly use these objects in the long-term, but since it works...
        return [v.to_dict() for v in (Volume.from_dict(r) for r in raw) if v is not None]

    @staticmethod
    def networks() -> list[dict]:
        """List networks"""
        raw = Docker._run_json(["network", "ls", "--format", "{{json .}}"])
        # Claude did this, not me - we should directly use these objects in the long-term, but since it works...
        return [n.to_dict() for n in (Network.from_dict(r) for r in raw) if n is not None]

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
    def logs(container_id: str, follow: bool = False, tail: int | None = None) -> None:
        """Show container logs (streams to stdout)"""
        if not is_valid_id(container_id):
            log_error("docker.invalid_id", f"Invalid container ID: {container_id}")
            return
        args = ["logs"]
        if follow:
            args.append("-f")
        if tail:
            args.extend(["--tail", str(tail)])
        args.append(container_id)
        subprocess.run(["docker"] + args)

    @staticmethod
    def exec(container_id: str, command: list[str], interactive: bool = True) -> None:
        """Execute command in container"""
        if not is_valid_id(container_id):
            log_error("docker.invalid_id", f"Invalid container ID: {container_id}")
            return
        args = ["exec"]
        if interactive:
            args.extend(["-it"])
        args.append(container_id)
        args.extend(command)
        subprocess.run(["docker"] + args)

    @staticmethod
    def shell(container_id: str, shell: str = "/bin/sh") -> None:
        """Open interactive shell in container"""
        if not is_valid_id(container_id):
            log_error("docker.invalid_id", f"Invalid container ID: {container_id}")
            return
        Docker.exec(container_id, [shell], interactive=True)

    @staticmethod
    def _inspect_raw(object_id: str) -> dict | None:
        """Inspect a Docker object (raw dict output) - caller must validate ID"""
        result = Docker._run(["inspect", object_id])
        if result.returncode != 0:
            return None
        try:
            data = json.loads(result.stdout)
            return data[0] if data else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def inspect_container(container_id: str) -> ContainerInspect | None:
        """Inspect a container, returning typed ContainerInspect"""
        if not is_valid_id(container_id):
            log_error("docker.invalid_id", f"Invalid container ID: {container_id}")
            return None
        raw = Docker._inspect_raw(container_id)
        if not raw:
            return None
        return ContainerInspect.from_dict(raw)

    @staticmethod
    def inspect_image(image_id: str) -> ImageInspect | None:
        """Inspect an image, returning typed ImageInspect"""
        if not is_valid_image_ref(image_id):
            log_error("docker.invalid_id", f"Invalid image ref: {image_id}")
            return None
        raw = Docker._inspect_raw(image_id)
        if not raw:
            return None
        return ImageInspect.from_dict(raw)

    @staticmethod
    def inspect_volume(volume_name: str) -> VolumeInspect | None:
        """Inspect a volume, returning typed VolumeInspect"""
        if not is_valid_id(volume_name):
            log_error("docker.invalid_id", f"Invalid volume name: {volume_name}")
            return None
        raw = Docker._inspect_raw(volume_name)
        if not raw:
            return None
        return VolumeInspect.from_dict(raw)

    @staticmethod
    def inspect_network(network_id: str) -> NetworkInspect | None:
        """Inspect a network, returning typed NetworkInspect"""
        if not is_valid_id(network_id):
            log_error("docker.invalid_id", f"Invalid network ID: {network_id}")
            return None
        raw = Docker._inspect_raw(network_id)
        if not raw:
            return None
        return NetworkInspect.from_dict(raw)

    @staticmethod
    def investigate_source(container_id: str) -> dict:
        """Investigate container source (compose, systemd, devcontainer, etc.)"""
        if not is_valid_id(container_id):
            log_error("docker.invalid_id", f"Invalid container ID: {container_id}")
            return {"source": "error", "details": ["Invalid container ID"]}
        result = {"source": "cli", "details": []}

        # Get container info
        info = Docker.inspect_container(container_id)
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


# =============================================================================
# Background Data Fetcher
# =============================================================================

class DataFetcher:
    """Background thread that fetches Docker data every N seconds"""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self.lock = threading.Lock()
        self.data: dict[str, list[dict]] = {
            "containers": [],
            "images": [],
            "volumes": [],
            "networks": [],
        }
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
        if self._thread:
            self._thread.join(timeout=1.0)

    def get(self, key: str) -> list[dict]:
        """Get cached data for a key (thread-safe)"""
        with self.lock:
            return self.data.get(key, []).copy()

    def _fetch_all(self) -> None:
        """Fetch all data from Docker"""
        containers = Docker.containers()
        images = Docker.images()
        volumes = Docker.volumes()
        networks = Docker.networks()

        with self.lock:
            self.data["containers"] = containers
            self.data["images"] = images
            self.data["volumes"] = volumes
            self.data["networks"] = networks

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
    fetch: Callable[[], list[dict]]
    columns: list[tuple[str, str, int]]  # (key, header, width)
    id_key: str
    actions: dict[str, Callable[[str], bool]]


class UI:
    """UI rendering components - all methods write to a Screen buffer"""

    @staticmethod
    def tab_bar(scr: Screen, tabs: list[Tab], selected: int, width: int, row: int) -> None:
        """Render tab bar"""
        scr.move(row, 1)
        scr.clear_line()

        x = 1
        for i, tab in enumerate(tabs):
            # Errors tab in red if there are errors
            if tab.key == "errors" and _errors:
                if i == selected:
                    text = Term.style(f" {tab.name} ", Term.BOLD, Term.INVERT, Term.RED)
                else:
                    text = Term.style(f" {tab.name} ", Term.RED)
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
            if col_width == 0:  # Flex column
                col_width = remaining_width
            text = header[:col_width].ljust(col_width)
            line += text + " "
            remaining_width -= col_width + 1

        scr.move(row, 2)
        scr.write(Term.style(line, Term.BOLD, Term.CYAN))

    @staticmethod
    def list_row(
        scr: Screen,
        tab: Tab,
        item: dict,
        row: int,
        width: int,
        selected: bool,
        status_colors: dict[str, str] | None = None
    ) -> None:
        """Render a single list row"""
        scr.move(row, 1)
        scr.clear_line()

        line = ""
        remaining_width = width - 2

        for key, header, col_width in tab.columns:
            if col_width == 0:  # Flex column
                col_width = remaining_width

            value = str(item.get(key, ""))
            text = value[:col_width].ljust(col_width)

            # Apply status colors
            if status_colors and key in status_colors:
                status = item.get(key, "").lower()
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
    def confirm(prompt: str, row: int) -> bool:
        """Show confirmation prompt and get y/n response (flushes immediately)"""
        sys.stdout.write(f"\033[{row};1H" + Term.CLEAR_LINE)
        sys.stdout.write(Term.style(f"{prompt} [y/N] ", Term.YELLOW))
        sys.stdout.flush()

        while True:
            key = Term.getch()
            if key.lower() == 'y':
                return True
            if key.lower() == 'n' or key == Term.KEY_ENTER or key == Term.KEY_ESC:
                return False

    @staticmethod
    def info_box(title: str, lines: list[str], rows: int, cols: int) -> None:
        """Display an info box modal and wait for keypress to dismiss"""
        # Calculate box dimensions
        content_width = max(len(title), max((len(line) for line in lines), default=20)) + 4
        box_width = min(content_width, cols - 4)
        box_height = len(lines) + 4
        start_row = (rows - box_height) // 2
        start_col = (cols - box_width) // 2

        # Draw box
        top_border = "┌" + "─" * (box_width - 2) + "┐"
        bottom_border = "└" + "─" * (box_width - 2) + "┘"
        empty_line = "│" + " " * (box_width - 2) + "│"

        sys.stdout.write(f"\033[{start_row};{start_col}H" + Term.style(top_border, Term.CYAN))
        sys.stdout.write(f"\033[{start_row + 1};{start_col}H" + Term.style("│ ", Term.CYAN) + Term.style(title.center(box_width - 4), Term.BOLD) + Term.style(" │", Term.CYAN))
        sys.stdout.write(f"\033[{start_row + 2};{start_col}H" + Term.style("├" + "─" * (box_width - 2) + "┤", Term.CYAN))

        for i, line in enumerate(lines):
            display_line = line[:box_width - 4].ljust(box_width - 4)
            sys.stdout.write(f"\033[{start_row + 3 + i};{start_col}H" + Term.style("│ ", Term.CYAN) + display_line + Term.style(" │", Term.CYAN))

        sys.stdout.write(f"\033[{start_row + 3 + len(lines)};{start_col}H" + Term.style(bottom_border, Term.CYAN))
        sys.stdout.write(f"\033[{start_row + 4 + len(lines)};{start_col}H" + Term.DIM + "[Press any key to close]".center(box_width) + Term.RESET)
        sys.stdout.flush()

        # Wait for keypress
        Term.getch()


# =============================================================================
# Application
# =============================================================================

class App:
    """Main TUI application"""

    def __init__(self):
        self.running = True
        self.current_tab = 0
        self.selected_index = 0
        self.scroll_offset = 0
        self.message_text = ""
        self.message_style = ""
        self.scr = Screen()  # Buffered screen for flicker-free rendering
        self.fetcher = DataFetcher(interval=2.0)  # Background data fetcher

        # Status colors for containers
        self.container_status_colors = {
            "up": Term.GREEN,
            "running": Term.GREEN,
            "exited": Term.RED,
            "created": Term.YELLOW,
            "paused": Term.YELLOW,
        }

        # Define tabs
        self.tabs = [
            Tab(
                name="Containers",
                key="containers",
                fetch=Docker.containers,
                columns=[
                    ("Names", "NAME", 25),
                    ("Status", "STATUS", 20),
                    ("Image", "IMAGE", 30),
                    ("Ports", "PORTS", 0),
                ],
                id_key="ID",
                actions={
                    "s": lambda id: Docker.start(id),
                    "S": lambda id: Docker.stop(id),
                    "r": lambda id: Docker.restart(id),
                },
            ),
            Tab(
                name="Images",
                key="images",
                fetch=Docker.images,
                columns=[
                    ("Repository", "REPOSITORY", 35),
                    ("Tag", "TAG", 20),
                    ("Size", "SIZE", 15),
                    ("ID", "ID", 0),
                ],
                id_key="ID",
                actions={},
            ),
            Tab(
                name="Volumes",
                key="volumes",
                fetch=Docker.volumes,
                columns=[
                    ("Name", "NAME", 50),
                    ("Driver", "DRIVER", 15),
                    ("Mountpoint", "MOUNTPOINT", 0),
                ],
                id_key="Name",
                actions={},
            ),
            Tab(
                name="Networks",
                key="networks",
                fetch=Docker.networks,
                columns=[
                    ("Name", "NAME", 30),
                    ("Driver", "DRIVER", 15),
                    ("Scope", "SCOPE", 10),
                    ("ID", "ID", 0),
                ],
                id_key="ID",
                actions={},
            ),
            Tab(
                name="Errors",
                key="errors",
                fetch=fetch_errors,
                columns=[
                    ("Category", "CATEGORY", 30),
                    ("Count", "COUNT", 8),
                    ("Latest", "LATEST", 0),
                ],
                id_key="_category",
                actions={},
            ),
        ]

    def current(self) -> Tab:
        """Get current tab"""
        return self.tabs[self.current_tab]

    def items(self) -> list[dict]:
        """Get items for current tab from cache"""
        if self.current().key == "errors":
            return fetch_errors()
        return self.fetcher.get(self.current().key)

    def selected_item(self) -> dict | None:
        """Get currently selected item"""
        items = self.items()
        if 0 <= self.selected_index < len(items):
            return items[self.selected_index]
        return None

    def selected_id(self) -> str | None:
        """Get ID of selected item"""
        item = self.selected_item()
        if item:
            return item.get(self.current().id_key)
        return None

    def set_message(self, text: str, style: str = "") -> None:
        """Set status message"""
        self.message_text = text
        self.message_style = style

    def render(self) -> None:
        """Render the entire UI (buffered - single flush)"""
        rows, cols = Term.size()
        scr = self.scr
        items = self.items()  # Get from cache once

        # Adjust selection if needed
        if self.selected_index >= len(items):
            self.selected_index = max(0, len(items) - 1)

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
        UI.tab_bar(scr, self.tabs, self.current_tab, cols, tab_row)

        # Divider
        UI.divider(scr, 2, cols)

        # List header
        UI.list_header(scr, self.current(), header_row, cols)

        # List items
        visible_items = items[self.scroll_offset:self.scroll_offset + list_height]
        for i, item in enumerate(visible_items):
            actual_index = self.scroll_offset + i
            is_selected = actual_index == self.selected_index

            # Apply status colors only for containers tab
            status_colors = None
            if self.current().key == "containers":
                status_colors = self.container_status_colors

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

        # Empty state
        if not items:
            UI.message(
                scr,
                f"  No {self.current().name.lower()} found",
                list_start,
                cols,
                Term.DIM,
            )

        # Clear message line if no message
        if self.message_text:
            UI.message(scr, self.message_text, rows - 1, cols, self.message_style)
        else:
            scr.move(rows - 1, 1)
            scr.clear_line()

        # Status bar with keybindings
        if self.current().key == "containers":
            keys = "[s]tart [S]top [r]estart [L]ogs [e]xec [d]elete [o]rigin [q]uit"
        elif self.current().key == "errors":
            keys = "[c]lear [q]uit"
        else:
            keys = "[d]elete [q]uit"
        keys += f"  │  {len(items)} items"
        UI.status_bar(scr, keys, status_row, cols)

        # Single flush - write everything at once
        scr.flush()

    def handle_key(self, key: str) -> None:
        """Handle keypress"""
        rows, cols = Term.size()
        list_height = rows - 6

        # Clear message on any key
        self.message_text = ""

        # Quit (check first, before any other handlers)
        if key == 'q':
            self.running = False
            return

        # Navigation (jikl: j=left, i=up, k=down, l=right)
        items = self.items()

        if key in ('k', Term.KEY_DOWN):
            if self.selected_index < len(items) - 1:
                self.selected_index += 1
                # Scroll down if needed
                if self.selected_index >= self.scroll_offset + list_height:
                    self.scroll_offset += 1

        elif key in ('i', Term.KEY_UP):
            if self.selected_index > 0:
                self.selected_index -= 1
                # Scroll up if needed
                if self.selected_index < self.scroll_offset:
                    self.scroll_offset = self.selected_index

        elif key == 'g':  # Go to top
            self.selected_index = 0
            self.scroll_offset = 0

        elif key == 'G':  # Go to bottom
            self.selected_index = max(0, len(items) - 1)
            self.scroll_offset = max(0, len(items) - list_height)

        # Tab switching (no refresh needed - background thread handles it)
        elif key == Term.KEY_TAB or key == 'l' or key == Term.KEY_RIGHT:
            self.current_tab = (self.current_tab + 1) % len(self.tabs)
            self.selected_index = 0
            self.scroll_offset = 0

        elif key == 'j' or key == Term.KEY_LEFT:
            self.current_tab = (self.current_tab - 1) % len(self.tabs)
            self.selected_index = 0
            self.scroll_offset = 0

        # Number keys for tabs
        elif key in '12345':
            idx = int(key) - 1
            if idx < len(self.tabs):
                self.current_tab = idx
                self.selected_index = 0
                self.scroll_offset = 0

        # Container-specific actions
        elif self.current().key == "containers" and self.selected_id():
            container_id = self.selected_id()

            if key == 's':  # Start
                if Docker.start(container_id):
                    self.set_message(f"Started {container_id[:12]}", Term.GREEN)
                else:
                    self.set_message(f"Failed to start {container_id[:12]}", Term.RED)
                
            elif key == 'S':  # Stop
                if Docker.stop(container_id):
                    self.set_message(f"Stopped {container_id[:12]}", Term.GREEN)
                else:
                    self.set_message(f"Failed to stop {container_id[:12]}", Term.RED)
                
            elif key == 'r':  # Restart
                if Docker.restart(container_id):
                    self.set_message(f"Restarted {container_id[:12]}", Term.GREEN)
                else:
                    self.set_message(f"Failed to restart {container_id[:12]}", Term.RED)
                
            elif key == 'L':  # Logs
                self.exit_tui_temporarily()
                Docker.logs(container_id, tail=100)
                input("\nPress Enter to return...")
                self.enter_tui()

            elif key == 'e':  # Exec
                self.exit_tui_temporarily()
                Docker.shell(container_id)
                self.enter_tui()
                
            elif key == 'd':  # Delete
                name = self.selected_item().get("Names", container_id[:12])
                if UI.confirm(f"Delete container '{name}'?", rows - 1):
                    if Docker.remove_container(container_id, force=True):
                        self.set_message(f"Deleted {name}", Term.GREEN)
                    else:
                        self.set_message(f"Failed to delete {name}", Term.RED)

            elif key == 'o':  # Origin/source
                name = self.selected_item().get("Names", container_id[:12])
                result = Docker.investigate_source(container_id)
                title = f"Source: {result['source']}"
                lines = [f"Container: {name}"] + result["details"]
                UI.info_box(title, lines, rows, cols)

        # Image deletion
        elif self.current().key == "images" and key == 'd' and self.selected_id():
            image_id = self.selected_id()
            repo = self.selected_item().get("Repository", image_id[:12])
            tag = self.selected_item().get("Tag", "")
            name = f"{repo}:{tag}" if tag else repo

            if UI.confirm(f"Delete image '{name}'?", rows - 1):
                if Docker.remove_image(image_id, force=True):
                    self.set_message(f"Deleted {name}", Term.GREEN)
                else:
                    self.set_message(f"Failed to delete {name}", Term.RED)
                
        # Volume deletion
        elif self.current().key == "volumes" and key == 'd' and self.selected_id():
            volume_name = self.selected_id()

            if UI.confirm(f"Delete volume '{volume_name}'?", rows - 1):
                if Docker.remove_volume(volume_name):
                    self.set_message(f"Deleted {volume_name}", Term.GREEN)
                else:
                    self.set_message(f"Failed to delete {volume_name}", Term.RED)
                
        # Network deletion
        elif self.current().key == "networks" and key == 'd' and self.selected_id():
            network_id = self.selected_id()
            name = self.selected_item().get("Name", network_id[:12])

            if UI.confirm(f"Delete network '{name}'?", rows - 1):
                if Docker.remove_network(network_id):
                    self.set_message(f"Deleted {name}", Term.GREEN)
                else:
                    self.set_message(f"Failed to delete {name}", Term.RED)

        # Error clearing
        elif self.current().key == "errors" and key == 'c' and self.selected_id():
            category = self.selected_id()
            if category:  # Not the "No errors" placeholder
                if UI.confirm(f"Clear all '{category}' errors?", rows - 1):
                    count = clear_errors(category)
                    self.set_message(f"Cleared {count} errors from {category}", Term.GREEN)

    def enter_tui(self) -> None:
        """Enter TUI mode"""
        print(Term.ALT_SCREEN + Term.HIDE_CURSOR, end="", flush=True)

    def exit_tui_temporarily(self) -> None:
        """Exit TUI mode temporarily (for exec/logs)"""
        print(Term.SHOW_CURSOR + Term.MAIN_SCREEN, end="", flush=True)

    def exit_tui(self) -> None:
        """Exit TUI mode completely"""
        print(Term.SHOW_CURSOR + Term.MAIN_SCREEN, end="", flush=True)

    def run(self) -> None:
        """Main TUI loop"""
        # Set up signal handler for window resize
        def handle_resize(signum, frame):
            self.render()

        signal.signal(signal.SIGWINCH, handle_resize)

        # Start background data fetcher
        self.fetcher.start()

        self.enter_tui()
        try:
            while self.running:
                self.render()
                key = Term.getch()
                self.handle_key(key)
        finally:
            self.fetcher.stop()
            self.exit_tui()


# =============================================================================
# CLI
# =============================================================================

def cli_ps(args: argparse.Namespace) -> None:
    """List containers"""
    containers = Docker.containers(all_containers=not args.running)

    if args.quiet:
        for c in containers:
            print(c.get("ID", ""))
        return

    # Print table
    print(f"{'NAME':<25} {'STATUS':<20} {'IMAGE':<30} {'PORTS'}")
    print("-" * 80)
    for c in containers:
        name = c.get("Names", "")[:24]
        status = c.get("Status", "")[:19]
        image = c.get("Image", "")[:29]
        ports = c.get("Ports", "")
        print(f"{name:<25} {status:<20} {image:<30} {ports}")


def cli_images(args: argparse.Namespace) -> None:
    """List images"""
    images = Docker.images()

    if args.quiet:
        for img in images:
            print(img.get("ID", ""))
        return

    print(f"{'REPOSITORY':<35} {'TAG':<20} {'SIZE':<15} {'ID'}")
    print("-" * 80)
    for img in images:
        repo = img.get("Repository", "")[:34]
        tag = img.get("Tag", "")[:19]
        size = img.get("Size", "")[:14]
        id_ = img.get("ID", "")[:12]
        print(f"{repo:<35} {tag:<20} {size:<15} {id_}")


def cli_volumes(args: argparse.Namespace) -> None:
    """List volumes"""
    volumes = Docker.volumes()

    if args.quiet:
        for v in volumes:
            print(v.get("Name", ""))
        return

    print(f"{'NAME':<50} {'DRIVER':<15}")
    print("-" * 65)
    for v in volumes:
        name = v.get("Name", "")[:49]
        driver = v.get("Driver", "")[:14]
        print(f"{name:<50} {driver:<15}")


def cli_networks(args: argparse.Namespace) -> None:
    """List networks"""
    networks = Docker.networks()

    if args.quiet:
        for n in networks:
            print(n.get("ID", ""))
        return

    print(f"{'NAME':<30} {'DRIVER':<15} {'SCOPE':<10} {'ID'}")
    print("-" * 70)
    for n in networks:
        name = n.get("Name", "")[:29]
        driver = n.get("Driver", "")[:14]
        scope = n.get("Scope", "")[:9]
        id_ = n.get("ID", "")[:12]
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
    Docker.logs(args.container, follow=args.follow, tail=args.tail)


def cli_exec(args: argparse.Namespace) -> None:
    """Execute command in container"""
    Docker.exec(args.container, args.command)


def cli_shell(args: argparse.Namespace) -> None:
    """Open shell in container"""
    Docker.shell(args.container, args.shell)


def usage() -> int:
    """Display usage information"""
    output_lines = [
        "containerctl - KISS containers CLI/TUI manager",
        "──────────────────────────────────────────────",
        "- containerctl                          ==> launch TUI",
        "- containerctl help                     ==> show this help",
        "──────────────────────────────────────────────",
        "- containerctl ps                       ==> list containers",
        "- containerctl ps -q                    ==> list container IDs only",
        "- containerctl ps -r                    ==> list running containers only",
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
        "- containerctl exec <id> <cmd...>       ==> execute command in container",
        "- containerctl shell <id>               ==> open shell in container",
        "- containerctl shell <id> -s /bin/bash  ==> open bash in container",
    ]
    print("\n" + "\n".join(output_lines) + "\n")
    return 0


def main() -> None:
    """Main entry point"""
    # Check for help before argparse
    if len(sys.argv) > 1 and sys.argv[1] in ("help", "-h", "--help"):
        sys.exit(usage())

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
    exec_parser.add_argument("command", nargs="+", help="Command to run")
    exec_parser.set_defaults(func=cli_exec)

    # shell
    shell_parser = subparsers.add_parser("shell", help="Open shell in container")
    shell_parser.add_argument("container", help="Container ID or name")
    shell_parser.add_argument("-s", "--shell", default="/bin/sh", help="Shell to use")
    shell_parser.set_defaults(func=cli_shell)

    args = parser.parse_args()

    if args.command is None:
        # No command - launch TUI
        app = App()
        app.run()
    else:
        # Run CLI command
        args.func(args)
        # Output errors to stderr as JSON if any
        if _errors:
            sys.stderr.write(json.dumps({"errors": _errors}, indent=2) + "\n")


if __name__ == "__main__":
    main()

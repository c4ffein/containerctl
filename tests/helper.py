"""Test helpers: fakes that capture the argv built for the container runtime.

containerctl shells out to `docker`. None of these tests need a real daemon — we
replace the two places a subprocess is launched and assert on the command that
*would* have run:

  - ``Docker._run`` for the captured-output commands (start/stop/rm/run/...)
  - ``containerctl.subprocess.run`` for the streaming ones (logs/exec/shell)
"""

import subprocess

import containerctl
from containerctl import DockerContainerRow


def container(
    name: str, state: str = "running", id_: str = "abc", image: str = "i", labels: str = ""
) -> DockerContainerRow:
    """A `docker ps` row as the parser would return it"""
    return DockerContainerRow(ID=id_, Names=name, Image=image, Status=f"Up ({state})", State=state, Labels=labels)


def as_inspected(
    service: containerctl.ServiceConfig,
    secrets: dict | None = None,
    id_: str = "abc",
    image_id: str = "sha256:" + "1" * 64,
) -> containerctl.DockerContainerInspect:
    """What docker answers an inspect of a container made from `service` as it is now, as far as
    service_differences reads it: a stand-in, so only as right as our reading of docker (tests/test_real_docker.py
    asks the real one). `secrets` is secrets.toml when it was made."""
    env = dict(service.env) | {name: (secrets or {}).get(key, "?") for name, key in service.secrets.items()}
    bindings: dict = {}
    for spec in service.ports:
        for port, ip, to in sorted(containerctl.port_bindings(spec)):
            bindings.setdefault(port, []).append({"HostIp": ip, "HostPort": to})
    log = {"Type": "journald", "Config": {}}  # the daemon's own, whatever it is
    if service.log_max_size:
        log = {"Type": "json-file", "Config": {"max-size": service.log_max_size, "max-file": str(service.log_max_file)}}
    host = {
        "RestartPolicy": {"Name": service.restart, "MaximumRetryCount": 0},
        "PortBindings": bindings,
        "Binds": list(service.volumes) or None,
        "ReadonlyRootfs": service.read_only,
        "Tmpfs": dict(spec.partition(":")[::2] for spec in service.tmpfs) or None,
        "CapDrop": list(service.cap_drop) or None,
        "CapAdd": list(service.cap_add) or None,
        "PidsLimit": service.pids_limit or None,
        "Memory": containerctl.size_bytes(service.memory) if service.memory else 0,
        "NanoCpus": containerctl.nano_cpus(service.cpus) if service.cpus else 0,
        "SecurityOpt": ["no-new-privileges"] if service.no_new_privileges else None,
        "LogConfig": log,
    }
    labels = {containerctl.LABEL_SERVICE: service.name, containerctl.LABEL_ENV: ",".join(env)}
    if service.command:
        labels[containerctl.LABEL_COMMAND] = "1"
    config = {
        "Image": service.image,
        "User": service.user,
        "Cmd": list(service.command) or ["the-image's-own"],
        "Env": ["PATH=/usr/local/bin:/usr/bin", "IMAGE_OWN=1", *(f"{name}={value}" for name, value in env.items())],
        "Labels": labels,
    }
    return containerctl.DockerContainerInspect(
        Id=id_, Name=f"/{service.container_name}", Created="", State={}, Image=image_id, Config=config, HostConfig=host
    )


def listings(failed=None, **rows) -> dict:
    """The four docker listings as the fetcher gets them: `rows` by tab key (none by default), `failed` the
    keys whose call failed mapped to docker's stderr"""
    result = {key: containerctl.Listing(list(rows.get(key, []))) for key in containerctl.LISTINGS}
    for key, stderr in (failed or {}).items():
        call = subprocess.CompletedProcess(["docker", key], 1, "", stderr)
        result[key] = containerctl.Listing(failure=containerctl.docker_failure([key], call))
    return result


def rejected(parse, *args) -> containerctl.ConfigError:
    """Run a parser on input it must refuse: the ConfigError it raised"""
    try:
        parse(*args)
    except containerctl.ConfigError as e:
        return e
    raise AssertionError(f"{parse.__qualname__} accepted {args}")


def reset_errors():
    """Clear the module-global error registry between tests."""
    containerctl._errors.clear()


class RunRecorder:
    """Stand-in for ``Docker._run``: records argv (sans leading 'docker').

    Returns a canned ``CompletedProcess`` so callers that inspect ``returncode``
    or ``stdout`` keep working without a daemon.
    """

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.calls = []
        self.stdins = []  # one entry per call: the stdin= of a no-context `build -` lands here
        self.inputs = []  # one entry per call: the input= string of an `exec -i` lands here
        self.own_groups = []  # one entry per call: was it taken out of the terminal's process group?
        self.envs = []  # one entry per call: the env= of a `run -e NAME` lands here
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def __call__(self, args, capture=True, stdin=None, input=None, timeout=None, own_group=False, env=None):
        self.calls.append(list(args))
        self.envs.append(env)
        self.own_groups.append(own_group)
        self.stdins.append(stdin)
        self.inputs.append(input)
        return subprocess.CompletedProcess(
            args=["docker", *args],
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )

    @property
    def last(self):
        return self.calls[-1] if self.calls else None


class ApiRecorder:
    """Stand-in for ``Docker._api`` (the questions asked of the daemon on its socket): every path gets the same
    ``status`` and ``body``. 200 by default: whatever is asked about is there, as a RunRecorder's exit code 0 says
    every command worked; 404 is docker saying there is no such thing. With ``into``, each question is also noted
    in that RunRecorder's calls as ``["GET", path]``: one list, in order, for a test of what came before what."""

    def __init__(self, status=200, body="{}", into=None):
        self.calls, self.status, self.body, self.into = [], status, body, into

    def __call__(self, path, timeout=None):
        self.calls.append(path)
        if self.into is not None:
            self.into.calls.append(["GET", path])
            for column in (self.into.own_groups, self.into.stdins, self.into.inputs, self.into.envs):
                column.append(None)  # these go by the index of the call
        return self.status, self.body


class PopenRecorder:
    """Stand-in for ``subprocess.Popen``, which only the listings go through (they run in their own process
    group, tracked: see ``Docker._run_own_group``). Records each cmd, kwargs and the fake process handed back;
    with ``hang`` the first ``communicate`` of each raises TimeoutExpired, as for a daemon that does not answer."""

    def __init__(self, returncode=0, stdout="", stderr="", hang=False):
        self.calls, self.kwargs, self.procs = [], [], []
        self.returncode, self.stdout, self.stderr, self.hang = returncode, stdout, stderr, hang

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        self.kwargs.append(dict(kwargs))
        self.procs.append(FakeProcess(self, cmd))
        return self.procs[-1]


class FakeProcess:
    def __init__(self, recorder, cmd):
        self.recorder, self.cmd, self.returncode = recorder, cmd, recorder.returncode
        self.killed, self.timeouts = False, []  # the timeout= of each communicate

    def communicate(self, timeout=None):
        self.timeouts.append(timeout)
        if self.recorder.hang and not self.killed:
            raise subprocess.TimeoutExpired(self.cmd, timeout)
        return self.recorder.stdout, self.recorder.stderr

    def kill(self):
        self.killed = True


class SubprocessRecorder:
    """Stand-in for ``subprocess.run``: records the full cmd list (incl. 'docker')."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.calls = []
        self.kwargs = []  # one dict per call: the env= of an `exec -e NAME` lands here
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append(list(cmd))
        self.kwargs.append(dict(kwargs))
        return subprocess.CompletedProcess(
            args=list(cmd),
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )

    @property
    def last(self):
        return self.calls[-1] if self.calls else None

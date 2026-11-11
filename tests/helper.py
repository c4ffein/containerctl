"""Test helpers: fakes that capture the argv built for the container runtime.

containerctl shells out to `docker`. None of these tests need a real daemon — we
replace the two places a subprocess is launched and assert on the command that
*would* have run:

  - ``Docker._run`` for the captured-output commands (start/stop/rm/run/...)
  - ``containerctl.subprocess.run`` for the streaming ones (logs/exec/shell)
"""

import subprocess

import containerctl
from containerctl import Container


def container(name: str, state: str = "running", id_: str = "abc", image: str = "i") -> Container:
    """A `docker ps` row as the parser would return it"""
    return Container(ID=id_, Names=name, Image=image, Status=f"Up ({state})", State=state)


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
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def __call__(self, args, capture=True, stdin=None, input=None, timeout=None, own_group=False):
        self.calls.append(list(args))
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

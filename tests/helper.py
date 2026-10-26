"""Test helpers: fakes that capture the argv built for the container runtime.

containerctl shells out to `docker`. None of these tests need a real daemon — we
replace the two places a subprocess is launched and assert on the command that
*would* have run:

  - ``Docker._run`` for the captured-output commands (start/stop/rm/run/...)
  - ``containerctl.subprocess.run`` for the streaming ones (logs/exec/shell)
"""

import subprocess

import containerctl


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
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def __call__(self, args, capture=True, stdin=None):
        self.calls.append(list(args))
        self.stdins.append(stdin)
        return subprocess.CompletedProcess(
            args=["docker", *args],
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )

    @property
    def last(self):
        return self.calls[-1] if self.calls else None


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

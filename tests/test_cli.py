"""Smoke tests for the CLI surface: help output and argparse wiring."""

import argparse
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import containerctl
from containerctl import usage
from tests.helper import reset_errors


class UsageTests(unittest.TestCase):
    def test_usage_returns_zero_and_prints_help(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = usage()
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("containerctl - KISS containers CLI/TUI manager", out)
        self.assertIn("service up", out)

    def test_readme_help_block_matches_usage(self):
        """The README '## Help' code block must match usage() output byte-for-byte.

        The config-dir lines depend on $HOME, so the dirs are patched to their
        canonical `~` form — the form the README is generated with.
        """
        buf = io.StringIO()
        with (
            mock.patch.object(containerctl, "PROJECTS_DIR", "~/.config/containerctl/projects"),
            mock.patch.object(containerctl, "SERVICES_DIR", "~/.config/containerctl/services"),
            redirect_stdout(buf),
        ):
            usage()
        expected = buf.getvalue().strip()

        readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
        help_start = readme.find("## Help")
        self.assertNotEqual(help_start, -1, "README should have a '## Help' section")
        block_start = readme.find("```", help_start)
        self.assertNotEqual(block_start, -1, "README should have a code block after ## Help")
        block_end = readme.find("```", block_start + 3)
        self.assertNotEqual(block_end, -1, "README code block should be closed")
        readme_block = readme[block_start + 3 : block_end].strip()
        self.assertEqual(readme_block, expected, "README help block drifted from usage() — regenerate it")


class MainHelpDispatchTests(unittest.TestCase):
    def setUp(self):
        reset_errors()

    def test_help_argument_invokes_usage_and_exits_zero(self):
        for flag in ("help", "-h", "--help"):
            with mock.patch.object(containerctl.sys, "argv", ["containerctl", flag]):
                with redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as ctx:
                        containerctl.main()
            self.assertEqual(ctx.exception.code, 0)


class CliEnterOptionTests(unittest.TestCase):
    """cli_enter parses key=value modifiers before delegating to Docker."""

    def setUp(self):
        reset_errors()
        self.project = object()
        self._load = mock.patch.object(containerctl, "load_projects", return_value={"proj": self.project})
        self._load.start()

    def tearDown(self):
        self._load.stop()

    def _enter(self, options):
        with mock.patch.object(containerctl.Docker, "enter_project") as enter:
            containerctl.cli_enter(argparse.Namespace(project="proj", options=options))
        return enter

    def test_no_options_defaults_user(self):
        self._enter([]).assert_called_once_with(self.project, user=None)

    def test_user_option_is_forwarded(self):
        self._enter(["user=root"]).assert_called_once_with(self.project, user="root")

    def test_bad_options_exit_nonzero(self):
        for options in (["shell=zsh"], ["user"], ["user="], ["user=ro;ot"]):
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    containerctl.cli_enter(argparse.Namespace(project="proj", options=options))
            self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()

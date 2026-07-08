"""Smoke tests for the CLI surface: help output and argparse wiring."""

import argparse
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
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

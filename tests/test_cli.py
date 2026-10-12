"""Smoke tests for the CLI surface: help output and argparse wiring."""

import argparse
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import containerctl
from containerctl import ProjectConfig, config_usage, format_table, usage
from tests.helper import RunRecorder, reset_errors


def readme_block(section):
    """The first fenced code block under `## <section>` in the README, stripped."""
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    start = readme.find(f"## {section}")
    assert start != -1, f"README should have a '## {section}' section"
    block_start = readme.find("```", start)
    assert block_start != -1, f"README should have a code block after ## {section}"
    block_end = readme.find("```", block_start + 3)
    assert block_end != -1, "README code block should be closed"
    return readme[block_start + 3 : block_end].strip()


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
            mock.patch.object(containerctl, "SECRETS_FILE", "~/.config/containerctl/secrets.toml"),
            redirect_stdout(buf),
        ):
            usage()
        expected = buf.getvalue().strip()
        self.assertEqual(readme_block("Help"), expected, "README help block drifted from usage() — regenerate it")

    def test_config_examples_parse_as_valid_configs(self):
        """`help config` is the only place the config shape is documented (the README stays a mirror
        of `help`), so the samples must stay loadable: split on the `# <path>` headers, write each out, parse."""
        import tempfile

        buf = io.StringIO()
        with redirect_stdout(buf):
            config_usage()
        project_part, rest = buf.getvalue().strip().split(f"# {containerctl.SERVICES_DIR}")
        service_part, secrets_part = rest.split(f"# {containerctl.SECRETS_FILE}")
        with tempfile.TemporaryDirectory() as d:
            prj = Path(d) / "demo.toml"
            prj.write_text(project_part)
            svc = Path(d) / "web.toml"
            svc.write_text(service_part.split("\n", 1)[1])
            sec = Path(d) / "secrets.toml"
            sec.write_text(secrets_part.split("\n", 1)[1])
            sec.chmod(0o600)
            project = ProjectConfig.from_toml(prj)
            service = containerctl.ServiceConfig.from_toml(svc)
            with mock.patch.object(containerctl, "SECRETS_FILE", sec):
                env = containerctl.resolve_env(project)
        self.assertIsNotNone(project, containerctl.get_errors())
        self.assertIsNotNone(service, containerctl.get_errors())
        self.assertEqual(sorted(project.shortcuts), ["app", "lib"])
        self.assertEqual(project.secrets, {"CLAUDE_CODE_OAUTH_TOKEN": "claude_token"})
        self.assertEqual(env, {"TZ": "UTC", "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-..."})
        self.assertEqual(containerctl.get_errors(), [])
        self.assertEqual(len(project.repos), 2)
        self.assertEqual(service.ports, ["127.0.0.1:8080:80"])
        self.assertIn("reload", service.scripts)
        self.assertEqual(service.networks, ["proxy"])
        self.assertTrue(service.read_only)
        self.assertEqual((service.cap_drop, service.cap_add, service.pids_limit), (["ALL"], ["NET_BIND_SERVICE"], 256))


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

    def test_help_config_prints_samples(self):
        buf = io.StringIO()
        with mock.patch.object(containerctl.sys, "argv", ["containerctl", "help", "config"]):
            with redirect_stdout(buf):
                with self.assertRaises(SystemExit) as ctx:
                    containerctl.main()
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("[shortcuts]", buf.getvalue())
        self.assertIn("[scripts]", buf.getvalue())


class FormatTableTests(unittest.TestCase):
    HEADERS = ["ID", "NAME", "PORTS", "NETWORKS"]
    ROWS = [["abcdef123456", "web", "0.0.0.0:8080->80/tcp", "bridge, cctl-net-demo, extra-long-net"]]

    def test_no_width_never_clips(self):
        lines = format_table(self.HEADERS, self.ROWS, None, shrink=[3, 2])
        self.assertEqual(lines[0].split(), self.HEADERS)
        self.assertTrue(lines[2].endswith("bridge, cctl-net-demo, extra-long-net"))

    def test_widest_column_gives_up_space_first(self):
        # NETWORKS (37) is wider than PORTS (20): it alone absorbs a small excess
        lines = format_table(self.HEADERS, self.ROWS, 60, shrink=[3, 2])
        self.assertTrue(all(len(ln) <= 60 for ln in lines), lines)
        self.assertIn("abcdef123456", lines[2])  # ID untouched
        self.assertIn("0.0.0.0:8080->80/tcp", lines[2])  # PORTS untouched
        self.assertTrue(lines[2].endswith("…"))

    def test_clipped_columns_end_up_balanced(self):
        lines = format_table(self.HEADERS, self.ROWS, 50, shrink=[3, 2])
        self.assertTrue(all(len(ln) <= 50 for ln in lines), lines)
        self.assertEqual(lines[2].count("…"), 2)  # both long columns clipped...
        ports, networks = lines[2].split()[-2:]
        self.assertLessEqual(abs(len(ports) - len(networks)), 1)  # ...to about the same width

    def test_floor_stops_shrinking(self):
        lines = format_table(self.HEADERS, self.ROWS, 20, shrink=[3, 2])
        self.assertTrue(all(len(ln) == 12 + 1 + 4 + 1 + 8 + 1 + 8 for ln in lines[:1]), lines)  # can't fit: floors hold

    def test_empty_rows_prints_headers_only(self):
        lines = format_table(self.HEADERS, [], 20, shrink=[3])
        self.assertEqual(len(lines), 2)


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
        self._enter([]).assert_called_once_with(self.project, user=None, workdir=None)

    def test_user_option_is_forwarded(self):
        self._enter(["user=root"]).assert_called_once_with(self.project, user="root", workdir=None)

    def test_shortcut_resolves_to_project_and_workdir(self):
        project = ProjectConfig(name="proj", image="ubuntu", shortcuts={"lib": "lib/src"})
        with mock.patch.object(containerctl, "load_projects", return_value={"proj": project}):
            with mock.patch.object(containerctl.Docker, "enter_project") as enter:
                containerctl.cli_enter(argparse.Namespace(project="lib", options=["user=root"]))
        enter.assert_called_once_with(project, user="root", workdir="/home/dev/workspace/lib/src")

    def test_unknown_name_lists_projects_and_shortcuts(self):
        project = ProjectConfig(name="proj", image="ubuntu", shortcuts={"lib": "lib"})
        err = io.StringIO()
        with mock.patch.object(containerctl, "load_projects", return_value={"proj": project}):
            with redirect_stderr(err):
                with self.assertRaises(SystemExit) as ctx:
                    containerctl.cli_enter(argparse.Namespace(project="nope", options=[]))
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("Available shortcuts: lib", err.getvalue())

    def test_bad_options_exit_nonzero(self):
        for options in (["shell=zsh"], ["user"], ["user="], ["user=ro;ot"]):
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    containerctl.cli_enter(argparse.Namespace(project="proj", options=options))
            self.assertEqual(ctx.exception.code, 1)


class ListingsQueryDockerOnceTests(unittest.TestCase):
    """`projects` / `services` must not run one `docker ps` per configured item (13 projects = 13 calls)."""

    PS = ["ps", "--size=false", "-a", "--format", "{{json .}}"]

    def _ps_rows(self, *names):
        return "\n".join(
            f'{{"ID": "id{i}", "Names": "{n}", "Image": "i", "Status": "Up", "State": "running"}}'
            for i, n in enumerate(names)
        )

    def test_projects_lists_status_with_a_single_docker_ps(self):
        projects = {n: ProjectConfig(name=n, image="img", shortcuts={"x": "x"} if n == "b" else {}) for n in "abc"}
        rec = RunRecorder(stdout=self._ps_rows("cctl-prj-a", "cctl-prj-c"))
        out = io.StringIO()
        with (
            mock.patch.object(containerctl, "load_projects", return_value=projects),
            mock.patch.object(containerctl.Docker, "_run", rec),
            redirect_stdout(out),
        ):
            containerctl.cli_projects(argparse.Namespace(wide=False))
        self.assertEqual(rec.calls, [self.PS])
        rows = {ln.split()[0]: ln for ln in out.getvalue().splitlines()[2:]}
        self.assertIn("running", rows["a"])
        self.assertIn("not created", rows["b"])
        self.assertIn("x", rows["b"].split()[-1])
        self.assertIn("running", rows["c"])

    def test_services_lists_status_with_a_single_docker_ps(self):
        services = {n: containerctl.ServiceConfig(name=n, image="img", networks=["proxy"]) for n in "ab"}
        rec = RunRecorder(stdout=self._ps_rows("cctl-svc-b"))
        out = io.StringIO()
        with (
            mock.patch.object(containerctl, "load_services", return_value=services),
            mock.patch.object(containerctl.Docker, "_run", rec),
            redirect_stdout(out),
        ):
            containerctl.cli_services(argparse.Namespace(wide=False))
        self.assertEqual(rec.calls, [self.PS])
        rows = {ln.split()[0]: ln for ln in out.getvalue().splitlines()[2:]}
        self.assertIn("not created", rows["a"])
        self.assertIn("running", rows["b"])
        self.assertTrue(rows["b"].endswith("proxy"))


class ProjectsShortcutsColumnTests(unittest.TestCase):
    """Long shortcut lists collapse to a count when the row would not fit the terminal."""

    def setUp(self):
        many = {f"alias-{i}": f"dir{i}" for i in range(14)}
        self.projects = {
            "big": ProjectConfig(name="big", image="dev:latest", shortcuts=many),
            "small": ProjectConfig(name="small", image="dev:latest", shortcuts={"a": "a", "b": "b"}),
            "none": ProjectConfig(name="none", image="dev:latest"),
        }

    def _run(self, width, wide=False):
        out = io.StringIO()
        with (
            mock.patch.object(containerctl, "load_projects", return_value=self.projects),
            mock.patch.object(containerctl.Docker, "_run", RunRecorder()),
            mock.patch.object(containerctl, "_table_width", return_value=None if wide else width),
            redirect_stdout(out),
        ):
            containerctl.cli_projects(argparse.Namespace(wide=wide))
        return out.getvalue().splitlines()

    def test_narrow_terminal_collapses_only_the_long_list(self):
        lines = self._run(80)
        self.assertTrue(all(len(ln) <= 80 for ln in lines), lines)
        rows = {ln.split()[0]: ln for ln in lines[2:]}
        self.assertTrue(rows["big"].endswith("(14 shortcuts)"))
        self.assertTrue(rows["small"].endswith("a, b"))
        self.assertTrue(rows["none"].endswith("-"))

    def test_wide_terminal_shows_everything(self):
        lines = self._run(300)
        self.assertIn("alias-13", lines[2])

    def test_wide_flag_or_pipe_never_collapses(self):
        lines = self._run(80, wide=True)
        self.assertIn("alias-13", lines[2])

    def test_singular_shortcut(self):
        self.projects["big"].shortcuts = {"only-one-very-long-alias-name-" + "x" * 60: "d"}
        rows = {ln.split()[0]: ln for ln in self._run(80)[2:]}
        self.assertTrue(rows["big"].endswith("(1 shortcut)"))


if __name__ == "__main__":
    unittest.main()

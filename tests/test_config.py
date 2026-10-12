"""Tests for TOML project/service config parsing and validation."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import containerctl
from containerctl import (
    ProjectConfig,
    RepoSpec,
    ServiceConfig,
    get_errors,
    parse_env_table,
    parse_secrets_table,
    resolve_enter_target,
    resolve_env,
)
from tests.helper import reset_errors


def write_toml(directory, name, content):
    path = Path(directory) / f"{name}.toml"
    path.write_text(content, encoding="utf-8")
    return path


class RepoSpecTests(unittest.TestCase):
    def test_name_derived_from_url(self):
        self.assertEqual(RepoSpec("https://github.com/u/repo.git").name, "repo")
        self.assertEqual(RepoSpec("https://github.com/u/repo").name, "repo")
        self.assertEqual(RepoSpec("https://github.com/u/repo/").name, "repo")

    def test_explicit_dir_wins(self):
        self.assertEqual(RepoSpec("https://github.com/u/repo.git", dir="custom").name, "custom")


class ProjectConfigTests(unittest.TestCase):
    def setUp(self):
        reset_errors()
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_minimal_valid(self):
        path = write_toml(self.dir, "myproj", 'image = "ubuntu:22.04"\n')
        cfg = ProjectConfig.from_toml(path)
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.name, "myproj")
        self.assertEqual(cfg.image, "ubuntu:22.04")
        self.assertEqual(cfg.container_name, "cctl-prj-myproj")
        self.assertEqual(cfg.repos_display, "-")

    def test_invalid_project_name_from_filename(self):
        path = write_toml(self.dir, "bad name", 'image = "ubuntu"\n')
        self.assertIsNone(ProjectConfig.from_toml(path))

    def test_missing_image(self):
        path = write_toml(self.dir, "p", 'ssh_key = "~/.ssh/id"\n')
        self.assertIsNone(ProjectConfig.from_toml(path))
        self.assertEqual(get_errors()[0]["cat"], "project.config")

    def test_invalid_image_ref(self):
        path = write_toml(self.dir, "p", 'image = "ubuntu; rm -rf /"\n')
        self.assertIsNone(ProjectConfig.from_toml(path))

    def test_legacy_single_repo_string(self):
        path = write_toml(self.dir, "p", 'image = "ubuntu"\nrepo = "https://github.com/u/repo.git"\n')
        cfg = ProjectConfig.from_toml(path)
        self.assertEqual(len(cfg.repos), 1)
        self.assertEqual(cfg.repos[0].name, "repo")
        self.assertEqual(cfg.repos_display, "https://github.com/u/repo.git")

    def test_repos_list_with_tables(self):
        content = (
            'image = "ubuntu"\n'
            "repos = [\n"
            '  {url = "https://github.com/u/a.git", branch = "main"},\n'
            '  {url = "https://github.com/u/b.git", dir = "bee"},\n'
            "]\n"
        )
        path = write_toml(self.dir, "p", content)
        cfg = ProjectConfig.from_toml(path)
        self.assertEqual([r.name for r in cfg.repos], ["a", "bee"])
        self.assertEqual(cfg.repos[0].branch, "main")
        self.assertEqual(cfg.repos_display, "2 repos")

    def test_repo_and_repos_both_rejected(self):
        content = 'image = "ubuntu"\nrepo = "https://x/a.git"\nrepos = ["https://x/b.git"]\n'
        path = write_toml(self.dir, "p", content)
        self.assertIsNone(ProjectConfig.from_toml(path))

    def test_invalid_repo_url(self):
        path = write_toml(self.dir, "p", 'image = "ubuntu"\nrepo = "https://x/ bad"\n')
        self.assertIsNone(ProjectConfig.from_toml(path))

    def test_duplicate_repo_dir_rejected(self):
        content = 'image = "ubuntu"\nrepos = ["https://x/repo.git", "https://y/repo.git"]\n'
        path = write_toml(self.dir, "p", content)
        self.assertIsNone(ProjectConfig.from_toml(path))

    def test_invalid_branch_rejected(self):
        content = 'image = "ubuntu"\nrepos = [{url = "https://x/a.git", branch = "bad branch"}]\n'
        path = write_toml(self.dir, "p", content)
        self.assertIsNone(ProjectConfig.from_toml(path))

    def test_env_files_parsed(self):
        content = 'image = "ubuntu"\nenv_files = [{src = "/home/dev/host.env", dest = "/home/dev/.env"}]\n'
        path = write_toml(self.dir, "p", content)
        cfg = ProjectConfig.from_toml(path)
        self.assertEqual(len(cfg.env_files), 1)
        self.assertEqual(cfg.env_files[0].dest, "/home/dev/.env")

    def test_image_digest_accepted(self):
        path = write_toml(self.dir, "p", f'image = "ubuntu@sha256:{"a" * 64}"\n')
        self.assertIsNotNone(ProjectConfig.from_toml(path))

    def test_shortcuts_parsed(self):
        content = 'image = "ubuntu"\n[shortcuts]\napp = "app"\nlib = "lib/src"\n'
        cfg = ProjectConfig.from_toml(write_toml(self.dir, "p", content))
        self.assertEqual(cfg.shortcuts, {"app": "app", "lib": "lib/src"})

    def test_shortcuts_default_empty(self):
        cfg = ProjectConfig.from_toml(write_toml(self.dir, "p", 'image = "ubuntu"\n'))
        self.assertEqual(cfg.shortcuts, {})

    def test_shortcuts_rejected(self):
        for body in (
            'shortcuts = "app"\n',  # not a table
            '[shortcuts]\n"bad name" = "app"\n',
            '[shortcuts]\napp = "/abs"\n',
            '[shortcuts]\napp = "../up"\n',
            '[shortcuts]\napp = ""\n',
            "[shortcuts]\napp = 1\n",
        ):
            path = write_toml(self.dir, "p", 'image = "ubuntu"\n' + body)
            self.assertIsNone(ProjectConfig.from_toml(path), body)


class ResolveEnterTargetTests(unittest.TestCase):
    def setUp(self):
        self.a = ProjectConfig(name="a", image="ubuntu", shortcuts={"x": "x", "both": "a-both"})
        self.b = ProjectConfig(name="b", image="ubuntu", shortcuts={"y": "y/sub", "both": "b-both", "a": "a"})
        self.projects = {"a": self.a, "b": self.b}

    def test_project_name_lands_in_default_dir(self):
        self.assertEqual(resolve_enter_target("a", self.projects), (self.a, None))

    def test_project_name_wins_over_shortcut(self):
        # b declares a shortcut "a", but "a" is a project
        self.assertEqual(resolve_enter_target("a", self.projects), (self.a, None))

    def test_unique_shortcut_lands_in_its_dir(self):
        self.assertEqual(resolve_enter_target("y", self.projects), (self.b, "/home/dev/workspace/y/sub"))

    def test_ambiguous_or_unknown_is_none(self):
        self.assertIsNone(resolve_enter_target("both", self.projects))
        self.assertIsNone(resolve_enter_target("nope", self.projects))


class ServiceConfigTests(unittest.TestCase):
    def setUp(self):
        reset_errors()
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_full_valid(self):
        content = (
            'image = "nginx:latest"\n'
            'restart = "always"\n'
            'ports = ["8080:80", "4000-5000:4000-5000/tcp"]\n'
            'volumes = ["/home/dev/data:/data:ro"]\n'
            'env = ["FOO=bar", "BAZ=qux"]\n'
            "[scripts]\n"
            'migrate = ["python", "manage.py", "migrate"]\n'
        )
        path = write_toml(self.dir, "web", content)
        cfg = ServiceConfig.from_toml(path)
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.container_name, "cctl-svc-web")
        self.assertEqual(cfg.restart, "always")
        self.assertEqual(cfg.ports, ["8080:80", "4000-5000:4000-5000/tcp"])
        self.assertEqual(cfg.volumes, ["/home/dev/data:/data:ro"])
        self.assertEqual(cfg.scripts["migrate"], ["python", "manage.py", "migrate"])
        # hardening + networks default off
        self.assertEqual(cfg.networks, [])
        self.assertFalse(cfg.read_only)
        self.assertEqual((cfg.tmpfs, cfg.cap_drop, cfg.cap_add, cfg.pids_limit), ([], [], [], 0))

    def test_networks_and_hardening_parsed(self):
        content = (
            'image = "nginx"\n'
            'networks = ["proxy", "cctl-net-x"]\n'
            "read_only = true\n"
            'tmpfs = ["/tmp", "/run:rw,noexec,size=16m,mode=1777"]\n'
            'cap_drop = ["ALL"]\n'
            'cap_add = ["NET_BIND_SERVICE", "CHOWN"]\n'
            "pids_limit = 256\n"
        )
        cfg = ServiceConfig.from_toml(write_toml(self.dir, "web", content))
        self.assertIsNotNone(cfg, get_errors())
        self.assertEqual(cfg.networks, ["proxy", "cctl-net-x"])
        self.assertTrue(cfg.read_only)
        self.assertEqual(cfg.tmpfs, ["/tmp", "/run:rw,noexec,size=16m,mode=1777"])
        self.assertEqual(cfg.cap_drop, ["ALL"])
        self.assertEqual(cfg.cap_add, ["NET_BIND_SERVICE", "CHOWN"])
        self.assertEqual(cfg.pids_limit, 256)

    def test_networks_and_hardening_rejected(self):
        for body in (
            'networks = ["bad net"]\n',
            'networks = ["a;b"]\n',
            'networks = [""]\n',
            "networks = [1]\n",
            'read_only = "yes"\n',
            'tmpfs = ["tmp"]\n',  # relative
            'tmpfs = ["/tmp/../etc"]\n',
            'tmpfs = ["/tmp:size=16m; rm -rf /"]\n',
            "tmpfs = [1]\n",
            'cap_drop = ["ALL CAPS"]\n',
            'cap_add = [""]\n',
            'pids_limit = "256"\n',
            "pids_limit = -1\n",
            "pids_limit = true\n",
            "pids_limit = 2.5\n",
        ):
            reset_errors()
            path = write_toml(self.dir, "web", 'image = "nginx"\n' + body)
            self.assertIsNone(ServiceConfig.from_toml(path), body)
            self.assertEqual(get_errors()[0]["cat"], "service.config", body)

    def test_env_is_parsed_into_config(self):
        path = write_toml(self.dir, "web", 'image = "nginx"\nenv = ["FOO=bar"]\n')
        cfg = ServiceConfig.from_toml(path)
        self.assertEqual(cfg.env, ["FOO=bar"])

    def test_invalid_restart_policy(self):
        path = write_toml(self.dir, "web", 'image = "nginx"\nrestart = "sometimes"\n')
        self.assertIsNone(ServiceConfig.from_toml(path))

    def test_invalid_port_spec(self):
        path = write_toml(self.dir, "web", 'image = "nginx"\nports = ["8080:80; echo"]\n')
        self.assertIsNone(ServiceConfig.from_toml(path))

    def test_port_spec_with_host_ip(self):
        path = write_toml(self.dir, "web", 'image = "nginx"\nports = ["127.0.0.1:8080:80"]\n')
        self.assertEqual(ServiceConfig.from_toml(path).ports, ["127.0.0.1:8080:80"])

    def test_invalid_volume_part_count(self):
        path = write_toml(self.dir, "web", 'image = "nginx"\nvolumes = ["/onlyhost"]\n')
        self.assertIsNone(ServiceConfig.from_toml(path))

    def test_invalid_volume_mode(self):
        path = write_toml(self.dir, "web", 'image = "nginx"\nvolumes = ["/home/dev/d:/d:xx"]\n')
        self.assertIsNone(ServiceConfig.from_toml(path))

    def test_env_without_equals_rejected(self):
        path = write_toml(self.dir, "web", 'image = "nginx"\nenv = ["NOEQUALS"]\n')
        self.assertIsNone(ServiceConfig.from_toml(path))

    def test_empty_script_argv_rejected(self):
        path = write_toml(self.dir, "web", 'image = "nginx"\n[scripts]\nbroken = []\n')
        self.assertIsNone(ServiceConfig.from_toml(path))


if __name__ == "__main__":
    unittest.main()


class EnvTablesTests(unittest.TestCase):
    """`[env]` (plain values) and `[secrets]` (env name -> secrets.toml key) in a project TOML."""

    def setUp(self):
        reset_errors()
        self.path = Path("/cfg/x.toml")

    def test_absent_is_empty(self):
        self.assertEqual(parse_env_table(None, self.path), {})
        self.assertEqual(parse_secrets_table(None, self.path), {})

    def test_plain_values(self):
        self.assertEqual(parse_env_table({"TZ": "UTC", "DEBUG": "1"}, self.path), {"TZ": "UTC", "DEBUG": "1"})
        self.assertEqual(parse_env_table({"EMPTY": ""}, self.path), {"EMPTY": ""})  # empty is a legal value

    def test_secret_names_map_to_keys(self):
        table = {"CLAUDE_CODE_OAUTH_TOKEN": "claude_token", "OAUTH_TOKEN": "claude_token"}
        self.assertEqual(parse_secrets_table(table, self.path), table)

    def test_invalid_names_rejected(self):
        for name in ("1ABC", "A-B", "", "A B", "A=B"):
            for parse in (parse_env_table, parse_secrets_table):
                reset_errors()
                self.assertIsNone(parse({name: "x"}, self.path), name)
                self.assertEqual(get_errors()[0]["cat"], "project.config")

    def test_not_a_table_rejected(self):
        self.assertIsNone(parse_env_table(["A=B"], self.path))
        self.assertIsNone(parse_secrets_table("TOK", self.path))

    def test_values_must_be_strings(self):
        for bad in (3, True, ["x"], {"value": "x"}):
            reset_errors()
            self.assertIsNone(parse_env_table({"A": bad}, self.path), bad)
            self.assertIsNone(parse_secrets_table({"A": bad}, self.path), bad)

    def test_nul_in_plain_value_rejected(self):
        self.assertIsNone(parse_env_table({"A": "x\x00y"}, self.path))

    def test_secret_key_must_be_a_name(self):
        """A value here is a key into secrets.toml, never the secret itself: only [A-Za-z_][A-Za-z0-9_]*."""
        for bad in ("", "sk-ant-oat01-...", "a key", "1x"):
            reset_errors()
            self.assertIsNone(parse_secrets_table({"TOK": bad}, self.path), bad)

    def test_project_toml_tables(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_toml(d, "p", 'image = "ubuntu"\n[env]\nDEBUG = "1"\n[secrets]\nTOK = "tok"\n')
            cfg = ProjectConfig.from_toml(path)
            self.assertEqual((cfg.env, cfg.secrets), ({"DEBUG": "1"}, {"TOK": "tok"}))
            bare = ProjectConfig.from_toml(write_toml(d, "q", 'image = "ubuntu"\n'))
            self.assertEqual((bare.env, bare.secrets), ({}, {}))
            bad = write_toml(d, "r", 'image = "ubuntu"\n[secrets]\nTOK = { value = "x" }\n')
            self.assertIsNone(ProjectConfig.from_toml(bad))

    def test_name_in_both_tables_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_toml(d, "p", 'image = "ubuntu"\n[env]\nTOK = "x"\n[secrets]\nTOK = "tok"\n')
            self.assertIsNone(ProjectConfig.from_toml(path))
        self.assertIn("both [env] and [secrets]", get_errors()[0]["msg"])


class ResolveEnvTests(unittest.TestCase):
    """[env] + [secrets] looked up in secrets.toml -> what `enter` injects; problems -> env.secret."""

    def setUp(self):
        reset_errors()
        self._tmp = tempfile.TemporaryDirectory()
        self.sec = Path(self._tmp.name) / "secrets.toml"
        self._patch = mock.patch.object(containerctl, "SECRETS_FILE", self.sec)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def write(self, secrets, mode=0o600):
        self.sec.write_text(secrets)
        self.sec.chmod(mode)

    def project(self, env=None, secrets=None):
        return ProjectConfig(name="p", image="i", env=env or {}, secrets=secrets or {})

    def cats(self):
        return [e["cat"] for e in get_errors()]

    def test_nothing_declared_nothing_injected(self):
        self.assertEqual(resolve_env(self.project()), {})
        self.assertEqual(self.cats(), [])

    def test_plain_and_secret_resolved(self):
        self.write('tok = "s3cret"\n')
        self.assertEqual(resolve_env(self.project({"TZ": "UTC"}, {"TOK": "tok"})), {"TZ": "UTC", "TOK": "s3cret"})
        self.assertEqual(self.cats(), [])

    def test_same_key_under_two_names(self):
        self.write('tok = "s3cret"\n')
        env = resolve_env(self.project(secrets={"CLAUDE_CODE_OAUTH_TOKEN": "tok", "OAUTH_TOKEN": "tok"}))
        self.assertEqual(env, {"CLAUDE_CODE_OAUTH_TOKEN": "s3cret", "OAUTH_TOKEN": "s3cret"})

    def test_missing_secret_is_logged_and_left_out(self):
        self.write("")
        self.assertEqual(resolve_env(self.project({"TZ": "UTC"}, {"TOK": "tok"})), {"TZ": "UTC"})
        self.assertEqual(self.cats(), ["env.secret"])
        self.assertIn("TOK: no 'tok' in", get_errors()[0]["msg"])

    def test_missing_file_is_just_missing_secrets(self):
        self.assertEqual(resolve_env(self.project(secrets={"TOK": "tok"})), {})
        self.assertEqual(self.cats(), ["env.secret"])

    def test_secret_values_must_be_non_empty_strings(self):
        for raw in ('tok = ""', "tok = 3", 'tok = { value = "s" }', 'tok = "a\\u0000b"'):
            reset_errors()
            self.write(raw + "\n")
            self.assertEqual(resolve_env(self.project(secrets={"TOK": "tok"})), {}, raw)
            self.assertEqual(self.cats(), ["env.secret", "env.secret"], raw)  # bad entry, then missing

    def test_invalid_key_is_skipped(self):
        self.write('"bad key" = "s"\ntok = "s3cret"\n')
        self.assertEqual(resolve_env(self.project(secrets={"TOK": "tok"})), {"TOK": "s3cret"})
        self.assertEqual(self.cats(), ["env.secret"])

    def test_world_readable_secrets_file_is_reported(self):
        self.write('tok = "s"\n', mode=0o644)
        self.assertEqual(resolve_env(self.project(secrets={"TOK": "tok"})), {"TOK": "s"})  # still works, but
        self.assertEqual(self.cats(), ["env.secret"])
        self.assertIn("mode 0644, readable by others: chmod 600", get_errors()[0]["msg"])

    def test_malformed_file_is_logged(self):
        self.write("tok = \n")
        self.assertEqual(resolve_env(self.project({"TZ": "UTC"}, {"TOK": "tok"})), {"TZ": "UTC"})
        self.assertEqual(self.cats(), ["env.secret", "env.secret"])  # load failure, then missing
        self.assertIn("Failed to load", get_errors()[0]["msg"])

    def test_undeclared_key_is_never_injected(self):
        self.write('tok = "s"\nother = "o"\n')
        self.assertEqual(resolve_env(self.project({"TZ": "UTC"}, {"TOK": "tok"})), {"TZ": "UTC", "TOK": "s"})

    def test_config_objects_are_not_mutated(self):
        self.write('tok = "s"\n')
        project = self.project({"TZ": "UTC"}, {"TOK": "tok"})
        resolve_env(project)
        self.assertEqual((project.env, project.secrets), ({"TZ": "UTC"}, {"TOK": "tok"}))

    def test_secrets_file_not_read_when_nothing_is_secret(self):
        self.write("garbage = [\n")  # would log if parsed
        self.assertEqual(resolve_env(self.project({"TZ": "UTC"})), {"TZ": "UTC"})
        self.assertEqual(self.cats(), [])

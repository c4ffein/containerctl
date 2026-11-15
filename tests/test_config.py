"""Tests for TOML project/service config parsing and validation."""

import dataclasses
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import containerctl
from containerctl import (
    DockerImageRow,
    ImageRecipeConfig,
    ProjectConfig,
    RepoSpec,
    ServiceConfig,
    find_image_recipe,
    get_errors,
    load_image_recipes,
    normalize_image_ref,
    parse_env_table,
    parse_secrets_table,
    resolve_enter_target,
    resolve_env,
)
from tests.helper import as_inspected, container, listings, rejected, reset_errors


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

    def test_user_and_home_default_to_dev_and_follow_each_other(self):
        for extra, expected in (
            ("", ("dev", "/home/dev", "/home/dev/workspace")),
            ('user = "ubuntu"\n', ("ubuntu", "/home/ubuntu", "/home/ubuntu/workspace")),
            ('user = "root"\n', ("root", "/root", "/root/workspace")),
            ('user = "app"\nhome = "/srv/app/"\n', ("app", "/srv/app", "/srv/app/workspace")),
        ):
            cfg = ProjectConfig.from_toml(write_toml(self.dir, "p", 'image = "ubuntu"\n' + extra))
            self.assertEqual((cfg.user, cfg.home, cfg.workspace), expected)

    def test_an_unsafe_user_or_home_is_refused(self):
        for extra in (
            'user = "-u"\n',
            'user = "a b"\n',
            "user = 1000\n",
            'home = "relative/dir"\n',
            'home = "/"\n',
            'home = "/home/../etc"\n',
            'home = "/home/$(id)"\n',
            'home = "/home/it\'s"\n',
        ):
            error = rejected(ProjectConfig.from_toml, write_toml(self.dir, "p", 'image = "ubuntu"\n' + extra))
            self.assertRegex(str(error), "Invalid (user|home)", extra)

    def test_minimal_valid(self):
        path = write_toml(self.dir, "myproj", 'image = "ubuntu:22.04"\n')
        cfg = ProjectConfig.from_toml(path)
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.name, "myproj")
        self.assertEqual(cfg.image, "ubuntu:22.04")
        self.assertEqual(cfg.container_name, "cctl-prj-myproj")
        self.assertEqual(cfg.repos_display, "-")

    def test_volumes_are_parsed_like_a_services(self):
        content = 'image = "ubuntu"\nvolumes = ["/srv/drop:/home/dev/dropzone", "cache:/var/cache/x:ro"]\n'
        cfg = ProjectConfig.from_toml(write_toml(self.dir, "p", content))
        self.assertEqual(cfg.volumes, ["/srv/drop:/home/dev/dropzone", "cache:/var/cache/x:ro"])
        home = write_toml(self.dir, "p", 'image = "ubuntu"\nvolumes = ["~/drop:/drop"]\n')
        self.assertEqual(ProjectConfig.from_toml(home).volumes, [f"{Path.home()}/drop:/drop"])
        self.assertEqual(ProjectConfig.from_toml(write_toml(self.dir, "p", 'image = "ubuntu"\n')).volumes, [])

    def test_an_unsafe_volume_is_refused(self):
        for volumes in (
            '"/srv/drop:/drop"',  # not a list
            '["/onlyhost"]',
            '["/a:/b:ro:z"]',
            '["/a:/b:xx"]',
            '["/a:relative"]',
            '["/a:/b/../c"]',
            '["/a/../b:/c"]',
            '[":/c"]',
            '["--privileged:/c"]',
            '["/a:/$(id)"]',
            "[1]",
        ):
            path = write_toml(self.dir, "p", f'image = "ubuntu"\nvolumes = {volumes}\n')
            error = rejected(ProjectConfig.from_toml, path)
            self.assertEqual(error.cat, "project.config", volumes)
            self.assertIn("volume", str(error), volumes)

    def test_invalid_project_name_from_filename(self):
        path = write_toml(self.dir, "bad name", 'image = "ubuntu"\n')
        rejected(ProjectConfig.from_toml, path)

    def test_missing_image(self):
        path = write_toml(self.dir, "p", 'ssh_key = "~/.ssh/id"\n')
        self.assertEqual(rejected(ProjectConfig.from_toml, path).cat, "project.config")

    def test_invalid_image_ref(self):
        path = write_toml(self.dir, "p", 'image = "ubuntu; rm -rf /"\n')
        rejected(ProjectConfig.from_toml, path)

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
        rejected(ProjectConfig.from_toml, path)

    def test_invalid_repo_url(self):
        path = write_toml(self.dir, "p", 'image = "ubuntu"\nrepo = "https://x/ bad"\n')
        rejected(ProjectConfig.from_toml, path)

    def test_duplicate_repo_dir_rejected(self):
        content = 'image = "ubuntu"\nrepos = ["https://x/repo.git", "https://y/repo.git"]\n'
        path = write_toml(self.dir, "p", content)
        rejected(ProjectConfig.from_toml, path)

    def test_invalid_branch_rejected(self):
        content = 'image = "ubuntu"\nrepos = [{url = "https://x/a.git", branch = "bad branch"}]\n'
        path = write_toml(self.dir, "p", content)
        rejected(ProjectConfig.from_toml, path)

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
            rejected(ProjectConfig.from_toml, path)


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
        self.b.home = "/home/ubuntu"
        self.assertEqual(resolve_enter_target("y", self.projects), (self.b, "/home/ubuntu/workspace/y/sub"))

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
        self.assertEqual(cfg.command, [])
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
            self.assertEqual(rejected(ServiceConfig.from_toml, path).cat, "service.config")

    def test_env_and_secrets_are_tables_as_for_a_project(self):
        body = 'image = "nginx"\n[env]\nFOO = "bar"\nEMPTY = ""\n[secrets]\nAPI_TOKEN = "web_token"\n'
        cfg = ServiceConfig.from_toml(write_toml(self.dir, "web", body))
        self.assertEqual(cfg.env, {"FOO": "bar", "EMPTY": ""})
        self.assertEqual(cfg.secrets, {"API_TOKEN": "web_token"})
        self.assertEqual((ServiceConfig(name="w", image="i").env, ServiceConfig(name="w", image="i").secrets), ({}, {}))

    def test_the_env_list_of_before_is_refused_saying_what_to_write(self):
        path = write_toml(self.dir, "web", 'image = "nginx"\nenv = ["FOO=bar"]\n')
        error = rejected(ServiceConfig.from_toml, path)
        self.assertEqual(error.cat, "service.config")
        self.assertIn("'env' is a table now", str(error))
        self.assertIn("[env]", str(error))

    def test_env_and_secrets_rejected(self):
        for body in (
            'env = "FOO=bar"\n',
            '[env]\n"BAD-NAME" = "x"\n',
            '[env]\n"1X" = "x"\n',
            "[env]\nFOO = 1\n",
            '[secrets]\nTOKEN = "not a key"\n',
            '[secrets]\n"A B" = "key"\n',
            '[env]\nTOKEN = "x"\n[secrets]\nTOKEN = "key"\n',  # which one would win?
        ):
            reset_errors()
            path = write_toml(self.dir, "web", 'image = "nginx"\n' + body)
            self.assertEqual(rejected(ServiceConfig.from_toml, path).cat, "service.config", body)

    def test_limits_and_logs_parsed(self):
        cfg = ServiceConfig.from_toml(write_toml(self.dir, "web", 'image = "nginx"\n'))
        self.assertEqual((cfg.memory, cfg.cpus, cfg.user, cfg.no_new_privileges), ("", "", "", False))
        self.assertEqual((cfg.log_max_size, cfg.log_max_file), ("10m", 3))  # rotation is on unless said otherwise
        body = (
            'image = "nginx"\nmemory = "512m"\ncpus = 1.5\nuser = "101:101"\nno_new_privileges = true\n'
            'log_max_size = "1g"\nlog_max_file = 5\n'
        )
        cfg = ServiceConfig.from_toml(write_toml(self.dir, "web", body))
        self.assertEqual((cfg.memory, cfg.cpus, cfg.user, cfg.no_new_privileges), ("512m", "1.5", "101:101", True))
        self.assertEqual((cfg.log_max_size, cfg.log_max_file), ("1g", 5))
        for written, read in (("2", "2"), ('"0.25"', "0.25"), ("0.5", "0.5"), ("2.0", "2")):
            cfg = ServiceConfig.from_toml(write_toml(self.dir, "web", f'image = "nginx"\ncpus = {written}\n'))
            self.assertEqual(cfg.cpus, read)
        cfg = ServiceConfig.from_toml(write_toml(self.dir, "web", 'image = "nginx"\nlog_max_size = ""\n'))
        self.assertEqual(cfg.log_max_size, "")

    def test_healthcheck_parsed(self):
        cfg = ServiceConfig.from_toml(write_toml(self.dir, "web", 'image = "nginx"\n'))
        self.assertEqual((cfg.health_cmd, cfg.health_interval, cfg.health_retries), ("", "", 0))
        body = (
            'image = "nginx"\nhealth_cmd = "curl -fsS http://localhost/ || exit 1"\nhealth_interval = "30s"\n'
            'health_timeout = "500ms"\nhealth_start_period = "1m"\nhealth_retries = 5\n'
        )
        cfg = ServiceConfig.from_toml(write_toml(self.dir, "web", body))
        self.assertEqual(cfg.health_cmd, "curl -fsS http://localhost/ || exit 1")
        self.assertEqual((cfg.health_interval, cfg.health_timeout, cfg.health_start_period), ("30s", "500ms", "1m"))
        self.assertEqual(cfg.health_retries, 5)
        durations = {"30s": 30 * 10**9, "500ms": 5 * 10**8, "2m": 120 * 10**9, "1h": 3600 * 10**9}
        self.assertEqual({text: containerctl.duration_ns(text) for text in durations}, durations)
        for text in ("", "30", "s", "0s", "1m30s", "-5s", "5 s", "5d", "1.5s", "٣s"):
            self.assertIsNone(containerctl.duration_ns(text), text)

    def test_healthcheck_rejected(self):
        for body in (
            'health_cmd = ["curl", "localhost"]\n',
            'health_cmd = " true"\n',
            'health_cmd = "true"\nhealth_interval = 30\n',
            'health_cmd = "true"\nhealth_interval = "30"\n',
            'health_cmd = "true"\nhealth_timeout = "1m30s"\n',
            'health_cmd = "true"\nhealth_start_period = "soon"\n',
            'health_cmd = "true"\nhealth_retries = -1\n',
            'health_cmd = "true"\nhealth_retries = "3"\n',
            'health_interval = "30s"\n',  # of what?
            "health_retries = 3\n",
        ):
            reset_errors()
            path = write_toml(self.dir, "web", 'image = "nginx"\n' + body)
            self.assertEqual(rejected(ServiceConfig.from_toml, path).cat, "service.config", body)

    def test_limits_and_logs_rejected(self):
        for body in (
            "memory = 512\n",
            'memory = "512mb"\n',
            'memory = "0"\n',
            'memory = "-1g"\n',
            'memory = "1g --privileged"\n',
            "cpus = 0\n",
            "cpus = -1\n",
            "cpus = true\n",
            'cpus = "two"\n',
            'cpus = "1.5.2"\n',
            'user = "-u"\n',
            'user = "a:b:c"\n',
            'user = "web user"\n',
            "user = 101\n",
            'no_new_privileges = "yes"\n',
            "log_max_size = 10\n",
            'log_max_size = "ten"\n',
            "log_max_file = 0\n",
            'log_max_file = "3"\n',
            "log_max_file = true\n",
        ):
            reset_errors()
            path = write_toml(self.dir, "web", 'image = "nginx"\n' + body)
            self.assertEqual(rejected(ServiceConfig.from_toml, path).cat, "service.config", body)

    def test_invalid_restart_policy(self):
        path = write_toml(self.dir, "web", 'image = "nginx"\nrestart = "sometimes"\n')
        rejected(ServiceConfig.from_toml, path)

    def test_invalid_port_spec(self):
        path = write_toml(self.dir, "web", 'image = "nginx"\nports = ["8080:80; echo"]\n')
        rejected(ServiceConfig.from_toml, path)

    def test_port_spec_with_host_ip(self):
        path = write_toml(self.dir, "web", 'image = "nginx"\nports = ["127.0.0.1:8080:80"]\n')
        self.assertEqual(ServiceConfig.from_toml(path).ports, ["127.0.0.1:8080:80"])

    def test_invalid_volume_part_count(self):
        path = write_toml(self.dir, "web", 'image = "nginx"\nvolumes = ["/onlyhost"]\n')
        rejected(ServiceConfig.from_toml, path)

    def test_invalid_volume_mode(self):
        path = write_toml(self.dir, "web", 'image = "nginx"\nvolumes = ["/home/dev/d:/d:xx"]\n')
        rejected(ServiceConfig.from_toml, path)

    def test_invalid_volume_container_path(self):
        path = write_toml(self.dir, "web", 'image = "nginx"\nvolumes = ["/home/dev/d:d"]\n')
        self.assertEqual(rejected(ServiceConfig.from_toml, path).cat, "service.config")

    def test_env_without_equals_rejected(self):
        path = write_toml(self.dir, "web", 'image = "nginx"\nenv = ["NOEQUALS"]\n')
        rejected(ServiceConfig.from_toml, path)

    def test_empty_script_argv_rejected(self):
        path = write_toml(self.dir, "web", 'image = "nginx"\n[scripts]\nbroken = []\n')
        rejected(ServiceConfig.from_toml, path)


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
                self.assertEqual(rejected(parse, {name: "x"}, self.path).cat, "project.config")

    def test_not_a_table_rejected(self):
        rejected(parse_env_table, ["A=B"], self.path)
        rejected(parse_secrets_table, "TOK", self.path)

    def test_values_must_be_strings(self):
        for bad in (3, True, ["x"], {"value": "x"}):
            reset_errors()
            rejected(parse_env_table, {"A": bad}, self.path)
            rejected(parse_secrets_table, {"A": bad}, self.path)

    def test_nul_in_plain_value_rejected(self):
        rejected(parse_env_table, {"A": "x\x00y"}, self.path)

    def test_secret_key_must_be_a_name(self):
        """A value here is a key into secrets.toml, never the secret itself: only [A-Za-z_][A-Za-z0-9_]*."""
        for bad in ("", "sk-ant-oat01-...", "a key", "1x"):
            reset_errors()
            rejected(parse_secrets_table, {"TOK": bad}, self.path)

    def test_project_toml_tables(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_toml(d, "p", 'image = "ubuntu"\n[env]\nDEBUG = "1"\n[secrets]\nTOK = "tok"\n')
            cfg = ProjectConfig.from_toml(path)
            self.assertEqual((cfg.env, cfg.secrets), ({"DEBUG": "1"}, {"TOK": "tok"}))
            bare = ProjectConfig.from_toml(write_toml(d, "q", 'image = "ubuntu"\n'))
            self.assertEqual((bare.env, bare.secrets), ({}, {}))
            bad = write_toml(d, "r", 'image = "ubuntu"\n[secrets]\nTOK = { value = "x" }\n')
            rejected(ProjectConfig.from_toml, bad)

    def test_name_in_both_tables_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_toml(d, "p", 'image = "ubuntu"\n[env]\nTOK = "x"\n[secrets]\nTOK = "tok"\n')
            self.assertIn("both [env] and [secrets]", str(rejected(ProjectConfig.from_toml, path)))


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
        return [e.cat for e in get_errors()]

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
        self.assertIn("TOK: no 'tok' in", get_errors()[0].msg)

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
        self.assertIn("mode 0644, readable by others: chmod 600", get_errors()[0].msg)

    def test_malformed_file_is_logged(self):
        self.write("tok = \n")
        self.assertEqual(resolve_env(self.project({"TZ": "UTC"}, {"TOK": "tok"})), {"TZ": "UTC"})
        self.assertEqual(self.cats(), ["env.secret", "env.secret"])  # load failure, then missing
        self.assertIn("Failed to load", get_errors()[0].msg)

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


class ImageConfigTests(unittest.TestCase):
    """`<name>.toml` + `<name>.Dockerfile` [+ `<name>/`] -> ImageRecipeConfig; the recipe hash covers all of it."""

    def setUp(self):
        reset_errors()
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def recipe(self, name="hermes", toml="", dockerfile="FROM debian:12\n"):
        path = write_toml(self.dir, name, toml)
        (self.dir / f"{name}.Dockerfile").write_text(dockerfile)
        return path

    def test_defaults(self):
        cfg = ImageRecipeConfig.from_toml(self.recipe())
        self.assertEqual((cfg.name, cfg.tag, cfg.upstream, cfg.has_context), ("hermes", "hermes:latest", "", False))
        self.assertEqual(cfg.dockerfile, self.dir / "hermes.Dockerfile")
        self.assertEqual(get_errors(), [])

    def test_tag_and_upstream(self):
        toml = 'tag = "registry.local/hermes:1.2"\nupstream = "https://raw.githubusercontent.com/u/r/main/Dockerfile"\n'
        cfg = ImageRecipeConfig.from_toml(self.recipe(toml=toml))
        self.assertEqual(cfg.tag, "registry.local/hermes:1.2")
        self.assertEqual(cfg.upstream, "https://raw.githubusercontent.com/u/r/main/Dockerfile")
        self.assertEqual(ImageRecipeConfig.from_toml(self.recipe(toml='tag = "hermes"\n')).tag, "hermes:latest")

    def test_invalid_rejected(self):
        for toml in (
            "tag = 3\n",
            'tag = "a b"\n',
            'tag = "x@sha256:abc"\n',
            'upstream = "http://x/Dockerfile"\n',
            'upstream = "https://x/Dockerfile?x=1"\n',
            "tag = [\n",
        ):
            reset_errors()
            self.assertEqual(rejected(ImageRecipeConfig.from_toml, self.recipe(toml=toml)).cat, "image.config")

    def test_missing_dockerfile_rejected(self):
        path = write_toml(self.dir, "hermes", "")
        self.assertIn("Missing hermes.Dockerfile", str(rejected(ImageRecipeConfig.from_toml, path)))

    def test_invalid_name_rejected(self):
        rejected(ImageRecipeConfig.from_toml, self.recipe(name="a b"))

    def test_load_images_and_find_recipe(self):
        self.recipe("a")
        self.recipe("b", toml='tag = "other:2"\n')
        with mock.patch.object(containerctl, "IMAGES_DIR", self.dir):
            images = load_image_recipes()
        self.assertEqual(sorted(images), ["a", "b"])
        self.assertIs(find_image_recipe("a", images), images["a"])
        self.assertIs(find_image_recipe("a:latest", images), images["a"])
        self.assertIs(find_image_recipe("other:2", images), images["b"])
        self.assertIsNone(find_image_recipe("other", images))
        self.assertIsNone(find_image_recipe("a@sha256:abc", images))
        with mock.patch.object(containerctl, "IMAGES_DIR", self.dir / "nope"):
            self.assertEqual(load_image_recipes(), {})

    def test_recipe_hash_covers_dockerfile_and_context(self):
        cfg = ImageRecipeConfig.from_toml(self.recipe())
        h0 = cfg.recipe_hash()
        self.assertEqual(len(h0), 64)
        self.assertEqual(cfg.recipe_hash(), h0)  # deterministic
        (self.dir / "hermes.Dockerfile").write_text("FROM debian:13\n")
        h1 = cfg.recipe_hash()
        self.assertNotEqual(h1, h0)
        ctx = self.dir / "hermes"
        ctx.mkdir()
        self.assertTrue(cfg.has_context)
        h2 = cfg.recipe_hash()
        self.assertEqual(h2, h1)  # an empty context adds nothing
        (ctx / "sub").mkdir()
        (ctx / "sub" / "conf").write_text("x")
        h3 = cfg.recipe_hash()
        self.assertNotEqual(h3, h2)
        (ctx / "sub" / "conf").write_text("y")
        self.assertNotEqual(cfg.recipe_hash(), h3)

    def test_recipe_hash_ignores_git_history(self):
        cfg = ImageRecipeConfig.from_toml(self.recipe())
        ctx = self.dir / "hermes"
        (ctx / ".git" / "objects").mkdir(parents=True)
        (ctx / "src").mkdir()
        (ctx / "src" / "a").write_text("a")
        h = cfg.recipe_hash()
        (ctx / ".git" / "objects" / "pack").write_text("blob")
        (ctx / ".git" / "HEAD").write_text("ref")
        self.assertEqual(cfg.recipe_hash(), h)
        (ctx / "src" / "a").write_text("b")
        self.assertNotEqual(cfg.recipe_hash(), h)

    def test_recipe_hash_unreadable(self):
        cfg = ImageRecipeConfig.from_toml(self.recipe())
        (self.dir / "hermes.Dockerfile").unlink()
        self.assertIsNone(cfg.recipe_hash())
        self.assertEqual(get_errors()[0].cat, "image.recipe")


class NormalizeImageRefTests(unittest.TestCase):
    def test_cases(self):
        for ref, want in (
            ("hermes", "hermes:latest"),
            ("hermes:1", "hermes:1"),
            ("registry.local:5000/hermes", "registry.local:5000/hermes:latest"),
            ("registry.local:5000/hermes:1", "registry.local:5000/hermes:1"),
            ("x@sha256:abc", "x@sha256:abc"),
        ):
            self.assertEqual(normalize_image_ref(ref), want, ref)


class ServiceCommandTests(unittest.TestCase):
    def setUp(self):
        reset_errors()
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_command_parsed(self):
        cfg = ServiceConfig.from_toml(write_toml(self.dir, "h", 'image = "x"\ncommand = ["gateway", "run"]\n'))
        self.assertEqual(cfg.command, ["gateway", "run"])

    def test_command_rejected(self):
        for bad in ('command = "gateway run"', 'command = ["a", ""]', "command = [1]"):
            reset_errors()
            self.assertIn(
                "command", str(rejected(ServiceConfig.from_toml, write_toml(self.dir, "h", f'image = "x"\n{bad}\n')))
            )


class IssueTests(unittest.TestCase):
    """Issues: detected from loaded config and the docker listings, never logged, gone once fixed."""

    def setUp(self):
        reset_errors()

    def test_shortcut_defined_twice_and_shadowed_by_a_project(self):
        projects = {
            "lib": ProjectConfig(name="lib", image="i"),
            "a": ProjectConfig(name="a", image="i", shortcuts={"lib": "lib", "docs": "d"}),
            "b": ProjectConfig(name="b", image="i", shortcuts={"docs": "d", "only": "o"}),
        }
        issues = containerctl.shortcut_issues(projects)
        self.assertEqual([(i.level, i.cat) for i in issues], [(containerctl.WARNING, "project.shortcut")] * 2)
        self.assertEqual(issues[0].msg, "Shortcut 'docs' is defined by several projects: a, b")
        self.assertEqual(issues[1].msg, "Shortcut 'lib' (a) is shadowed by project 'lib'")
        self.assertEqual(issues[1].hint, "rename it in a.toml")
        self.assertEqual(get_errors(), [])  # an issue, not an error
        self.assertEqual(containerctl.shortcut_issues({"a": ProjectConfig(name="a", image="i")}), [])

    def test_secrets_file_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            sec = Path(tmp) / "secrets.toml"
            with mock.patch.object(containerctl, "SECRETS_FILE", sec):
                self.assertIsNone(containerctl.secrets_mode_issue())  # absent: nothing to protect
                sec.write_text('tok = "s"\n')
                sec.chmod(0o600)
                self.assertIsNone(containerctl.secrets_mode_issue())
                sec.chmod(0o644)
                issue = containerctl.secrets_mode_issue()
        self.assertEqual(issue.cat, "env.secret")
        self.assertEqual(issue.msg, f"{sec} is mode 0644, readable by others")
        self.assertEqual(issue.hint, f"chmod 600 {sec}")
        self.assertEqual(get_errors(), [])

    def test_a_broken_file_is_an_error_issue_and_hides_no_other_file(self):
        with tempfile.TemporaryDirectory() as d:
            write_toml(d, "good", 'image = "ubuntu"\n')
            write_toml(d, "bad", "image = \n")
            write_toml(d, "worse", 'image = "ubuntu; rm -rf /"\n')
            with mock.patch.object(containerctl, "PROJECTS_DIR", Path(d)):
                projects, problems = containerctl.read_projects()
                self.assertEqual(get_errors(), [])  # handed back, not logged
                self.assertEqual(sorted(containerctl.load_projects()), ["good"])  # the one-shot flavour logs them
        self.assertEqual(sorted(projects), ["good"])
        self.assertEqual(
            [(i.level, i.cat, i.hint) for i in problems],
            [
                (containerctl.ERROR, "project.config", "edit bad.toml"),
                (containerctl.ERROR, "project.config", "edit worse.toml"),
            ],
        )
        self.assertIn("Failed to load", problems[0].msg)
        self.assertEqual([(e.cat, e.msg) for e in get_errors()], [(i.cat, i.msg) for i in problems])

    def test_a_missing_config_dir_is_no_config_and_no_issue(self):
        for read in (containerctl.read_projects, containerctl.read_services, containerctl.read_image_recipes):
            with (
                mock.patch.object(containerctl, "PROJECTS_DIR", Path("/nonexistent/p")),
                mock.patch.object(containerctl, "SERVICES_DIR", Path("/nonexistent/s")),
                mock.patch.object(containerctl, "IMAGES_DIR", Path("/nonexistent/i")),
            ):
                self.assertEqual(read(), ({}, []))

    def test_failed_listings_are_one_issue_per_reason(self):
        down = "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?"
        with mock.patch.object(containerctl, "SECRETS_FILE", Path("/nonexistent/secrets.toml")):
            all_down = containerctl.detect_issues(
                {}, {}, {}, listings(failed=dict.fromkeys(containerctl.LISTINGS, down))
            )
            partial = containerctl.detect_issues({}, {}, {}, listings(failed={"volumes": "plugin exploded"}))
        self.assertEqual(
            all_down,
            [
                containerctl.Issue(
                    containerctl.ERROR,
                    "docker",
                    "Docker daemon is not running (containers, images, volumes, networks)",
                    "start docker",
                    "\n".join(f"docker {key}: {down}" for key in containerctl.LISTINGS),  # what docker said, raw
                )
            ],
        )
        self.assertEqual(
            [(i.level, i.msg) for i in partial], [(containerctl.ERROR, "Unknown docker failure (volumes)")]
        )
        self.assertEqual(get_errors(), [])

    def test_a_failed_listing_is_not_read_as_an_empty_one(self):
        """No image rows because `docker images` failed: no verdict on versions. Same for orphans and `docker ps`."""
        image = containerctl.ImageRecipeConfig(name="hermes", dockerfile=Path("h.Dockerfile"), context=Path("h"))
        with mock.patch.object(containerctl, "SECRETS_FILE", Path("/nonexistent/secrets.toml")):
            issues = containerctl.detect_issues(
                {}, {}, {"hermes": image}, listings(failed={"images": "boom", "containers": "boom"})
            )
        self.assertEqual([i.cat for i in issues], ["docker"])

    def test_containers_without_a_config_are_orphans(self):
        with tempfile.TemporaryDirectory() as d:
            write_toml(d, "broken", "image = \n")  # there but unusable: still owns its container
            with (
                mock.patch.object(containerctl, "PROJECTS_DIR", Path(d)),
                mock.patch.object(containerctl, "SERVICES_DIR", Path(d) / "none"),
                mock.patch.object(containerctl, "SECRETS_FILE", Path("/nonexistent/secrets.toml")),
            ):
                names = ["cctl-prj-live", "cctl-prj-broken", "cctl-prj-gone", "cctl-svc-web", "cctl-svc-old", "redis"]
                rows = [container(name) for name in names]
                issues = containerctl.detect_issues(
                    {"live": ProjectConfig(name="live", image="i")},
                    {"web": ServiceConfig(name="web", image="i")},
                    {},
                    listings(containers=rows),
                )
        self.assertEqual(
            [(i.level, i.cat, i.msg, i.hint) for i in issues],
            [
                (
                    containerctl.WARNING,
                    "project.orphan",
                    "Container cctl-prj-gone has no project config",
                    "restore gone.toml, or docker rm -f cctl-prj-gone",
                ),
                (
                    containerctl.WARNING,
                    "service.orphan",
                    "Container cctl-svc-old has no service config",
                    "restore old.toml, or docker rm -f cctl-svc-old",
                ),
            ],
        )

    def test_an_unhealthy_service_is_an_issue(self):
        services = {name: ServiceConfig(name=name, image="i") for name in ("ok", "sick")}
        sick = container("cctl-svc-sick")
        sick.Status = "Up 3 hours (unhealthy)"
        rows = [container("cctl-svc-ok"), sick, container("other")]
        rows[0].Status, rows[2].Status = "Up 3 hours (healthy)", "Up 1 hour (unhealthy)"  # not ours to speak of
        (issue,) = containerctl.service_issues(services, rows)
        self.assertEqual((issue.level, issue.cat), (containerctl.WARNING, "service.health"))
        self.assertEqual(issue.msg, "Service sick is unhealthy: its healthcheck fails")
        self.assertEqual(issue.hint, "containerctl logs cctl-svc-sick")


SECRETS = {"web_token": "s3cr3t-value"}
IMAGE_ID = "sha256:" + "1" * 64
FULL = ServiceConfig(
    name="web",
    image="nginx:1.27",
    restart="always",
    ports=["127.0.0.1:8080:80", "4000-4002:5000-5002/udp", "9000"],
    volumes=["/srv/www:/usr/share/nginx/html:ro", "cache:/var/cache"],
    env={"TZ": "UTC", "EMPTY": ""},
    secrets={"API_TOKEN": "web_token"},
    command=["nginx", "-g", "daemon off;"],
    networks=["proxy"],
    read_only=True,
    tmpfs=["/tmp", "/run:size=16m"],
    cap_drop=["ALL"],
    cap_add=["NET_BIND_SERVICE"],
    pids_limit=256,
    memory="512m",
    cpus="1.5",
    user="101:101",
    no_new_privileges=True,
    health_cmd="curl -fsS http://localhost/ || exit 1",
    health_interval="30s",
    health_timeout="500ms",
    health_start_period="1m",
    health_retries=5,
    log_max_size="10m",
    log_max_file=3,
)


class ServiceDifferencesTests(unittest.TestCase):
    """A service's container, as an inspect has it, against its TOML: service_differences. The inspects here are
    stand-ins (tests.helper.as_inspected): the real docker is asked in tests/test_real_docker.py."""

    def setUp(self):
        reset_errors()

    def whats(self, service, info, secrets=SECRETS, tag_id=IMAGE_ID):
        return [what for what, _ in containerctl.service_differences(service, info, secrets, tag_id)]

    def test_a_container_as_its_toml_would_make_it_has_none(self):
        for service in (FULL, ServiceConfig(name="bare", image="nginx"), ServiceConfig("x", "i", log_max_size="")):
            self.assertEqual(
                containerctl.service_differences(service, as_inspected(service, SECRETS), SECRETS, IMAGE_ID), []
            )

    def test_what_is_not_the_containers_to_keep_is_not_compared(self):
        info = as_inspected(FULL, SECRETS)
        for same in (
            dataclasses.replace(FULL, networks=[]),  # `up` connects new ones
            dataclasses.replace(FULL, scripts={"reload": ["nginx", "-s", "reload"]}),  # for `exec`
            dataclasses.replace(FULL, user=""),  # left to the image
            dataclasses.replace(FULL, log_max_size=""),  # left to the daemon
            dataclasses.replace(FULL, cap_add=["cap_net_bind_service"]),  # one capability, written another way
            dataclasses.replace(FULL, ports=[*FULL.ports[:2], "9000/tcp"]),
        ):
            self.assertEqual(self.whats(same, info), [])

    def test_every_edit_of_the_toml_is_named(self):
        info = as_inspected(FULL, SECRETS)
        edits = {
            "restart": {"restart": "unless-stopped"},
            "ports": {"ports": ["127.0.0.1:8081:80", *FULL.ports[1:]]},
            "volumes": {"volumes": FULL.volumes[:1]},
            "read_only": {"read_only": False},
            "tmpfs": {"tmpfs": ["/tmp", "/run:size=32m"]},
            "cap_drop": {"cap_drop": []},
            "cap_add": {"cap_add": ["NET_BIND_SERVICE", "CHOWN"]},
            "pids_limit": {"pids_limit": 0},
            "memory": {"memory": "1g"},
            "cpus": {"cpus": "2"},
            "user": {"user": "0"},
            "no_new_privileges": {"no_new_privileges": False},
            "logs": {"log_max_file": 5},
            "health_cmd": {"health_cmd": "true"},
            "health_interval": {"health_interval": "10s"},
            "health_timeout": {"health_timeout": ""},
            "health_start_period": {"health_start_period": "2m"},
            "health_retries": {"health_retries": 3},
            "command": {"command": ["nginx"]},
            "env TZ": {"env": {"TZ": "Europe/Paris", "EMPTY": ""}},
            "env NEW": {"env": {**FULL.env, "NEW": "1"}},
            "env EMPTY": {"env": {"TZ": "UTC"}},  # no longer in the TOML: its label says it was given
            "env API_TOKEN": {"secrets": {}},
            "env OTHER": {"secrets": {**FULL.secrets, "OTHER": "web_token"}},
        }
        for what, change in edits.items():
            self.assertEqual(self.whats(dataclasses.replace(FULL, **change), info), [what], what)
        self.assertEqual(self.whats(dataclasses.replace(FULL, log_max_size="20m"), info), ["logs"])
        # DockerServices.options writes a field, service_differences reads it back: two lists, kept by hand. A field
        # added to ServiceConfig must be in the edits above (so: compared), or here, with why it is not
        elsewhere = {
            "name": "what the container is found by",
            "image": "compared by name, then by ID: its own test",
            "networks": "`up` connects new ones",
            "scripts": "for `exec`, never in the container",
            "log_max_size": "just above",
        }
        edited = {field for change in edits.values() for field in change}
        self.assertEqual(edited | set(elsewhere), {field.name for field in dataclasses.fields(ServiceConfig)})
        (said,) = containerctl.service_differences(dataclasses.replace(FULL, memory="1g"), info, SECRETS, IMAGE_ID)
        self.assertEqual(said, ("memory", "memory: 536870912 on the container, 1073741824 in its TOML"))
        self.assertEqual(self.whats(dataclasses.replace(FULL, command=[]), info), ["command"])  # by its label too
        no_health = dataclasses.replace(
            FULL, health_cmd="", health_interval="", health_timeout="", health_start_period="", health_retries=0
        )
        self.assertEqual(self.whats(no_health, info), ["health_cmd"])  # by its label as well
        self.assertEqual(self.whats(no_health, as_inspected(no_health, SECRETS)), [])  # the image's own: not ours

    def test_a_docker_update_by_hand_is_named(self):
        """The TOML has not moved: only an inspect can say"""
        info = as_inspected(FULL, SECRETS)
        info.HostConfig.update(Memory=1024**3, NanoCpus=2 * 10**9, PidsLimit=-1)
        info.HostConfig["RestartPolicy"] = {"Name": "no", "MaximumRetryCount": 0}
        self.assertEqual(self.whats(FULL, info), ["restart", "pids_limit", "memory", "cpus"])
        info = as_inspected(ServiceConfig(name="bare", image="nginx"))
        info.HostConfig["RestartPolicy"] = {"Name": "", "MaximumRetryCount": 0}  # how an old docker says "no"
        self.assertEqual(self.whats(ServiceConfig(name="bare", image="nginx", restart="no"), info), [])

    def test_the_image_is_compared_by_name_then_by_what_the_name_is_now(self):
        info = as_inspected(FULL, SECRETS, image_id=IMAGE_ID)
        ((what, said),) = containerctl.service_differences(
            dataclasses.replace(FULL, image="nginx:1.28"), info, SECRETS, None
        )
        self.assertEqual((what, said), ("image", "image: nginx:1.27 on the container, nginx:1.28 in its TOML"))
        moved = "sha256:" + "2" * 64
        ((what, said),) = containerctl.service_differences(FULL, info, SECRETS, moved)
        self.assertEqual(what, "image")
        self.assertEqual(said, "image: an older nginx:1.27, the tag has moved since the container was made")
        for same in (IMAGE_ID, "1" * 12, "sha256:" + "1" * 12, None):  # short as `docker images` has it; not known
            self.assertEqual(self.whats(FULL, info, tag_id=same), [], same)
        bare = ServiceConfig(name="bare", image="nginx")  # `nginx` and `nginx:latest` are one ref
        self.assertEqual(self.whats(dataclasses.replace(bare, image="nginx:latest"), as_inspected(bare)), [])

    def test_a_rotated_secret_is_named_and_never_shown(self):
        info = as_inspected(FULL, SECRETS)
        ((what, said),) = containerctl.service_differences(FULL, info, {"web_token": "the-new-one"}, IMAGE_ID)
        self.assertEqual(what, "env API_TOKEN")
        self.assertEqual(said, "env API_TOKEN: not the value secrets.toml has under 'web_token' now")
        self.assertEqual(self.whats(FULL, info, secrets={}), [])  # no such key any more: nothing to compare with
        every = containerctl.service_differences(dataclasses.replace(FULL, secrets={}), info, {"web_token": "x"}, None)
        self.assertNotIn("s3cr3t-value", str(every))

    def test_a_container_made_before_the_labels_is_compared_as_far_as_it_can_be(self):
        info = as_inspected(FULL, SECRETS)
        del info.Config["Labels"][containerctl.LABEL_ENV], info.Config["Labels"][containerctl.LABEL_COMMAND]
        self.assertEqual(self.whats(FULL, info), [])
        # what was given and is no longer in the TOML can't be told from the image's own: not said
        self.assertEqual(self.whats(dataclasses.replace(FULL, env={}, command=[]), info), [])
        self.assertEqual(self.whats(dataclasses.replace(FULL, env={"TZ": "CET", "EMPTY": ""}), info), ["env TZ"])

    def test_sizes_cpus_and_ports_as_docker_keeps_them(self):
        sizes = {"1": 1, "10b": 10, "2k": 2048, "512m": 512 * 1024**2, "2G": 2 * 1024**3}
        self.assertEqual({text: containerctl.size_bytes(text) for text in sizes}, sizes)
        cpus = {"2": 2 * 10**9, "1.5": 1_500_000_000, "0.1": 100_000_000, "0.25": 250_000_000}
        self.assertEqual({text: containerctl.nano_cpus(text) for text in cpus}, cpus)
        self.assertEqual(containerctl.port_bindings("80"), {("80/tcp", "", "")})
        self.assertEqual(containerctl.port_bindings("8080:80/udp"), {("80/udp", "", "8080")})
        self.assertEqual(containerctl.port_bindings("127.0.0.1:8080:80"), {("80/tcp", "127.0.0.1", "8080")})
        self.assertEqual(containerctl.port_bindings("1-2:3-4"), {("3/tcp", "", "1"), ("4/tcp", "", "2")})
        self.assertIsNone(containerctl.port_bindings("8000-8010:80"))  # docker picks one of them
        ranged = ServiceConfig(name="r", image="i", ports=["8000-8010:80"])
        info = as_inspected(ServiceConfig(name="r", image="i", ports=["8005:80"]))
        self.assertEqual(self.whats(ranged, info, tag_id=None), [])  # so its ports are not compared


class ServiceDriftTests(unittest.TestCase):
    """service_drift: which containers are inspected and when, and what becomes of the differences"""

    def setUp(self):
        reset_errors()
        self.services = {name: ServiceConfig(name=name, image="nginx") for name in ("a", "b", "down")}
        self.rows = [
            container("cctl-svc-a", id_="ida"),
            container("cctl-svc-b", id_="idb"),
            container("other", id_="x"),
        ]
        self.answers = {"ida": as_inspected(self.services["a"], id_="ida"), "idb": as_inspected(self.services["b"])}
        self.inspect = mock.Mock(side_effect=lambda cls, ref, timeout=None: self.answer(ref))
        for patch in (
            mock.patch.object(containerctl.Docker, "inspect", self.inspect),
            mock.patch.object(containerctl, "SECRETS_FILE", Path("/nonexistent/secrets.toml")),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def answer(self, ref):
        if isinstance(self.answers.get(ref), Exception):
            raise self.answers[ref]
        return self.answers.get(ref)

    def drift(self, cache=None, **kwargs):
        return containerctl.service_drift(self.services, listings(containers=self.rows), cache, **kwargs)

    def asked(self):
        return [call.args[1] for call in self.inspect.call_args_list]

    def test_only_the_containers_of_services_are_inspected(self):
        self.assertEqual(self.drift(timeout=1), [])
        self.assertEqual(self.asked(), ["ida", "idb"])
        self.assertEqual(self.inspect.call_args.kwargs, {"timeout": 1})

    def test_a_difference_is_one_issue_per_service_with_every_sentence_in_its_detail(self):
        self.services["a"] = dataclasses.replace(self.services["a"], memory="1g", restart="always")
        (issue,) = self.drift()
        self.assertEqual((issue.level, issue.cat), (containerctl.WARNING, "service.drift"))
        self.assertEqual(issue.msg, "Service a is not what its TOML says: restart, memory")
        self.assertEqual(issue.hint, "containerctl service recreate a")
        self.assertEqual(len(issue.detail.splitlines()), 2)

    def test_the_tag_is_looked_up_in_the_images_listing(self):
        rows = [containerctl.DockerImageRow("2" * 12, "nginx", "latest", "1MB")]
        every = listings(containers=self.rows, images=rows)
        self.assertEqual(len(containerctl.service_drift(self.services, every)), 2)  # both are of an older nginx
        every = listings(failed={"images": "boom"}, containers=self.rows)
        self.assertEqual(containerctl.service_drift(self.services, every), [])  # not known: not said

    def test_with_a_cache_a_container_is_asked_about_again_only_when_its_time_is_up(self):
        cache = {}
        with mock.patch.object(containerctl.time, "monotonic", return_value=1000.0):
            self.drift(cache)
            self.drift(cache)
        self.assertEqual(self.asked(), ["ida", "idb"])
        with mock.patch.object(containerctl.time, "monotonic", return_value=1000.0 + containerctl.INSPECT_TTL - 1):
            self.drift(cache)
        self.assertEqual(self.asked(), ["ida", "idb"])
        self.answers["ida"].HostConfig["Memory"] = 1024  # a `docker update` by hand, meanwhile
        with mock.patch.object(containerctl.time, "monotonic", return_value=1000.0 + containerctl.INSPECT_TTL):
            (issue,) = self.drift(cache)
        self.assertEqual(self.asked(), ["ida", "idb", "ida", "idb"])
        self.assertEqual(issue.msg, "Service a is not what its TOML says: memory")

    def test_a_recreated_container_is_new_and_the_gone_one_forgotten(self):
        cache = {}
        self.drift(cache)
        self.rows[0] = container("cctl-svc-a", id_="idnew")
        self.answers["idnew"] = as_inspected(self.services["a"], id_="idnew")
        self.drift(cache)
        self.assertEqual(self.asked(), ["ida", "idb", "idnew"])
        self.assertEqual(sorted(cache), ["idb", "idnew"])

    def test_the_first_inspect_docker_does_not_answer_is_the_last_asked(self):
        cache = {}
        self.drift(cache)
        silent = containerctl.DockerError(containerctl.Failure("Docker did not answer in time", "is it hung?", "raw"))
        self.answers = {"ida": silent, "idb": silent}
        self.services["b"] = dataclasses.replace(self.services["b"], memory="1g")
        with mock.patch.object(containerctl.time, "monotonic", return_value=time.monotonic() + 3600):
            issues = self.drift(cache)
        self.assertEqual(self.asked(), ["ida", "idb", "ida"])  # one timeout, not one per service
        self.assertEqual(
            [(i.cat, i.msg) for i in issues],
            [
                ("service.drift", "Service containers could not be inspected: Docker did not answer in time"),
                ("service.drift", "Service b is not what its TOML says: memory"),  # what is known is still compared
            ],
        )
        self.assertEqual(get_errors(), [])  # state: an issue, never the log

    def test_no_listing_no_inspect_and_a_container_gone_meanwhile_is_nothing(self):
        failed = listings(failed={"containers": "boom"})
        self.assertEqual(containerctl.service_drift(self.services, failed), [])
        self.assertEqual(self.asked(), [])
        self.answers["ida"] = None  # removed between the listing and the inspect
        self.assertEqual(self.drift(), [])

    def test_secrets_are_read_quietly_and_only_when_a_running_service_has_some(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "secrets.toml"
            path.write_text('web_token = "the-new-one"\n"bad key" = "x"\n')  # world-readable, one bad entry
            with mock.patch.object(containerctl, "SECRETS_FILE", path):
                with mock.patch.object(containerctl, "read_secrets", wraps=containerctl.read_secrets) as read:
                    self.drift()
                    read.assert_not_called()
                    self.services["a"] = dataclasses.replace(self.services["a"], secrets={"API_TOKEN": "web_token"})
                    self.answers["ida"] = as_inspected(self.services["a"], SECRETS, id_="ida")
                    (issue,) = self.drift()
        self.assertEqual(issue.msg, "Service a is not what its TOML says: env API_TOKEN")
        self.assertNotIn("the-new-one", issue.detail + issue.msg)
        self.assertEqual(get_errors(), [])  # every fetch reads it: its problems are load_secrets's to log


class ImageVersionIssueTests(unittest.TestCase):
    def setUp(self):
        reset_errors()

    def test_image_versions_from_docker_images_rows(self):
        image = containerctl.ImageRecipeConfig(name="hermes", dockerfile=Path("h.Dockerfile"), context=Path("h"))
        rows = [DockerImageRow(f"id{n}", "hermes", f"cctl-local-{n:06d}", "1MB") for n in range(4)]
        rows.append(DockerImageRow("id3", "hermes", "latest", "1MB"))  # the recipe tag points at version 3
        rows.append(DockerImageRow("x", "other", "cctl-local-000009", "1MB"))  # another repository
        self.assertEqual(containerctl.local_versions(image, rows), ([0, 1, 2, 3], 3, "id3"))
        with mock.patch.object(containerctl, "SECRETS_FILE", Path("/nonexistent/secrets.toml")):
            issues = containerctl.detect_issues({}, {}, {"hermes": image}, listings(images=rows))
            fine = containerctl.detect_issues({}, {}, {"hermes": image}, listings(images=rows[:3]))  # three is fine
        self.assertEqual(fine, [])
        self.assertEqual(
            [(i.level, i.cat, i.msg) for i in issues],
            [(containerctl.WARNING, "image.versions", "hermes has 4 local versions")],
        )
        self.assertEqual(
            issues[0].hint, "docker rmi hermes:cctl-local-000000 hermes:cctl-local-000001 hermes:cctl-local-000002"
        )

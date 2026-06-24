"""Tests for TOML project/service config parsing and validation."""

import tempfile
import unittest
from pathlib import Path

from containerctl import ProjectConfig, RepoSpec, ServiceConfig, get_errors
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

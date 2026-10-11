"""Tests for the input-validation helpers that guard every shell-out."""

import unittest

from containerctl import (
    has_path_traversal,
    is_valid_id,
    is_valid_image_ref,
    is_valid_path,
    is_valid_port_spec,
    is_valid_rel_dir,
    is_valid_repo_url,
    is_valid_tmpfs_spec,
)


class IsValidIdTests(unittest.TestCase):
    def test_accepts_alnum_dash_underscore(self):
        self.assertTrue(is_valid_id("abc123"))
        self.assertTrue(is_valid_id("my-container_1"))

    def test_empty_string_is_valid(self):
        # all() over an empty iterable is True; documented behaviour.
        self.assertTrue(is_valid_id(""))

    def test_rejects_shell_metacharacters(self):
        for bad in ["a;b", "a b", "a$b", "a|b", "a/b", "a:b", "a.b", "a\nb"]:
            self.assertFalse(is_valid_id(bad), bad)


class IsValidImageRefTests(unittest.TestCase):
    def test_allows_repo_tag_registry(self):
        self.assertTrue(is_valid_image_ref("ubuntu:22.04"))
        self.assertTrue(is_valid_image_ref("ghcr.io/owner/img:latest"))

    def test_rejects_spaces_and_semicolons(self):
        self.assertFalse(is_valid_image_ref("ubuntu; rm -rf /"))
        self.assertFalse(is_valid_image_ref("img latest"))

    def test_allows_digest_pin(self):
        self.assertTrue(is_valid_image_ref("ubuntu@sha256:" + "a" * 64))
        self.assertTrue(is_valid_image_ref("ghcr.io/owner/img:1.2@sha256:" + "0" * 64))


class IsValidPortSpecTests(unittest.TestCase):
    def test_accepts_docker_publish_forms(self):
        for good in ["80", "8080:80", "8080:80/tcp", "53:53/udp", "4000-5000:4000-5000", "127.0.0.1:8080:80"]:
            self.assertTrue(is_valid_port_spec(good), good)

    def test_rejects_garbage(self):
        bad_specs = [
            "",
            "8080:80; echo",
            "8080:80/sctp",
            "8080:80/",
            "localhost:8080:80",
            "::1:8080:80",
            "999.0.0.1:80:80",
            "a:b",
            "8080:",
            "80-",
            "70000:80",
            "1:2:3:4",
            "٨٠:80",  # non-ASCII digits pass str.isdigit
        ]
        for bad in bad_specs:
            self.assertFalse(is_valid_port_spec(bad), bad)


class IsValidRelDirTests(unittest.TestCase):
    def test_accepts_relative_dirs(self):
        for good in ["app", "lib/src", "a.b-c_d"]:
            self.assertTrue(is_valid_rel_dir(good), good)

    def test_rejects_absolute_traversal_and_metachars(self):
        for bad in ["", "/abs", "../up", "a/../b", "a b", "a;b", "$HOME"]:
            self.assertFalse(is_valid_rel_dir(bad), bad)


class PathTraversalTests(unittest.TestCase):
    def test_detects_dotdot_component(self):
        self.assertTrue(has_path_traversal("../etc/passwd"))
        self.assertTrue(has_path_traversal("/home/dev/../root"))

    def test_dotdot_only_as_full_component(self):
        # "a..b" is not a traversal; ".." must be its own path segment.
        self.assertFalse(has_path_traversal("a..b"))
        self.assertFalse(has_path_traversal("/home/dev/file..bak"))

    def test_valid_path_rejects_traversal(self):
        self.assertFalse(is_valid_path("/home/../etc"))

    def test_valid_path_accepts_normal_paths(self):
        self.assertTrue(is_valid_path("/home/dev/.ssh/id_ed25519"))
        self.assertTrue(is_valid_path("~/config/app.env"))


class IsValidRepoUrlTests(unittest.TestCase):
    def test_accepts_https_and_scp_style(self):
        self.assertTrue(is_valid_repo_url("https://github.com/u/repo.git"))
        self.assertTrue(is_valid_repo_url("git@github.com:u/repo.git"))

    def test_rejects_traversal_and_spaces(self):
        self.assertFalse(is_valid_repo_url("https://x/../y"))
        self.assertFalse(is_valid_repo_url("https://x/ y"))


class IsValidTmpfsSpecTests(unittest.TestCase):
    def test_accepts_path_with_optional_options(self):
        for good in ["/tmp", "/run:size=16m", "/var/cache/app:rw,noexec,nosuid,size=64m,mode=1777,uid=1000"]:
            self.assertTrue(is_valid_tmpfs_spec(good), good)

    def test_rejects_relative_traversal_and_metachars(self):
        for bad in ["", "tmp", "/tmp/../x", "/tmp:size=1m;id", "/tmp:$(x)", "/tmp:a b", "/tmp:size=1m:extra"]:
            self.assertFalse(is_valid_tmpfs_spec(bad), bad)


if __name__ == "__main__":
    unittest.main()

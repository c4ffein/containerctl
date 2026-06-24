"""Tests for the input-validation helpers that guard every shell-out."""

import unittest

from containerctl import (
    has_path_traversal,
    is_valid_id,
    is_valid_image_ref,
    is_valid_path,
    is_valid_repo_url,
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


if __name__ == "__main__":
    unittest.main()

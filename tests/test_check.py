"""Tests for the workspace cleanliness check (host + local share this core)."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import containerctl
from containerctl import (
    RepoStatus,
    check_workspace,
    inspect_repo,
    print_report,
    read_manifest,
)


class FakeGit:
    """A stand-in git runner. Each repo is described by a small spec dict:

    {"is_git": bool, "porcelain": str, "detached": bool,
     "upstream": bool, "ahead": int, "stashes": int}
    """

    def __init__(self, repos):
        self.repos = repos

    def __call__(self, name, args):
        spec = self.repos.get(name, {})
        if args[:2] == ["rev-parse", "--git-dir"]:
            return (0, ".git\n") if spec.get("is_git", True) else (128, "")
        if args[:2] == ["status", "--porcelain"]:
            return (0, spec.get("porcelain", ""))
        if args[:1] == ["symbolic-ref"]:
            return (1, "") if spec.get("detached", False) else (0, "refs/heads/main\n")
        if args[:2] == ["rev-list", "--count"]:
            if not spec.get("upstream", True):
                return (128, "")
            return (0, f"{spec.get('ahead', 0)}\n")
        if args[:2] == ["stash", "list"]:
            return (0, "\n".join(f"stash@{{{i}}}" for i in range(spec.get("stashes", 0))))
        raise AssertionError(f"unexpected git args: {args}")


class RepoStatusTests(unittest.TestCase):
    def test_clean_repo_has_no_problems(self):
        s = RepoStatus(name="a")
        self.assertTrue(s.clean)
        self.assertEqual(s.problems(), [])

    def test_not_git_is_dirty(self):
        s = RepoStatus(name="a", is_git=False)
        self.assertFalse(s.clean)
        self.assertEqual(s.problems(), ["not a git repository"])

    def test_no_upstream_masks_ahead_message(self):
        s = RepoStatus(name="a", has_upstream=False, ahead=3)
        self.assertFalse(s.clean)
        self.assertIn("no upstream branch", s.problems())
        self.assertNotIn("3 unpushed commits", s.problems())

    def test_singular_plural_wording(self):
        self.assertIn("1 unpushed commit", RepoStatus(name="a", ahead=1).problems())
        self.assertIn("2 unpushed commits", RepoStatus(name="a", ahead=2).problems())
        self.assertIn("1 stash", RepoStatus(name="a", stashes=1).problems())
        self.assertIn("2 stashes", RepoStatus(name="a", stashes=2).problems())


class InspectRepoTests(unittest.TestCase):
    def test_clean(self):
        git = FakeGit({"a": {}})
        s = inspect_repo("a", git)
        self.assertTrue(s.clean)

    def test_uncommitted_and_untracked(self):
        git = FakeGit({"a": {"porcelain": " M file.py\n?? new.py\n"}})
        s = inspect_repo("a", git)
        self.assertTrue(s.uncommitted)
        self.assertTrue(s.untracked)
        self.assertFalse(s.clean)

    def test_only_untracked(self):
        git = FakeGit({"a": {"porcelain": "?? new.py\n"}})
        s = inspect_repo("a", git)
        self.assertFalse(s.uncommitted)
        self.assertTrue(s.untracked)

    def test_unpushed_commits(self):
        git = FakeGit({"a": {"ahead": 4}})
        s = inspect_repo("a", git)
        self.assertEqual(s.ahead, 4)
        self.assertTrue(s.has_upstream)
        self.assertFalse(s.clean)

    def test_no_upstream(self):
        git = FakeGit({"a": {"upstream": False}})
        s = inspect_repo("a", git)
        self.assertFalse(s.has_upstream)
        self.assertFalse(s.clean)

    def test_detached_head(self):
        git = FakeGit({"a": {"detached": True}})
        s = inspect_repo("a", git)
        self.assertTrue(s.detached)
        self.assertFalse(s.clean)

    def test_stashes(self):
        git = FakeGit({"a": {"stashes": 2}})
        s = inspect_repo("a", git)
        self.assertEqual(s.stashes, 2)

    def test_not_a_git_repo_short_circuits(self):
        git = FakeGit({"a": {"is_git": False}})
        s = inspect_repo("a", git)
        self.assertFalse(s.is_git)
        self.assertFalse(s.clean)


class CheckWorkspaceTests(unittest.TestCase):
    def test_extra_and_missing_detection(self):
        git = FakeGit({"a": {}, "b": {}, "forgotten": {}})
        report = check_workspace(
            expected=["a", "b", "gone"],
            list_dirs=lambda: ["a", "b", "forgotten"],
            run_git=git,
        )
        self.assertEqual(report.extra, ["forgotten"])
        self.assertEqual(report.missing, ["gone"])
        self.assertFalse(report.clean)

    def test_all_clean_and_matching(self):
        git = FakeGit({"a": {}, "b": {}})
        report = check_workspace(["a", "b"], lambda: ["a", "b"], git)
        self.assertTrue(report.clean)

    def test_expected_none_skips_extra_missing(self):
        git = FakeGit({"a": {}})
        report = check_workspace(None, lambda: ["a"], git)
        self.assertEqual(report.extra, [])
        self.assertEqual(report.missing, [])
        self.assertTrue(report.clean)

    def test_dirty_repo_makes_report_dirty(self):
        git = FakeGit({"a": {"porcelain": " M x\n"}})
        report = check_workspace(["a"], lambda: ["a"], git)
        self.assertFalse(report.clean)


class PrintReportTests(unittest.TestCase):
    def _render(self, report):
        buf = io.StringIO()
        with redirect_stdout(buf):
            clean = print_report("workspace", report)
        return clean, buf.getvalue()

    def test_clean_output(self):
        git = FakeGit({"a": {}})
        report = check_workspace(["a"], lambda: ["a"], git)
        clean, out = self._render(report)
        self.assertTrue(clean)
        self.assertIn("✓ a", out)
        self.assertIn("clean", out)
        self.assertNotIn("DIRTY", out)

    def test_dirty_output_lists_reasons(self):
        git = FakeGit({"a": {"porcelain": " M x\n", "ahead": 1}, "extra": {}})
        report = check_workspace(["a"], lambda: ["a", "extra"], git)
        clean, out = self._render(report)
        self.assertFalse(clean)
        self.assertIn("uncommitted changes", out)
        self.assertIn("not in config", out)
        self.assertIn("DIRTY", out)

    def test_missing_dir_reported(self):
        git = FakeGit({})
        report = check_workspace(["gone"], lambda: [], git)
        clean, out = self._render(report)
        self.assertFalse(clean)
        self.assertIn("gone — configured but absent", out)


class ManifestTests(unittest.TestCase):
    def test_read_manifest_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / ".config" / "containerctl"
            cfg.mkdir(parents=True)
            (cfg / "expected.json").write_text(json.dumps(["a", "b"]))
            with mock.patch.object(containerctl.Path, "home", return_value=Path(d)):
                self.assertEqual(read_manifest(), ["a", "b"])

    def test_read_manifest_absent_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(containerctl.Path, "home", return_value=Path(d)):
                self.assertIsNone(read_manifest())

    def test_read_manifest_rejects_non_list(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / ".config" / "containerctl"
            cfg.mkdir(parents=True)
            (cfg / "expected.json").write_text(json.dumps({"not": "a list"}))
            with mock.patch.object(containerctl.Path, "home", return_value=Path(d)):
                self.assertIsNone(read_manifest())


if __name__ == "__main__":
    unittest.main()

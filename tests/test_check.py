"""Tests for the workspace cleanliness check (host + local share this core)."""

import hashlib
import io
import json
import os
import shutil
import subprocess
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

    {"is_git": bool, "porcelain": str, "detached": bool, "branches": {name: track}, "stashes": int, "tagged": int,
     "worktrees": [path], "fail": [git subcommand], "submodules": [path], "head": int, "mirror": bool}

    `submodules` lists the paths `git submodule foreach` prints; each is then a repo of its own in the dict, under
    "<repo>/<path>", where `head` is how many commits its HEAD has that no remote ref does.

    `branches` maps each local branch to what `%(upstream:track)` says of it ("" level, "[ahead 2]", "[gone]"),
    or to None when it has no upstream; one level `main` by default. `fail` names the calls that exit non-zero.
    `mirror` is whether the repo fetches the remote's tags into refs/upstream-tags (it does, by default).
    """

    def __init__(self, repos):
        self.repos = repos

    def __call__(self, name, args):
        spec = self.repos.get(name, {})
        if args[0] in spec.get("fail", []):
            return (1, "")
        if args[:2] == ["rev-parse", "--git-dir"]:
            return (0, ".git\n") if spec.get("is_git", True) else (128, "")
        if args[:2] == ["status", "--porcelain"]:
            self.status_args = args
            return (0, spec.get("porcelain", ""))
        if args[:1] == ["symbolic-ref"]:
            return (1, "") if spec.get("detached", False) else (0, "refs/heads/main\n")
        if args[:1] == ["for-each-ref"]:
            self.format = args[1]
            lines = [
                f"{branch}\t\t" if track is None else f"{branch}\trefs/remotes/origin/{branch}\t{track}"
                for branch, track in spec.get("branches", {"main": ""}).items()
            ]
            return (0, "\n".join(lines) + "\n")
        if args[:2] == ["worktree", "list"]:
            paths = [f"/home/dev/workspace/{name}", *spec.get("worktrees", [])]
            return (0, "".join(f"worktree {p}\nHEAD abc\nbranch refs/heads/x\n\n" for p in paths))
        if args[:1] == ["submodule"]:
            self.submodule_args = args
            return (0, "".join(f"{path}\n" for path in spec.get("submodules", [])))
        if args[:3] == ["rev-list", "--count", "HEAD"]:
            self.head_args = args
            return (0, f"{spec.get('head', 0)}\n")
        if args[:1] == ["rev-list"]:
            self.rev_list = args
            return (0, f"{spec.get('tagged', 0)}\n")
        if args[:2] == ["config", "--get-regexp"]:
            if spec.get("mirror", True):
                return (0, "remote.origin.fetch +refs/tags/*:refs/upstream-tags/*\n")
            return (0, "remote.origin.fetch +refs/heads/*:refs/remotes/origin/*\n")
        if args[:2] == ["stash", "list"]:
            return (0, "\n".join(f"stash@{{{i}}}" for i in range(spec.get("stashes", 0))))
        raise AssertionError(f"unexpected git args: {args}")


def dirs(*names):
    """Workspace entries that are all directories"""
    return dict.fromkeys(names, True)


class RepoStatusTests(unittest.TestCase):
    def test_clean_repo_has_no_problems(self):
        s = RepoStatus(name="a")
        self.assertTrue(s.clean)
        self.assertEqual(s.problems(), [])

    def test_not_git_is_dirty(self):
        s = RepoStatus(name="a", is_git=False)
        self.assertFalse(s.clean)
        self.assertEqual(s.problems(), ["not a git repository"])

    def test_every_kind_of_finding_is_a_problem(self):
        s = RepoStatus(name="a", unpushed=["branch x: no upstream"], worktrees=["/tmp/wt"], unknown=["status"])
        self.assertFalse(s.clean)
        self.assertEqual(
            s.problems(),
            ["branch x: no upstream", "linked worktree /tmp/wt", "`git status` failed, could not check"],
        )

    def test_singular_plural_wording(self):
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
        git = FakeGit({"a": {"branches": {"main": "[ahead 4]"}}})
        s = inspect_repo("a", git)
        self.assertEqual(s.unpushed, ["branch main: 4 unpushed commits"])
        self.assertFalse(s.clean)

    def test_no_upstream(self):
        git = FakeGit({"a": {"branches": {"main": None}}})
        s = inspect_repo("a", git)
        self.assertEqual(s.unpushed, ["branch main: no upstream"])
        self.assertFalse(s.clean)

    def test_every_local_branch_counts_not_just_the_one_checked_out(self):
        branches = {"main": "", "wip": "[ahead 1, behind 3]", "idea": None, "old": "[gone]", "late": "[behind 2]"}
        s = inspect_repo("a", FakeGit({"a": {"branches": branches}}))
        self.assertEqual(
            s.unpushed,
            ["branch wip: 1 unpushed commit", "branch idea: no upstream", "branch old: upstream is gone"],
        )

    def test_an_upstream_that_is_another_local_branch_proves_nothing(self):
        listing = "main\trefs/remotes/origin/main\t\nwip\trefs/heads/main\t\n"
        self.assertEqual(containerctl.unpushed_branches(listing), ["branch wip: upstream main is local, not a remote"])

    def test_the_branch_listing_is_asked_tab_separated(self):
        git = FakeGit({"a": {}})
        inspect_repo("a", git)
        self.assertEqual(git.format, "--format=%(refname:short)%09%(upstream)%09%(upstream:track)")

    def test_commits_only_a_tag_holds_are_not_clean(self):
        s = inspect_repo("a", FakeGit({"a": {"tagged": 2}}))
        self.assertEqual(s.problems(), ["2 unpushed commits held by tags only"])
        self.assertFalse(s.clean)

    def test_untracked_files_are_asked_for_whatever_the_git_config_says(self):
        git = FakeGit({"a": {}})
        inspect_repo("a", git)
        self.assertEqual(git.status_args, ["status", "--porcelain", "--untracked-files=normal"])

    def test_linked_worktrees_are_named(self):
        s = inspect_repo("a", FakeGit({"a": {"worktrees": ["/home/dev/wt-1", "/tmp/wt-2"]}}))
        self.assertEqual(s.worktrees, ["/home/dev/wt-1", "/tmp/wt-2"])
        self.assertIn("linked worktree /tmp/wt-2", s.problems())
        self.assertFalse(s.clean)

    def test_a_git_call_that_fails_is_not_read_as_nothing_to_report(self):
        for call in ("status", "for-each-ref", "rev-list", "stash", "worktree", "submodule"):
            s = inspect_repo("a", FakeGit({"a": {"fail": [call]}}))
            self.assertEqual(s.unknown, [call])
            self.assertFalse(s.clean, call)

    def test_tags_are_counted_off_the_remote_branches_and_off_the_mirrored_upstream_tags(self):
        git = FakeGit({"a": {"tagged": 2}})
        s = inspect_repo("a", git)
        self.assertEqual(s.tagged, 2)
        expected = ["rev-list", "--count", "--tags", "--not", "--remotes", "--glob=refs/upstream-tags/*"]
        self.assertEqual(git.rev_list, expected)

    def test_a_tag_count_that_is_not_a_number_is_not_read_as_zero(self):
        s = inspect_repo("a", FakeGit({"a": {"tagged": "warning: something"}}))
        self.assertEqual((s.tagged, s.unknown), (0, ["rev-list"]))
        self.assertFalse(s.clean)

    def test_each_submodule_is_inspected_as_the_repo_it_is(self):
        git = FakeGit(
            {
                "a": {"submodules": ["libs/x", "libs/x/deep"]},
                "a/libs/x": {"detached": True, "head": 2, "stashes": 1},
                "a/libs/x/deep": {"detached": True, "branches": {"wip": None}},
            }
        )
        s = inspect_repo("a", git)
        self.assertEqual(git.submodule_args, containerctl.LIST_SUBMODULES)
        head = ["rev-list", "--count", "HEAD", "--not", "--remotes", "--glob=refs/upstream-tags/*"]
        self.assertEqual(git.head_args, head)
        self.assertEqual([sub.name for sub in s.submodules], ["libs/x", "libs/x/deep"])
        self.assertEqual(
            s.problems(),
            [
                "submodule libs/x: 1 stash",
                "submodule libs/x: 2 unpushed commits on HEAD",
                "submodule libs/x/deep: branch wip: no upstream",
            ],
        )
        self.assertFalse(s.clean)

    def test_a_submodule_on_a_pushed_commit_is_clean_detached_as_git_leaves_it(self):
        git = FakeGit({"a": {"submodules": ["sub"]}, "a/sub": {"detached": True}})
        s = inspect_repo("a", git)
        self.assertFalse(s.submodules[0].detached)  # not asked: it is a submodule's normal state
        self.assertTrue(s.clean)

    def test_a_submodule_whose_git_calls_fail_is_not_clean(self):
        git = FakeGit({"a": {"submodules": ["sub"]}, "a/sub": {"fail": ["rev-list"]}})
        self.assertEqual(
            inspect_repo("a", git).problems(),
            ["submodule sub: `git rev-list` failed, could not check"] * 2,  # HEAD's count, and the tags'
        )

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


@unittest.skipUnless(shutil.which("git"), "needs git")
class RealGitTests(unittest.TestCase):
    """inspect_repo against git itself: FakeGit only says what we think git says"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self.repo = self.home / "workspace" / "app"
        env = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null", "LC_ALL": "C"}
        ident = {
            f"GIT_{who}_{what}": value
            for who in ("AUTHOR", "COMMITTER")
            for what, value in (("NAME", "t"), ("EMAIL", "t@t"))
        }
        self._env = mock.patch.dict(os.environ, {**env, **ident})
        self._env.start()
        self._home = mock.patch.object(containerctl.Path, "home", return_value=self.home)
        self._home.start()
        self.git("init", "-q", "--bare", "-b", "main", str(self.home / "remote.git"), cwd=self.home)
        self.git("clone", "-q", str(self.home / "remote.git"), str(self.repo), cwd=self.home)
        self.git("commit", "-q", "--allow-empty", "-m", "one")
        self.git("push", "-q", "-u", "origin", "HEAD:main")

    def tearDown(self):
        self._home.stop()
        self._env.stop()
        self._tmp.cleanup()

    def git(self, *args, cwd=None):
        subprocess.run(["git", *args], cwd=cwd or self.repo, check=True, capture_output=True)

    def problems(self):
        return inspect_repo("app", containerctl._local_git).problems()

    def test_a_pushed_clone_is_clean(self):
        self.assertEqual(self.problems(), [])

    def test_every_branch_is_read_as_git_really_prints_it(self):
        self.git("branch", "idea")
        self.git("branch", "--track", "local", "main")
        self.git("checkout", "-q", "-b", "wip", "--track", "origin/main")
        self.git("commit", "-q", "--allow-empty", "-m", "two")
        self.git("checkout", "-q", "main")
        self.assertEqual(
            self.problems(),
            [
                "branch idea: no upstream",
                "branch local: upstream main is local, not a remote",
                "branch wip: 1 unpushed commit",
            ],
        )

    def add_submodule(self):
        """A pushed superproject with a submodule `libs/sub`, itself with a submodule `deep`; all clean"""
        allow = ("-c", "protocol.file.allow=always")
        for name in ("deep", "sub"):
            self.git("init", "-q", "-b", "main", str(self.home / name), cwd=self.home)
            self.git("commit", "-q", "--allow-empty", "-m", "one", cwd=self.home / name)
        self.git(*allow, "submodule", "-q", "add", str(self.home / "deep"), "deep", cwd=self.home / "sub")
        self.git("commit", "-q", "-m", "deep", cwd=self.home / "sub")
        self.git(*allow, "submodule", "-q", "add", str(self.home / "sub"), "libs/sub")
        self.git(*allow, "submodule", "-q", "update", "--init", "--recursive")
        self.git("commit", "-q", "-m", "sub")
        self.git("push", "-q")

    def test_submodules_on_pushed_commits_are_clean(self):
        self.add_submodule()
        self.assertEqual(self.problems(), [])

    def test_a_submodule_commit_its_remote_does_not_have_is_seen_once_the_superproject_is_pushed(self):
        self.add_submodule()
        sub = self.repo / "libs" / "sub"
        self.git("checkout", "-q", "--detach", cwd=sub)  # as `submodule update` leaves it
        self.git("commit", "-q", "--allow-empty", "-m", "never pushed", cwd=sub)
        self.git("commit", "-q", "-am", "bump sub")
        self.git("push", "-q")  # the superproject is level with its remote: its own status has nothing to say
        self.assertEqual(self.problems(), ["submodule libs/sub: 1 unpushed commit on HEAD"])

    def test_what_sits_in_a_nested_submodule_is_seen(self):
        self.add_submodule()
        deep = self.repo / "libs" / "sub" / "deep"
        (deep / "f").write_text("x")
        self.git("add", "f", cwd=deep)
        self.git("stash", "-q", cwd=deep)
        self.git("branch", "idea", cwd=deep)
        self.assertEqual(
            self.problems(), ["submodule libs/sub/deep: branch idea: no upstream", "submodule libs/sub/deep: 1 stash"]
        )

    def test_a_commit_only_a_tag_holds_is_seen(self):
        self.git("checkout", "-q", "--detach")
        self.git("commit", "-q", "--allow-empty", "-m", "side")
        self.git("tag", "keep")
        self.git("checkout", "-q", "main")
        # this clone has no mirror of the remote's tags: whose tag it is, nothing here says, and the line says so
        self.assertEqual(self.problems(), ["1 unpushed commit held by tags only" + containerctl.MAYBE_THE_REMOTES])
        self.git("tag", "-d", "keep")
        self.git("tag", "pushed-commit")  # a tag on a pushed commit holds nothing that would be lost
        self.assertEqual(self.problems(), [])

    def test_untracked_files_are_seen_even_when_the_config_hides_them(self):
        self.git("config", "status.showUntrackedFiles", "no")
        (self.repo / "notes.txt").write_text("x")
        self.assertEqual(self.problems(), ["untracked files"])


class CheckWorkspaceTests(unittest.TestCase):
    def test_extra_and_missing_detection(self):
        git = FakeGit({"a": {}, "b": {}, "forgotten": {}})
        report = check_workspace(
            expected=["a", "b", "gone"],
            list_entries=lambda: dirs(*["a", "b", "forgotten"]),
            run_git=git,
        )
        self.assertEqual(report.extra, ["forgotten"])
        self.assertEqual(report.missing, ["gone"])
        self.assertFalse(report.clean)

    def test_all_clean_and_matching(self):
        git = FakeGit({"a": {}, "b": {}})
        report = check_workspace(["a", "b"], lambda: dirs(*["a", "b"]), git)
        self.assertTrue(report.clean)

    def test_expected_none_skips_extra_missing(self):
        git = FakeGit({"a": {}})
        report = check_workspace(None, lambda: dirs(*["a"]), git)
        self.assertEqual(report.extra, [])
        self.assertEqual(report.missing, [])
        self.assertTrue(report.clean)

    def test_dirty_repo_makes_report_dirty(self):
        git = FakeGit({"a": {"porcelain": " M x\n"}})
        report = check_workspace(["a"], lambda: dirs(*["a"]), git)
        self.assertFalse(report.clean)


class LooseEntriesTests(unittest.TestCase):
    """What sits in the workspace and is not a directory is in no repo: it dies with the container."""

    def test_a_loose_file_makes_the_workspace_dirty_and_is_named(self):
        report = check_workspace(["a"], lambda: {"a": True, "notes.txt": False}, FakeGit({"a": {}}))
        self.assertEqual(report.loose, ["notes.txt"])
        self.assertFalse(report.clean)
        self.assertEqual(report.dirty, ["notes.txt"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_report("workspace", report)
        self.assertIn("✗ notes.txt — not a directory, in no repo", buf.getvalue())

    def test_a_file_we_injected_is_expected_while_it_is_as_the_host_has_it(self):
        entries = {"a": True, ".env": False, "notes.txt": False}
        report = check_workspace(["a"], lambda: entries, FakeGit({"a": {}}), {".env": "abc"}, lambda name: "abc")
        self.assertEqual((report.loose, report.changed), (["notes.txt"], []))  # only what nobody put there

    def test_an_injected_file_that_changed_or_cannot_be_read_is_loose_and_said_as_such(self):
        for found in ("other", None):
            report = check_workspace([], lambda: {".env": False}, FakeGit({}), {".env": "abc"}, lambda name: found)
            self.assertEqual((report.loose, report.changed, report.clean), ([".env"], [".env"], False))
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_report("workspace", report)
        self.assertIn("✗ .env — injected, but no longer as the host has it", buf.getvalue())

    def test_only_env_files_landing_in_the_workspace_itself_are_listed_with_the_hosts_digest(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "env"
            src.write_bytes(b"A=1\n")
            files = [
                containerctl.EnvFile(src=str(src), dest="/home/dev/workspace/.env"),
                containerctl.EnvFile(src=str(src), dest="/home/dev/workspace/app/.env"),  # in a repo: git's business
                containerctl.EnvFile(src=str(src), dest="/etc/x.env"),
                containerctl.EnvFile(src=str(Path(d) / "gone"), dest="/home/dev/workspace/.gone"),
            ]
            project = containerctl.ProjectConfig(name="demo", image="img", env_files=files)
            digest = hashlib.sha256(b"A=1\n").hexdigest()
            self.assertEqual(containerctl.injected_files(project), {".env": digest})

    def test_the_host_side_digest_asks_sha256sum_and_a_failure_is_no_answer(self):
        for returncode, stdout, expected in ((0, "abc  /home/dev/workspace/.env\n", "abc"), (1, "", None)):
            result = subprocess.CompletedProcess([], returncode, stdout, "")
            with mock.patch.object(containerctl.Docker, "exec_noninteractive", return_value=result) as run:
                digest = containerctl._host_digest("cctl-prj-demo", "/home/dev/workspace", "dev")
                self.assertEqual(digest(".env"), expected)
            run.assert_called_once_with("cctl-prj-demo", ["sha256sum", "--", "/home/dev/workspace/.env"], user="dev")

    def test_loose_files_count_without_a_manifest_too(self):
        report = check_workspace(None, lambda: {".env": False}, FakeGit({}))
        self.assertFalse(report.clean)

    def test_hidden_directories_are_checked_like_any_other(self):
        report = check_workspace(["a"], lambda: dirs("a", ".cache"), FakeGit({"a": {}, ".cache": {"is_git": False}}))
        self.assertEqual(report.extra, [".cache"])
        self.assertEqual(report.dirty, [".cache"])


@unittest.skipUnless(shutil.which("git"), "needs git")
class UpstreamTagsTests(unittest.TestCase):
    """With real git: a tag the remote has, on a commit that is on none of its branches, is not unpushed work.
    A fresh clone has all of them, and git doesn't say where a tag came from: spawn's clone mirrors them."""

    ENV = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        **{f"GIT_{who}_{what}": "t@example.invalid" for who in ("AUTHOR", "COMMITTER") for what in ("NAME", "EMAIL")},
    }

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.git(self.root, "init", "-q", "-b", "main", "up")
        self.git(self.root / "up", "commit", "-q", "--allow-empty", "-m", "on main")
        self.commit_off_branch("up", "v1")

    def tearDown(self):
        self._tmp.cleanup()

    def git(self, cwd, *args):
        subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, env=self.ENV)

    def commit_off_branch(self, repo, tag):
        """A commit no branch holds, and a tag on it"""
        self.git(self.root / repo, "checkout", "-q", "--detach")
        self.git(self.root / repo, "commit", "-q", "--allow-empty", "-m", f"only {tag} holds this")
        self.git(self.root / repo, "tag", tag)
        self.git(self.root / repo, "checkout", "-q", "main")

    def clone(self, name, *flags):
        self.git(self.root, "clone", "-q", *flags, "--", str(self.root / "up"), name)

    def runner(self):
        def run_git(repo, args):
            done = subprocess.run(
                ["git", "-C", str(self.root / repo), *args], capture_output=True, text=True, env=self.ENV
            )
            return done.returncode, done.stdout

        return run_git

    def status(self, name):
        return inspect_repo(name, self.runner())

    def test_a_fresh_spawn_clone_is_clean_whatever_tags_the_remote_has(self):
        self.clone("spawned", *containerctl.CLONE_MIRRORS_TAGS)
        status = self.status("spawned")
        self.assertEqual((status.tagged, status.problems()), (0, []))

    def test_a_tag_made_here_on_a_commit_no_branch_holds_is_still_seen(self):
        self.clone("spawned", *containerctl.CLONE_MIRRORS_TAGS)
        self.commit_off_branch("spawned", "mine")
        self.assertEqual(self.status("spawned").problems(), ["1 unpushed commit held by tags only"])

    def test_a_tag_that_came_with_a_later_fetch_is_the_remotes_too(self):
        self.clone("spawned", *containerctl.CLONE_MIRRORS_TAGS)
        self.commit_off_branch("up", "v2")
        self.git(self.root / "spawned", "fetch", "-q")
        self.assertEqual(self.status("spawned").tagged, 0)

    HINT = (
        " (or the remote's own tags: `git config --add remote.origin.fetch '+refs/tags/*:refs/upstream-tags/*'"
        " && git fetch` tells)"
    )

    def test_a_clone_without_the_mirror_errs_on_the_dirty_side_and_says_how_to_settle_it(self):
        self.clone("by-hand")
        status = self.status("by-hand")
        self.assertEqual(status.tagged, 1)
        self.assertEqual(status.problems(), ["1 unpushed commit held by tags only" + self.HINT])

    def test_what_the_hint_says_settles_it_and_a_tag_made_here_is_still_seen(self):
        self.clone("by-hand")
        refspec = "+refs/tags/*:refs/upstream-tags/*"
        self.git(self.root / "by-hand", "config", "--add", "remote.origin.fetch", refspec)
        self.git(self.root / "by-hand", "fetch", "-q")
        self.assertEqual(self.status("by-hand").problems(), [])
        self.commit_off_branch("by-hand", "mine")
        self.assertEqual(self.status("by-hand").problems(), ["1 unpushed commit held by tags only"])

    def test_a_submodule_on_a_commit_only_a_tag_of_its_remote_holds(self):
        """Never cloned by spawn, so never mirrored: said as maybe the remote's, and settled the same way"""
        allow = ["-c", "protocol.file.allow=always"]
        self.git(self.root, "init", "-q", "-b", "main", "super")
        self.git(self.root / "super", *allow, "submodule", "-q", "add", str(self.root / "up"), "lib")
        self.git(self.root / "super" / "lib", "checkout", "-q", "v1")
        sub = inspect_repo("lib", self.runner(), submodule_of="super")
        self.assertEqual(
            sub.problems(),
            ["1 unpushed commit held by tags only" + self.HINT, "1 unpushed commit on HEAD" + self.HINT],
        )
        lib = self.root / "super" / "lib"
        self.git(lib, "config", "--add", "remote.origin.fetch", "+refs/tags/*:refs/upstream-tags/*")
        self.git(lib, *allow, "fetch", "-q")
        self.assertEqual(inspect_repo("lib", self.runner(), submodule_of="super").problems(), [])


class HostAndLocalListingTests(unittest.TestCase):
    """Both sides answer the same question the same way: everything in the workspace, name -> is a directory."""

    def _host(self, returncode=0, stdout="", stderr=""):
        result = subprocess.CompletedProcess([], returncode, stdout, stderr)
        with mock.patch.object(containerctl.Docker, "exec_noninteractive", return_value=result) as run:
            return containerctl._host_list_entries("cctl-prj-demo", "/home/dev/workspace"), run

    def test_host_listing_reads_dirs_files_and_hidden_names(self):
        entries, run = self._host(stdout=".cache/\napp/\nnotes.txt\n")
        self.assertEqual(entries, {".cache": True, "app": True, "notes.txt": False})
        container, command = run.call_args.args
        self.assertEqual(container, "cctl-prj-demo")
        self.assertEqual(command[-1], "/home/dev/workspace")  # data ($1), not spliced into the script
        self.assertNotIn("/home/dev", command[2])

    def test_the_host_check_runs_as_the_projects_user_not_the_images_default(self):
        # git refuses a repo owned by somebody else ("dubious ownership"): every repo would read "not a git repo"
        project = containerctl.ProjectConfig(name="demo", image="i", user="ubuntu", home="/home/ubuntu")
        done = subprocess.CompletedProcess([], 0, "app/\n", "")
        with mock.patch.object(containerctl.Docker, "exec_noninteractive", return_value=done) as run:
            containerctl.project_report(project)
        self.assertGreater(run.call_count, 5)
        self.assertEqual({call.kwargs.get("user") for call in run.call_args_list}, {"ubuntu"})
        self.assertIn("/home/ubuntu/workspace/app", run.call_args_list[1].args[1])

    def test_what_ls_itself_said_is_not_read_as_docker_failing(self):
        with self.assertRaises(containerctl.DockerError) as ctx:
            self._host(returncode=2, stderr="ls: cannot open directory '/home/dev/workspace': Permission denied")
        self.assertIn("`ls` failed in the container", str(ctx.exception))
        self.assertNotIn("socket", str(ctx.exception))

    def test_host_listing_that_failed_is_not_an_empty_workspace(self):
        with self.assertRaises(containerctl.DockerError) as ctx:
            self._host(returncode=1, stderr="Error response from daemon: container abc is not running")
        self.assertEqual(str(ctx.exception), "Container is not running")

    def test_host_listing_of_a_container_without_a_workspace_is_empty_not_a_failure(self):
        entries, _ = self._host(stdout="")  # the script's `[ -d "$1" ] || exit 0`
        self.assertEqual(entries, {})

    def test_local_listing_gives_the_same_shape(self):
        with tempfile.TemporaryDirectory() as d:
            ws = Path(d) / "workspace"
            (ws / "app").mkdir(parents=True)
            (ws / ".cache").mkdir()
            (ws / "notes.txt").write_text("x")
            (ws / "link").symlink_to(ws / "app")
            with mock.patch.object(containerctl.Path, "home", return_value=Path(d)):
                self.assertEqual(
                    containerctl._local_list_entries(),
                    {"app": True, ".cache": True, "notes.txt": False, "link": False},
                )

    def test_local_listing_without_a_workspace_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(containerctl.Path, "home", return_value=Path(d)):
                self.assertEqual(containerctl._local_list_entries(), {})


class DirtyNamesTests(unittest.TestCase):
    """`report.dirty` is what `destroy` shows: unclean and extra dirs first, then configured-but-absent ones."""

    def test_clean_report_has_no_names(self):
        report = check_workspace(["a"], lambda: dirs(*["a"]), FakeGit({"a": {}}))
        self.assertTrue(report.clean)
        self.assertEqual(report.dirty, [])

    def test_names_unclean_extra_then_missing(self):
        git = FakeGit({"a": {}, "b": {"porcelain": " M x\n"}, "c": {}, "x": {"is_git": False}})
        report = check_workspace(["a", "b", "c", "gone"], lambda: dirs(*["x", "c", "b", "a"]), git)
        self.assertFalse(report.clean)
        self.assertEqual(report.dirty, ["b", "x", "gone"])

    def test_clean_but_unconfigured_dir_counts(self):
        report = check_workspace([], lambda: dirs(*["stray"]), FakeGit({"stray": {}}))
        self.assertEqual(report.dirty, ["stray"])


class PrintReportTests(unittest.TestCase):
    def _render(self, report):
        buf = io.StringIO()
        with redirect_stdout(buf):
            clean = print_report("workspace", report)
        return clean, buf.getvalue()

    def test_clean_output(self):
        git = FakeGit({"a": {}})
        report = check_workspace(["a"], lambda: dirs(*["a"]), git)
        clean, out = self._render(report)
        self.assertTrue(clean)
        self.assertIn("✓ a", out)
        self.assertIn("clean", out)
        self.assertNotIn("DIRTY", out)

    def test_dirty_output_lists_reasons(self):
        git = FakeGit({"a": {"porcelain": " M x\n", "branches": {"main": "[ahead 1]"}}, "extra": {}})
        report = check_workspace(["a"], lambda: dirs(*["a", "extra"]), git)
        clean, out = self._render(report)
        self.assertFalse(clean)
        self.assertIn("uncommitted changes", out)
        self.assertIn("not in config", out)
        self.assertIn("DIRTY", out)

    def test_missing_dir_reported(self):
        git = FakeGit({})
        report = check_workspace(["gone"], lambda: dirs(*[]), git)
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

    def test_read_injected_roundtrip_and_anything_else_is_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / ".config" / "containerctl"
            cfg.mkdir(parents=True)
            with mock.patch.object(Path, "home", return_value=Path(d)):
                self.assertEqual(containerctl.read_injected(), {})
                for written, read in (({".env": "abc"}, {".env": "abc"}), (["a"], {}), ({".env": 1}, {})):
                    (cfg / "injected.json").write_text(json.dumps(written))
                    self.assertEqual(containerctl.read_injected(), read)

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

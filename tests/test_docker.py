"""Tests for Docker command construction and ID-validation guards.

No real daemon: ``Docker._run`` and ``subprocess.run`` are replaced with
recorders so we assert on the argv that *would* be executed.
"""

import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

import containerctl
from containerctl import Docker, ProjectConfig, RepoSpec, ServiceConfig, get_errors
from tests.helper import RunRecorder, SubprocessRecorder, reset_errors


class CommandConstructionTests(unittest.TestCase):
    def setUp(self):
        reset_errors()
        self.rec = RunRecorder()
        self._patch = mock.patch.object(Docker, "_run", self.rec)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def test_start_stop_restart(self):
        self.assertTrue(Docker.start("abc"))
        self.assertEqual(self.rec.last, ["start", "abc"])
        self.assertTrue(Docker.stop("abc"))
        self.assertEqual(self.rec.last, ["stop", "abc"])
        self.assertTrue(Docker.restart("abc"))
        self.assertEqual(self.rec.last, ["restart", "abc"])

    def test_remove_container_force_inserts_flag(self):
        Docker.remove_container("abc", force=True)
        self.assertEqual(self.rec.last, ["rm", "-f", "abc"])
        Docker.remove_container("abc")
        self.assertEqual(self.rec.last, ["rm", "abc"])

    def test_remove_image_and_volume_and_network(self):
        Docker.remove_image("img:tag", force=True)
        self.assertEqual(self.rec.last, ["rmi", "-f", "img:tag"])
        Docker.remove_volume("vol1")
        self.assertEqual(self.rec.last, ["volume", "rm", "vol1"])
        Docker.remove_network("net1")
        self.assertEqual(self.rec.last, ["network", "rm", "net1"])

    def test_containers_passes_all_flag(self):
        Docker.containers(all_containers=True)
        self.assertEqual(self.rec.last, ["ps", "-a", "--format", "{{json .}}"])
        Docker.containers(all_containers=False)
        self.assertEqual(self.rec.last, ["ps", "--format", "{{json .}}"])

    def test_returncode_propagates_to_bool(self):
        self.rec.returncode = 1
        self.assertFalse(Docker.start("abc"))

    def test_exec_noninteractive_plain(self):
        Docker.exec_noninteractive("abc", ["ls", "-la"])
        self.assertEqual(self.rec.last, ["exec", "abc", "ls", "-la"])

    def test_exec_noninteractive_workdir_inserts_flag(self):
        Docker.exec_noninteractive("abc", ["ls"], workdir="/home/dev/workspace")
        self.assertEqual(self.rec.last, ["exec", "-w", "/home/dev/workspace", "abc", "ls"])


class IdGuardTests(unittest.TestCase):
    """An invalid id must never reach the runtime, and must log an error."""

    def setUp(self):
        reset_errors()
        self.rec = RunRecorder()
        self._patch = mock.patch.object(Docker, "_run", self.rec)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def test_start_rejects_bad_id(self):
        self.assertFalse(Docker.start("a;rm -rf /"))
        self.assertEqual(self.rec.calls, [])
        self.assertEqual(get_errors()[0]["cat"], "docker.invalid_id")

    def test_remove_volume_rejects_bad_name(self):
        self.assertFalse(Docker.remove_volume("bad name"))
        self.assertEqual(self.rec.calls, [])

    def test_connect_network_validates_both_args(self):
        self.assertFalse(Docker.connect_network("net", "bad;id"))
        self.assertEqual(self.rec.calls, [])


class JsonParsingTests(unittest.TestCase):
    def setUp(self):
        reset_errors()

    def test_run_json_parses_lines_and_skips_blanks(self):
        rec = RunRecorder(stdout='{"a": 1}\n\n{"a": 2}\n')
        with mock.patch.object(Docker, "_run", rec):
            items = Docker._run_json(["ps"])
        self.assertEqual(items, [{"a": 1}, {"a": 2}])

    def test_run_json_nonzero_returns_empty_and_logs(self):
        rec = RunRecorder(returncode=1, stderr="boom")
        with mock.patch.object(Docker, "_run", rec):
            self.assertEqual(Docker._run_json(["ps"]), [])
        self.assertEqual(get_errors()[0]["cat"], "docker.command")

    def test_run_json_bad_line_logs_parse_error(self):
        rec = RunRecorder(stdout='{"a": 1}\nnot json\n')
        with mock.patch.object(Docker, "_run", rec):
            items = Docker._run_json(["ps"])
        self.assertEqual(items, [{"a": 1}])
        self.assertEqual(get_errors()[0]["cat"], "docker.json_parse")

    def test_containers_filters_invalid_rows(self):
        # One good row, one missing required field -> only the good one survives.
        good = '{"ID": "abc", "Names": "web", "Image": "nginx", "Status": "Up"}'
        bad = '{"Names": "x"}'
        rec = RunRecorder(stdout=f"{good}\n{bad}\n")
        with mock.patch.object(Docker, "_run", rec):
            rows = Docker.containers()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Names"], "web")


class StreamingCommandTests(unittest.TestCase):
    """logs/exec/shell call subprocess.run directly (no capture)."""

    def setUp(self):
        reset_errors()
        self.rec = SubprocessRecorder()
        self._patch = mock.patch.object(containerctl.subprocess, "run", self.rec)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def test_logs_flags(self):
        Docker.logs("abc", follow=True, tail=100)
        self.assertEqual(self.rec.last, ["docker", "logs", "-f", "--tail", "100", "abc"])

    def test_logs_minimal(self):
        Docker.logs("abc")
        self.assertEqual(self.rec.last, ["docker", "logs", "abc"])

    def test_exec_interactive(self):
        Docker.exec("abc", ["ls", "-la"], interactive=True)
        self.assertEqual(self.rec.last, ["docker", "exec", "-it", "abc", "ls", "-la"])

    def test_exec_with_user_inserts_flag(self):
        Docker.exec("abc", ["ls"], user="root")
        self.assertEqual(self.rec.last, ["docker", "exec", "-it", "-u", "root", "abc", "ls"])

    def test_exec_rejects_bad_user(self):
        for bad in ("ro;ot", ""):
            Docker.exec("abc", ["ls"], user=bad)
        self.assertEqual(self.rec.calls, [])
        self.assertEqual({e["cat"] for e in get_errors()}, {"docker.invalid_user"})

    def test_exec_with_workdir_inserts_flag(self):
        Docker.exec("abc", ["bash"], workdir="/home/dev/workspace")
        self.assertEqual(self.rec.last, ["docker", "exec", "-it", "-w", "/home/dev/workspace", "abc", "bash"])

    def test_exec_rejects_bad_workdir(self):
        for bad in ("/home;rm -rf /", ""):
            Docker.exec("abc", ["bash"], workdir=bad)
        self.assertEqual(self.rec.calls, [])
        self.assertEqual({e["cat"] for e in get_errors()}, {"docker.invalid_workdir"})

    def test_shell_delegates_to_exec(self):
        Docker.shell("abc", shell="/bin/bash")
        self.assertEqual(self.rec.last, ["docker", "exec", "-it", "abc", "/bin/bash"])

    def test_logs_rejects_bad_id(self):
        Docker.logs("bad;id")
        self.assertEqual(self.rec.calls, [])
        self.assertEqual(get_errors()[0]["cat"], "docker.invalid_id")


class ProjectAndServiceRunTests(unittest.TestCase):
    """create_*_container build the full `docker run ...` line."""

    def setUp(self):
        reset_errors()
        self.rec = RunRecorder()
        self._patch = mock.patch.object(Docker, "_run", self.rec)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def test_create_project_container_includes_labels_and_network(self):
        project = ProjectConfig(
            name="demo",
            image="ubuntu:22.04",
            repos=[RepoSpec("https://github.com/u/repo.git")],
        )
        # ensure_network runs `network inspect` (ok) then we expect the run line.
        self.assertTrue(Docker.create_project_container(project))
        argv = self.rec.last
        self.assertEqual(argv[:2], ["run", "-d"])
        self.assertIn("--name", argv)
        self.assertIn("cctl-prj-demo", argv)
        self.assertIn("cctl-net-demo", argv)
        self.assertIn("cctl.project=demo", argv)
        self.assertEqual(argv[-3:], ["ubuntu:22.04", "sleep", "infinity"])

    def test_create_service_container_maps_ports_volumes_env(self):
        service = ServiceConfig(
            name="web",
            image="nginx:latest",
            restart="always",
            ports=["8080:80"],
            volumes=["/data:/data:ro"],
            env=["FOO=bar"],
        )
        self.assertTrue(Docker.create_service_container(service))
        argv = self.rec.last
        self.assertEqual(argv[:2], ["run", "-d"])
        self.assertIn("cctl-svc-web", argv)
        self.assertEqual(argv[argv.index("--restart") + 1], "always")
        self.assertEqual(argv[argv.index("-p") + 1], "8080:80")
        self.assertEqual(argv[argv.index("-v") + 1], "/data:/data:ro")
        self.assertEqual(argv[argv.index("-e") + 1], "FOO=bar")
        self.assertEqual(argv[-1], "nginx:latest")


class SpawnAndEnterShellFreeTests(unittest.TestCase):
    """spawn clones through argv (no shell); enter uses a *constant* `bash -c` with the landing
    dir passed as $1 — never an interpolated string-join, so no injection and no TOCTOU."""

    def setUp(self):
        reset_errors()
        self.rec = RunRecorder()
        self._run_patch = mock.patch.object(Docker, "_run", self.rec)
        self._run_patch.start()
        self.sub = SubprocessRecorder()
        self._sub_patch = mock.patch.object(containerctl.subprocess, "run", self.sub)
        self._sub_patch.start()
        self._install_patch = mock.patch.object(containerctl, "install_self", return_value=True)
        self._install_patch.start()
        self.project = ProjectConfig(
            name="demo",
            image="ubuntu:22.04",
            repos=[RepoSpec("https://github.com/u/repo.git")],
        )

    def tearDown(self):
        self._install_patch.stop()
        self._sub_patch.stop()
        self._run_patch.stop()

    def test_spawn_clones_via_workdir_argv_with_option_terminator(self):
        with redirect_stdout(io.StringIO()):
            self.assertTrue(Docker.spawn_project(self.project))
        tail = ["git", "clone", "--", "https://github.com/u/repo.git"]
        self.assertIn(["exec", "-w", "/home/dev/workspace", "cctl-prj-demo"] + tail, self.rec.calls)
        self.assertTrue(all("bash" not in call for call in self.rec.calls))

    def test_spawn_clone_branch_and_dir_stay_before_and_after_terminator(self):
        self.project.repos = [RepoSpec("https://github.com/u/repo.git", dir="d", branch="main")]
        with redirect_stdout(io.StringIO()):
            self.assertTrue(Docker.spawn_project(self.project))
        tail = ["git", "clone", "-b", "main", "--", "https://github.com/u/repo.git", "d"]
        self.assertIn(["exec", "-w", "/home/dev/workspace", "cctl-prj-demo"] + tail, self.rec.calls)

    # enter attaches with a constant bash -c; the landing dir rides in as $1 (data, never
    # interpolated), and `cd -- "$1" || cd /home/dev` does the fallback atomically in-container.
    ENTER_SCRIPT = 'cd -- "$1" 2>/dev/null || cd /home/dev; exec bash'

    def _enter(self):
        with mock.patch.object(Docker, "find_project_container", return_value={"State": "running"}):
            with redirect_stdout(io.StringIO()):
                Docker.enter_project(self.project)

    def _expected(self, workdir):
        return ["docker", "exec", "-it", "cctl-prj-demo", "bash", "-c", self.ENTER_SCRIPT, "bash", workdir]

    def test_enter_single_repo_attaches_atomically_without_probe(self):
        self._enter()
        self.assertEqual(self.rec.calls, [])  # no `test -d` probe — the cd-or-home fallback is atomic, in-container
        self.assertEqual(self.sub.last, self._expected("/home/dev/workspace/repo"))

    def test_enter_multi_repo_lands_in_workspace_root(self):
        self.project.repos = [RepoSpec("https://github.com/u/a.git"), RepoSpec("https://github.com/u/b.git")]
        self._enter()
        self.assertEqual(self.sub.last, self._expected("/home/dev/workspace"))

    def test_enter_without_repos_lands_in_home(self):
        self.project.repos = []
        self._enter()
        self.assertEqual(self.rec.calls, [])
        self.assertEqual(self.sub.last, self._expected("/home/dev"))

    def test_enter_dir_is_passed_as_data_not_interpolated(self):
        # The script is a constant; the dir is the LAST argv element ($1), and it is quoted +
        # option-terminated so it can never be parsed as shell code or a cd option.
        self._enter()
        script = self.sub.last[self.sub.last.index("-c") + 1]
        self.assertEqual(script, self.ENTER_SCRIPT)
        self.assertIn('"$1"', script)
        self.assertIn("cd --", script)
        self.assertIn("|| cd /home/dev", script)
        self.assertEqual(self.sub.last[-1], "/home/dev/workspace/repo")


if __name__ == "__main__":
    unittest.main()

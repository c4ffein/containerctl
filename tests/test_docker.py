"""Tests for Docker command construction and ID-validation guards.

No real daemon: ``Docker._run`` and ``subprocess.run`` are replaced with
recorders so we assert on the argv that *would* be executed.
"""

import base64
import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import containerctl
from containerctl import KNOWN_HOSTS, Docker, ProjectConfig, RepoSpec, ServiceConfig, get_errors
from tests.helper import RunRecorder, SubprocessRecorder, container, reset_errors


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

    def test_containers_passes_all_flag_and_never_computes_size_by_default(self):
        # --size=false was dropped once by accident and `ps` went from 75ms to seconds; keep it pinned
        Docker.list_containers(all_containers=True).logged()
        self.assertEqual(self.rec.last, ["ps", "--size=false", "-a", "--format", "{{json .}}"])
        Docker.list_containers(all_containers=False).logged()
        self.assertEqual(self.rec.last, ["ps", "--size=false", "--format", "{{json .}}"])
        Docker.list_containers(include_size=True).logged()
        self.assertEqual(self.rec.last, ["ps", "-a", "--format", "{{json .}}"])

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
        self.assertEqual(get_errors()[0].cat, "docker.invalid_id")

    def test_remove_volume_rejects_bad_name(self):
        self.assertFalse(Docker.remove_volume("bad name"))
        self.assertEqual(self.rec.calls, [])

    def test_connect_network_validates_both_args(self):
        self.assertFalse(Docker.connect_network("net", "bad;id"))
        self.assertEqual(self.rec.calls, [])


class JsonParsingTests(unittest.TestCase):
    def setUp(self):
        reset_errors()

    def test_list_parses_lines_and_skips_blanks(self):
        rec = RunRecorder(stdout='{"a": 1}\n\n{"a": 2}\n')
        with mock.patch.object(Docker, "_run", rec):
            listing = Docker._list(["ps"], dict)
        self.assertEqual((listing.rows, listing.failure), ([{"a": 1}, {"a": 2}], None))

    def test_list_hands_a_failed_call_back_without_logging(self):
        rec = RunRecorder(returncode=1, stderr="boom\n")
        with mock.patch.object(Docker, "_run", rec):
            listing = Docker._list(["ps", "-a"], dict)
        self.assertEqual(listing.rows, [])
        self.assertEqual(
            listing.failure, containerctl.Failure(*containerctl.UNKNOWN_DOCKER_FAILURE, "docker ps -a: boom")
        )
        self.assertEqual(get_errors(), [])  # the caller's call: an issue for the fetcher, a log line for the CLI

    def test_the_one_shot_flavour_logs_the_failure_and_reads_as_no_rows(self):
        rec = RunRecorder(returncode=1, stderr="boom")
        with mock.patch.object(Docker, "_run", rec):
            self.assertEqual(Docker.list_containers().logged(), [])
            self.assertEqual(Docker.list_images().logged(), [])
        self.assertEqual(
            [(e.cat, e.msg) for e in get_errors()],
            [
                ("docker.command", "docker ps --size=false -a --format {{json .}}: boom"),
                ("docker.command", "docker images --format {{json .}}: boom"),
            ],
        )

    def test_each_line_stands_alone_a_bad_one_is_skipped_and_handed_back(self):
        good = '{"ID": "abc", "Names": "web", "Image": "nginx", "Status": "Up"}'
        rec = RunRecorder(stdout=f'{good}\nnot json\n{{"Names": "x"}}\n{good}\n')
        with mock.patch.object(Docker, "_run", rec):
            listing = Docker._list(["ps"], containerctl.Container.from_dict)
        self.assertEqual([c.Names for c in listing.rows], ["web", "web"])
        self.assertIsNone(listing.failure)
        self.assertEqual(len(listing.skipped), 2)
        self.assertRegex(listing.skipped[0], r"^docker ps: Expecting value.* in not json$")
        self.assertEqual(
            listing.skipped[1], """docker ps: Missing fields: ['ID', 'Image', 'Status'] in {"Names": "x"}"""
        )
        self.assertEqual(get_errors(), [])

    def test_a_lookup_does_not_read_a_failed_ps_as_no_such_container(self):
        rec = RunRecorder(returncode=1, stderr="Cannot connect to the Docker daemon. Is the docker daemon running?")
        with mock.patch.object(Docker, "_run", rec):
            for lookup in (Docker.find_project_container, Docker.find_service_container):
                with self.assertRaises(containerctl.DockerError) as raised:
                    lookup("demo")
                self.assertEqual(str(raised.exception), "Docker daemon is not running")
                self.assertEqual(raised.exception.failure.hint, "start docker")
        with mock.patch.object(Docker, "_run", RunRecorder(stdout="")):
            self.assertIsNone(Docker.find_project_container("demo"))  # docker answered: there is none

    def test_nothing_is_created_or_removed_when_ps_failed(self):
        """`up` must not create over a container it couldn't see, `check` must not say there is nothing to check"""
        rec = RunRecorder(returncode=1, stderr="boom")
        service = ServiceConfig(name="web", image="nginx")
        project = ProjectConfig(name="demo", image="i")
        actions = (
            lambda: Docker.service_up(service),
            lambda: Docker.service_down(service),
            lambda: Docker.service_recreate(service),
            lambda: Docker.destroy_project(project),
            lambda: Docker.enter_project(project),
            lambda: containerctl._check_one(project),
        )
        with mock.patch.object(Docker, "_run", rec), redirect_stdout(io.StringIO()) as out:
            for action in actions:
                with self.assertRaises(containerctl.DockerError):
                    action()
        self.assertEqual({call[0] for call in rec.calls}, {"ps"})  # the listing, and nothing after it
        self.assertEqual(out.getvalue(), "")

    def test_the_one_shot_flavour_logs_skipped_rows(self):
        rec = RunRecorder(stdout='{"Names": "x"}\n')
        with mock.patch.object(Docker, "_run", rec):
            self.assertEqual(Docker.list_containers().logged(), [])
        self.assertEqual([e.cat for e in get_errors()], ["docker.row_parse"])

    def test_known_failures_get_our_words_and_a_hint(self):
        def classify(stderr, returncode=1):
            call = subprocess.CompletedProcess(["docker", "ps"], returncode, "", stderr)
            failure = containerctl.docker_failure(["ps"], call)
            return failure.msg, failure.hint

        down = "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?"
        denied = "permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock"
        self.assertEqual(classify(down), ("Docker daemon is not running", "start docker"))
        self.assertEqual(classify(denied), ("No permission on the docker socket", "add yourself to the docker group"))
        self.assertEqual(
            classify("docker: command not found on PATH", 127),
            ("docker is not installed", "install docker, or fix PATH"),
        )
        self.assertEqual(classify("something new"), containerctl.UNKNOWN_DOCKER_FAILURE)

    def test_a_silent_failure_still_has_something_to_log(self):
        call = subprocess.CompletedProcess(["docker", "ps"], 3, "", "")
        self.assertEqual(containerctl.docker_failure(["ps"], call).detail, "docker ps: exit code 3")

    def test_no_docker_on_path_is_classified_end_to_end(self):
        with mock.patch.object(containerctl.subprocess, "run", side_effect=FileNotFoundError):
            listing = Docker.list_volumes()
        self.assertEqual(listing.failure.msg, "docker is not installed")

    def test_containers_filters_invalid_rows(self):
        # One good row, one missing required field -> only the good one survives.
        good = '{"ID": "abc", "Names": "web", "Image": "nginx", "Status": "Up"}'
        bad = '{"Names": "x"}'
        rec = RunRecorder(stdout=f"{good}\n{bad}\n")
        with mock.patch.object(Docker, "_run", rec):
            rows = Docker.list_containers().logged()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].Names, "web")


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
        self.assertEqual({e.cat for e in get_errors()}, {"docker.invalid_user"})

    def test_exec_with_workdir_inserts_flag(self):
        Docker.exec("abc", ["bash"], workdir="/home/dev/workspace")
        self.assertEqual(self.rec.last, ["docker", "exec", "-it", "-w", "/home/dev/workspace", "abc", "bash"])

    def test_exec_rejects_bad_workdir(self):
        for bad in ("/home;rm -rf /", ""):
            Docker.exec("abc", ["bash"], workdir=bad)
        self.assertEqual(self.rec.calls, [])
        self.assertEqual({e.cat for e in get_errors()}, {"docker.invalid_workdir"})

    def test_shell_delegates_to_exec(self):
        Docker.shell("abc", shell="/bin/bash")
        self.assertEqual(self.rec.last, ["docker", "exec", "-it", "abc", "/bin/bash"])

    def test_logs_rejects_bad_id(self):
        Docker.logs("bad;id")
        self.assertEqual(self.rec.calls, [])
        self.assertEqual(get_errors()[0].cat, "docker.invalid_id")


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
        self.assertEqual(argv[:3], ["run", "-d", "--init"])  # --init reaps what `docker exec` leaves behind
        self.assertIn("--name", argv)
        self.assertIn("cctl-prj-demo", argv)
        self.assertIn("cctl-net-prj-demo", argv)
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
        # no networks / hardening configured -> none of their flags, and no network commands at all;
        # the only call before the run is the image presence check (present with the recorder: no build)
        for flag in ("--network", "--read-only", "--tmpfs", "--cap-drop", "--cap-add", "--pids-limit"):
            self.assertNotIn(flag, argv)
        self.assertEqual(self.rec.calls[:-1], [["image", "inspect", "nginx:latest"]])

    def test_create_service_container_command_follows_the_image(self):
        service = ServiceConfig(name="h", image="hermes:latest", command=["gateway", "run"])
        self.assertTrue(Docker.create_service_container(service))
        self.assertEqual(self.rec.last[-3:], ["hermes:latest", "gateway", "run"])

    def test_create_service_container_hardening_flags(self):
        service = ServiceConfig(
            name="web",
            image="nginx",
            read_only=True,
            tmpfs=["/tmp", "/run:size=16m"],
            cap_drop=["ALL"],
            cap_add=["NET_BIND_SERVICE"],
            pids_limit=256,
        )
        self.assertTrue(Docker.create_service_container(service))
        argv = self.rec.last
        self.assertIn("--read-only", argv)
        self.assertEqual([argv[i + 1] for i, a in enumerate(argv) if a == "--tmpfs"], ["/tmp", "/run:size=16m"])
        self.assertEqual(argv[argv.index("--cap-drop") + 1], "ALL")
        self.assertEqual(argv[argv.index("--cap-add") + 1], "NET_BIND_SERVICE")
        self.assertEqual(argv[argv.index("--pids-limit") + 1], "256")
        self.assertEqual(argv[-1], "nginx")

    def test_create_service_container_networks_first_on_run_rest_connected(self):
        service = ServiceConfig(name="web", image="nginx", networks=["proxy", "backend"])
        self.assertTrue(Docker.create_service_container(service))
        calls = self.rec.calls
        # every network is ensured (inspect succeeds with the recorder, so no create), then the image, before the run
        run_idx = next(i for i, c in enumerate(calls) if c[0] == "run")
        self.assertEqual(
            calls[:run_idx],
            [["network", "inspect", "proxy"], ["network", "inspect", "backend"], ["image", "inspect", "nginx"]],
        )
        argv = calls[run_idx]
        self.assertEqual(argv[argv.index("--network") + 1], "proxy")
        self.assertEqual(argv.count("--network"), 1)
        # the second network is connected after creation, the first is not connected twice
        self.assertEqual(
            calls[run_idx + 1 :], [["network", "inspect", "backend"], ["network", "connect", "backend", "cctl-svc-web"]]
        )

    def test_create_service_container_fails_when_network_cannot_be_ensured(self):
        self.rec.returncode = 1  # inspect fails and so does create
        service = ServiceConfig(name="web", image="nginx", networks=["proxy"])
        self.assertFalse(Docker.create_service_container(service))
        self.assertTrue(all(c[0] != "run" for c in self.rec.calls))


class ServiceUpNetworkTests(unittest.TestCase):
    """`service up` on an existing container connects networks added to the config since creation."""

    def setUp(self):
        reset_errors()
        self.rec = RunRecorder()
        self._run = mock.patch.object(Docker, "_run", self.rec)
        self._run.start()
        self.service = ServiceConfig(name="web", image="nginx", networks=["proxy", "backend"])

    def tearDown(self):
        self._run.stop()

    def _inspect(self, *networks):
        info = containerctl.ContainerInspect(
            Id="abc",
            Name="/cctl-svc-web",
            Created="",
            State={},
            Image="nginx",
            Config={},
            NetworkSettings={"Networks": {n: {} for n in networks}},
        )
        return mock.patch.object(Docker, "inspect_container", return_value=info)

    def test_running_container_gets_missing_networks_only(self):
        with mock.patch.object(Docker, "find_service_container", return_value=container("c")):
            with self._inspect("proxy"), redirect_stdout(io.StringIO()):
                self.assertTrue(Docker.service_up(self.service))
        self.assertEqual(
            self.rec.calls, [["network", "inspect", "backend"], ["network", "connect", "backend", "cctl-svc-web"]]
        )

    def test_already_connected_everywhere_is_a_noop(self):
        with mock.patch.object(Docker, "find_service_container", return_value=container("c")):
            with self._inspect("proxy", "backend", "bridge"), redirect_stdout(io.StringIO()):
                self.assertTrue(Docker.service_up(self.service))
        self.assertEqual(self.rec.calls, [])  # extra networks (bridge) are left alone

    def test_stopped_container_is_started_then_connected(self):
        with mock.patch.object(Docker, "find_service_container", return_value=container("c", "exited")):
            with self._inspect(), redirect_stdout(io.StringIO()):
                self.assertTrue(Docker.service_up(self.service))
        self.assertEqual(self.rec.calls[0], ["start", "cctl-svc-web"])
        self.assertIn(["network", "connect", "proxy", "cctl-svc-web"], self.rec.calls)
        self.assertIn(["network", "connect", "backend", "cctl-svc-web"], self.rec.calls)

    def test_missing_container_is_created(self):
        with mock.patch.object(Docker, "find_service_container", return_value=None):
            with redirect_stdout(io.StringIO()):
                self.assertTrue(Docker.service_up(self.service))
        self.assertTrue(any(c[0] == "run" for c in self.rec.calls))

    def test_connect_failure_is_reported(self):
        with mock.patch.object(Docker, "find_service_container", return_value=container("c")):
            with self._inspect("proxy"), redirect_stdout(io.StringIO()):
                self.rec.returncode = 1
                self.assertFalse(Docker.service_up(self.service))
        self.assertEqual(get_errors()[-1].cat, "service.network")


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
        self._env_patch = mock.patch.object(containerctl, "resolve_env", return_value={})  # never the real ~/.config
        self._env_patch.start()
        self.project = ProjectConfig(
            name="demo",
            image="ubuntu:22.04",
            repos=[RepoSpec("https://github.com/u/repo.git")],
        )

    def tearDown(self):
        self._env_patch.stop()
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

    def test_spawn_pins_forge_host_keys_instead_of_scanning(self):
        with tempfile.NamedTemporaryFile(delete=False) as key:
            key.write(b"not a real key")
        self.addCleanup(os.unlink, key.name)
        self.project.ssh_key = key.name
        with redirect_stdout(io.StringIO()):
            self.assertTrue(Docker.spawn_project(self.project))
        tee = ["exec", "-i", "-u", "dev", "cctl-prj-demo", "tee", "-a", "/home/dev/.ssh/known_hosts"]
        self.assertIn(tee, self.rec.calls)
        self.assertEqual(self.rec.inputs[self.rec.calls.index(tee)], KNOWN_HOSTS)
        flat = " ".join(" ".join(c) for c in self.rec.calls)
        self.assertNotIn("ssh-keyscan", flat)
        self.assertNotIn("bash", flat)

    def test_known_hosts_constant_is_well_formed(self):
        # One `host ssh-ed25519 <b64>` line per forge; the blob is an ed25519 wire key (type string + 32 bytes)
        lines = KNOWN_HOSTS.splitlines()
        self.assertEqual([ln.split()[0] for ln in lines], ["github.com", "gitlab.com"])
        for line in lines:
            host, kind, blob = line.split(" ")
            self.assertEqual(kind, "ssh-ed25519")
            raw = base64.b64decode(blob, validate=True)
            self.assertEqual(raw[:4], (11).to_bytes(4, "big"))
            self.assertEqual(raw[4:15], b"ssh-ed25519")
            self.assertEqual(len(raw), 4 + 11 + 4 + 32)

    # enter attaches with a constant bash -c; the landing dir rides in as $1 (data, never
    # interpolated), and `cd -- "$1" || cd /home/dev` does the fallback atomically in-container.
    ENTER_SCRIPT = 'cd -- "$1" 2>/dev/null || cd /home/dev; exec bash'

    def _enter(self, **kwargs):
        with mock.patch.object(Docker, "find_project_container", return_value=container("c")):
            with redirect_stdout(io.StringIO()):
                Docker.enter_project(self.project, **kwargs)

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

    def test_enter_workdir_override_wins_over_repo_landing(self):
        self._enter(workdir="/home/dev/workspace/repo/sub")
        self.assertEqual(self.sub.last, self._expected("/home/dev/workspace/repo/sub"))

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


class ExecEnvTests(unittest.TestCase):
    """Shared env reaches `docker exec` as bare `-e NAME` flags; the value only rides in our own environment."""

    def setUp(self):
        reset_errors()
        self.sub = SubprocessRecorder()
        self._sub_patch = mock.patch.object(containerctl.subprocess, "run", self.sub)
        self._sub_patch.start()

    def tearDown(self):
        self._sub_patch.stop()

    def test_exec_env_is_bare_flag_plus_process_environment(self):
        Docker.exec("abc", ["bash"], env={"TOK": "s3cret", "TZ": "UTC"})
        self.assertEqual(self.sub.last, ["docker", "exec", "-it", "-e", "TOK", "-e", "TZ", "abc", "bash"])
        self.assertNotIn("s3cret", " ".join(self.sub.last))
        env = self.sub.kwargs[-1]["env"]
        self.assertEqual((env["TOK"], env["TZ"]), ("s3cret", "UTC"))
        self.assertEqual(env["PATH"], containerctl.os.environ["PATH"])  # ours, extended - not replaced

    def test_exec_without_env_leaves_environment_alone(self):
        Docker.exec("abc", ["bash"])
        Docker.exec("abc", ["bash"], env={})
        self.assertEqual(self.sub.kwargs, [{}, {}])

    def test_exec_rejects_bad_env_name(self):
        for bad in ("1TOK", "A-B", "A=B", ""):
            Docker.exec("abc", ["bash"], env={bad: "x"})
        self.assertEqual(self.sub.calls, [])
        self.assertEqual({e.cat for e in get_errors()}, {"docker.invalid_env"})

    def _enter(self, env, logged=()):
        def fake_resolve(project):
            for cat, msg in logged:
                containerctl.log_error(cat, msg)
            return env

        err = io.StringIO()
        with (
            mock.patch.object(Docker, "find_project_container", return_value=container("c")),
            mock.patch.object(containerctl, "install_self", return_value=True),
            mock.patch.object(containerctl, "resolve_env", side_effect=fake_resolve) as resolve,
            redirect_stdout(io.StringIO()),
            redirect_stderr(err),
        ):
            Docker.enter_project(ProjectConfig(name="demo", image="i"))
        return resolve, err.getvalue()

    def test_enter_injects_resolved_env(self):
        resolve, stderr = self._enter({"TOK": "s3cret", "TZ": "UTC"})
        self.assertEqual(resolve.call_args.args[0].name, "demo")
        self.assertEqual(self.sub.last[:7], ["docker", "exec", "-it", "-e", "TOK", "-e", "TZ"])
        self.assertEqual(self.sub.kwargs[-1]["env"]["TOK"], "s3cret")
        self.assertEqual(stderr, "")

    def test_enter_echoes_env_problems_only(self):
        problems = [("env.secret", "TOK: no 'tok' in secrets.toml"), ("other.cat", "nope")]
        _, stderr = self._enter({}, logged=problems)
        self.assertNotIn("-e", self.sub.last)
        self.assertEqual(stderr, "env: TOK: no 'tok' in secrets.toml\n")


class InstallSelfTests(unittest.TestCase):
    """install_self runs on every `enter`: it creates as dev and chowns the two files `docker cp` wrote as root,
    by name. The old `chown -R dev:dev ~/.local ~/.config` walked the whole home each time and flipped the owner
    of everything under it."""

    def setUp(self):
        reset_errors()
        self.rec = RunRecorder()
        self._patch = mock.patch.object(Docker, "_run", self.rec)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def _execs(self):
        return [c for c in self.rec.calls if c[0] == "exec"]

    def test_creates_as_dev_and_chowns_only_the_copied_files(self):
        self.assertTrue(containerctl.install_self("cctl-prj-demo", ["app"]))
        execs = self._execs()
        dirs = ["/home/dev/.local/bin", "/home/dev/.config/containerctl"]
        self.assertEqual(execs[0], ["exec", "-u", "dev", "cctl-prj-demo", "mkdir", "-p", *dirs])
        chowns = [c for c in execs if "chown" in c]
        self.assertEqual(
            chowns,
            [
                [
                    "exec",
                    "cctl-prj-demo",
                    "chown",
                    "dev:dev",
                    "/home/dev/.local/bin/containerctl",
                    "/home/dev/.config/containerctl/expected.json",
                ]
            ],
        )
        self.assertTrue(all("-R" not in c for c in self.rec.calls))

    def test_copies_self_and_manifest(self):
        containerctl.install_self("cctl-prj-demo", ["app", "lib"])
        cps = [c for c in self.rec.calls if c[0] == "cp"]
        self.assertEqual(cps[0][2], "cctl-prj-demo:/home/dev/.local/bin/containerctl")
        self.assertEqual(cps[1][2], "cctl-prj-demo:/home/dev/.config/containerctl/expected.json")

    def test_bashrc_append_runs_as_dev_with_a_constant_script(self):
        containerctl.install_self("cctl-prj-demo", [])
        shells = [c for c in self._execs() if "sh" in c]
        self.assertEqual(len(shells), 1)
        self.assertEqual(shells[0][:4], ["exec", "-u", "dev", "cctl-prj-demo"])
        self.assertIn(".local/bin", shells[0][-1])

    def test_failed_copy_of_self_aborts(self):
        self.rec.returncode = 1
        self.assertFalse(containerctl.install_self("cctl-prj-demo", []))
        self.assertTrue(all("chown" not in c for c in self.rec.calls))


class MissingDockerTests(unittest.TestCase):
    """No docker on PATH reads as a failed call, reported like any other, never a traceback"""

    def test_missing_binary_is_a_failed_call(self):
        reset_errors()
        missing = FileNotFoundError(2, "No such file or directory", "docker")
        with mock.patch.object(containerctl.subprocess, "run", side_effect=missing):
            self.assertEqual(Docker.list_containers().logged(), [])
            self.assertFalse(Docker.start("abc"))
        self.assertEqual({e.cat for e in get_errors()}, {"docker.command"})
        self.assertIn("command not found", " ".join(e.msg for e in get_errors()))


class ImageBuildTests(unittest.TestCase):
    """`docker build` argv, the no-context stdin path, and the auto-build before `docker run`."""

    def setUp(self):
        reset_errors()
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        (self.dir / "hermes.toml").write_text("")
        (self.dir / "hermes.Dockerfile").write_text("FROM debian:12\n")
        self.image = containerctl.ImageConfig.from_toml(self.dir / "hermes.toml")
        self.rec = RunRecorder()
        self._patches = [
            mock.patch.object(Docker, "_run", self.rec),
            mock.patch.object(containerctl, "IMAGES_DIR", self.dir),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def test_build_without_context_feeds_dockerfile_on_stdin(self):
        with redirect_stdout(io.StringIO()):
            self.assertTrue(Docker.build_image(self.image))
        argv = self.rec.last
        self.assertEqual(argv[:5], ["build", "-t", "hermes:cctl-local-000000", "-t", "hermes:latest"])
        self.assertIn("cctl.image=hermes", argv)
        self.assertIn(f"cctl.recipe={self.image.recipe_hash()}", argv)
        self.assertEqual(argv[-1], "-")
        self.assertNotIn("-f", argv)
        self.assertEqual(self.rec.stdins[-1].name, str(self.dir / "hermes.Dockerfile"))

    def test_build_with_context_sends_the_directory(self):
        (self.dir / "hermes").mkdir()
        with redirect_stdout(io.StringIO()):
            self.assertTrue(Docker.build_image(self.image))
        argv = self.rec.last
        self.assertEqual(argv[-3:], ["-f", str(self.dir / "hermes.Dockerfile"), str(self.dir / "hermes")])
        self.assertIsNone(self.rec.stdins[-1])

    def rows(self, *pairs):
        """`docker images --format {{json .}}` lines for the hermes repository"""
        return "\n".join(json.dumps({"Repository": "hermes", "Tag": t, "ID": i, "Size": "1MB"}) for t, i in pairs)

    def test_build_numbers_after_the_highest_existing_version(self):
        self.rec.stdout = self.rows(("cctl-local-000007", "aaa"), ("latest", "aaa"), ("cctl-local-000002", "bbb"))
        with redirect_stdout(io.StringIO()):
            self.assertTrue(Docker.build_image(self.image))
        self.assertEqual(self.rec.calls[0], ["images", "--format", "{{json .}}"])
        self.assertEqual(self.rec.last[1:5], ["-t", "hermes:cctl-local-000008", "-t", "hermes:latest"])

    def test_local_versions_over_the_live_listing(self):
        rows = self.rows

        def versions():
            return containerctl.local_versions(self.image, Docker.list_images().logged())

        self.rec.stdout = rows(("cctl-local-000007", "aaa"), ("latest", "aaa"), ("cctl-local-000002", "bbb"))
        self.rec.stdout += "\n" + rows(("cctl-local-x", "ddd"), ("<none>", "ccc"))
        self.assertEqual(versions(), ([2, 7], 7, "aaa"))
        self.assertEqual(self.rec.last, ["images", "--format", "{{json .}}"])  # the listing everyone else uses
        self.rec.stdout = rows(("cctl-local-000002", "bbb"), ("latest", "zzz"))  # tag points at a non-local build
        self.assertEqual(versions(), ([2], None, "zzz"))
        self.rec.stdout = ""
        self.assertEqual(versions(), ([], None, None))
        self.rec.returncode = 1
        self.assertEqual(versions(), ([], None, None))  # a failed listing is logged there and reads as nothing built
        self.assertEqual(get_errors()[-1].cat, "docker.command")

    def test_version_tag_keeps_registry_and_port(self):
        image = containerctl.ImageConfig(
            name="h", dockerfile=self.dir / "x", context=self.dir / "y", tag="reg:5000/h:1"
        )
        self.assertEqual((image.repository, image.version_tag(23)), ("reg:5000/h", "reg:5000/h:cctl-local-000023"))

    def test_build_failure_is_logged(self):
        listing = containerctl.Listing()  # the listing answers, the build doesn't
        self.rec.returncode = 1
        with mock.patch.object(Docker, "list_images", return_value=listing):
            self.assertFalse(Docker.build_image(self.image))
        self.assertEqual(get_errors()[-1].cat, "image.build")

    def test_build_does_not_number_from_a_failed_listing(self):
        self.rec.returncode, self.rec.stderr = 1, "Cannot connect to the Docker daemon"
        with self.assertRaises(containerctl.DockerError) as raised:
            Docker.build_image(self.image)
        self.assertEqual(str(raised.exception), "Docker daemon is not running")
        self.assertEqual(self.rec.calls, [["images", "--format", "{{json .}}"]])  # no build as cctl-local-000000

    def test_ensure_image_skips_when_present(self):
        self.assertTrue(Docker.ensure_image("hermes:latest"))
        self.assertEqual(self.rec.calls, [["image", "inspect", "hermes:latest"]])

    def test_ensure_image_builds_when_missing_and_recipe_claims_it(self):
        self.rec.returncode = 1  # inspect fails: absent
        calls = []

        def build(image):
            calls.append(image.name)
            return True

        with mock.patch.object(Docker, "build_image", side_effect=build), redirect_stdout(io.StringIO()):
            self.assertTrue(Docker.ensure_image("hermes"))
            self.assertTrue(Docker.ensure_image("ubuntu:24.04"))  # no recipe: docker will pull
        self.assertEqual(calls, ["hermes"])

    def test_create_containers_ensure_the_image_first(self):
        self.rec.returncode = 1  # every docker call fails, including `image inspect`
        with mock.patch.object(Docker, "build_image", return_value=False), redirect_stdout(io.StringIO()):
            self.assertFalse(Docker.create_project_container(ProjectConfig(name="p", image="hermes")))
            self.assertFalse(Docker.create_service_container(ServiceConfig(name="s", image="hermes")))
        self.assertNotIn("run", [c[0] for c in self.rec.calls])
        self.assertLessEqual({"project.create", "service.create"}, {e.cat for e in get_errors()})

"""
The door to the daemon (``Docker._api``, ``Docker._socket``), against a daemon of ours: a real HTTP server on a
real unix socket, answering what each test tells it to. So the whole way is run for real (the connection, HTTP, the
errors the OS gives), and only docker is played: what the real one answers is tests/test_real_docker.py's to say.
"""

import http.server
import json
import os
import socket
import socketserver
import tempfile
import threading
import time
import unittest
from unittest import mock

import containerctl
from containerctl import Docker
from tests.helper import RunRecorder, reset_errors

CONTAINER = {"Id": "abc123", "Name": "/web", "Created": "", "State": {"Running": True}, "Image": "i", "Config": {}}


class FakeDaemon:
    """`answers` maps a path to (status, body); a body that is a dict goes out as JSON, bytes go out raw with no
    HTTP around them, a number is how long to sleep before a 200, a tuple (seconds, count) is a 200 whose body comes
    one byte at a time, that long apart (with a third that is False: no Content-Length, the body ends when the
    connection does). Any other path is a 404, as docker says it."""

    def __init__(self, directory: str, answers: dict):
        self.path, self.answers, self.asked = os.path.join(directory, "docker.sock"), answers, []
        daemon = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                daemon.asked.append(self.path)
                status, body = daemon.answers.get(self.path, (404, {"message": f"No such thing: {self.path}"}))
                if isinstance(body, bytes):
                    self.wfile.write(body)
                    return
                if isinstance(body, tuple):
                    return self.dribble(*body)
                if isinstance(body, (int, float)):
                    time.sleep(body)
                    body = {}
                payload = (json.dumps(body) if isinstance(body, (dict, list)) else body).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except BrokenPipeError:  # whoever asked stopped waiting: the test of just that
                    pass

            def dribble(self, seconds, count, length=True):
                try:
                    self.send_response(200)
                    if length:
                        self.send_header("Content-Length", str(count))
                    self.end_headers()
                    for _ in range(count):
                        self.wfile.write(b" ")
                        self.wfile.flush()
                        time.sleep(seconds)
                except OSError:  # whoever asked stopped waiting
                    pass

            def log_message(self, *args):
                pass

        class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = True
            block_on_close = False  # a handler still sleeping is not waited for

        self.server = Server(self.path, Handler)
        threading.Thread(target=self.server.serve_forever, args=(0.01,), daemon=True).start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


class ApiBase(unittest.TestCase):
    ANSWERS: dict = {}

    def setUp(self):
        reset_errors()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.daemon = FakeDaemon(self.dir, dict(self.ANSWERS))
        self.addCleanup(self.daemon.stop)
        self.host(f"unix://{self.daemon.path}")

    def host(self, value: str | None) -> None:
        """Point DOCKER_HOST there for the rest of the test (None: unset, the CLI is then asked)"""
        env = {k: v for k, v in os.environ.items() if k != "DOCKER_HOST"} | ({"DOCKER_HOST": value} if value else {})
        for patch in (mock.patch.dict(os.environ, env, clear=True), mock.patch.object(Docker, "_endpoint", None)):
            patch.start()
            self.addCleanup(patch.stop)

    def failure_of(self, ask) -> containerctl.Failure:
        with self.assertRaises(containerctl.DockerError) as raised:
            ask()
        return raised.exception.failure


class AnswersTests(ApiBase):
    ANSWERS = {
        "/containers/web/json": (200, CONTAINER),
        "/networks/proxy": (200, {"Id": "n1", "Name": "proxy", "Created": "", "Scope": "local", "Driver": "bridge"}),
        "/images/reg.example:5000/a/b:1/json": (200, {"Id": "sha256:1"}),
        "/images/broken:1/json": (500, {"message": "layer does not exist: no such file or directory"}),
        "/volumes/odd": (200, "<html>not docker</html>"),
    }

    def test_what_is_there_comes_back_as_its_model(self):
        info = Docker.inspect(containerctl.DockerContainerInspect, "web")
        self.assertEqual((info.Id, info.name_clean, info.State), ("abc123", "web", {"Running": True}))
        self.assertEqual(Docker.inspect(containerctl.DockerNetworkInspect, "proxy").Name, "proxy")

    def test_what_is_not_there_is_none_for_each_kind(self):
        for kind in (
            containerctl.DockerContainerInspect,
            containerctl.DockerImageInspect,
            containerctl.DockerVolumeInspect,
            containerctl.DockerNetworkInspect,
        ):
            self.assertIsNone(Docker.inspect(kind, "nope"))
        self.assertEqual(
            self.daemon.asked, ["/containers/nope/json", "/images/nope/json", "/volumes/nope", "/networks/nope"]
        )
        self.assertEqual(containerctl.get_errors(), [])  # an answer, not an error

    def test_a_name_is_one_kind_of_thing(self):
        self.assertIsNone(
            Docker.inspect(containerctl.DockerContainerInspect, "proxy")
        )  # the network of that name is not it
        self.assertIsNone(Docker.inspect(containerctl.DockerNetworkInspect, "web"))

    def test_an_image_ref_goes_through_whole(self):
        self.assertTrue(Docker.image_exists("reg.example:5000/a/b:1"))
        self.assertFalse(Docker.image_exists("reg.example:5000/a/b:2"))
        self.assertFalse(Docker.image_exists("a@sha256:" + "0123abcd" * 8))
        self.assertEqual(self.daemon.asked[-1], "/images/a@sha256:" + "0123abcd" * 8 + "/json")

    def test_a_daemon_that_failed_did_not_say_no(self):
        failure = self.failure_of(lambda: Docker.image_exists("broken:1"))
        self.assertEqual(failure.msg, "Unknown docker failure")  # its "no such file" is not "no such image"
        self.assertIn("HTTP 500", failure.detail)
        self.assertIn("layer does not exist", failure.detail)

    def test_an_answer_that_is_not_dockers_is_not_one(self):
        self.assertIn(
            "HTTP 200: <html>", self.failure_of(lambda: Docker.inspect(containerctl.DockerVolumeInspect, "odd")).detail
        )

    def test_ensure_network_creates_only_what_the_daemon_says_is_not_there(self):
        rec = RunRecorder()
        with mock.patch.object(Docker, "_run", rec):
            self.assertTrue(Docker.ensure_network("proxy"))
            self.assertEqual(rec.calls, [])
            self.assertTrue(Docker.ensure_network("backend"))
            self.assertEqual(rec.calls, [["network", "create", "backend"]])

    def test_every_question_is_its_own_connection(self):
        for _ in range(3):
            Docker.image_exists("reg.example:5000/a/b:1")
        self.assertEqual(len(self.daemon.asked), 3)


class NoAnswerTests(ApiBase):
    """What the OS says of the connection is what we say of docker: no wording of anybody's is read"""

    ANSWERS = {
        "/images/slow:1/json": (200, 1),
        "/images/dribbled:1/json": (200, (0.05, 40)),  # 2s in all, never 0.2s without a byte
        "/images/endless:1/json": (200, (0.05, 40, False)),  # the same, and nothing says how long it will be
        "/images/garbage:1/json": (200, b"\x00\x01 not http\r\n\r\n"),
    }

    def ask(self):
        return Docker.image_exists("a:b")

    def test_no_socket_there(self):
        self.host(f"unix://{self.dir}/nothing.sock")
        failure = self.failure_of(self.ask)
        self.assertEqual((failure.msg, failure.hint), containerctl.DAEMON_DOWN)
        self.assertIn(f"GET /images/a:b/json on {self.dir}/nothing.sock", failure.detail)

    def test_a_socket_nobody_is_behind(self):
        dead = os.path.join(self.dir, "dead.sock")
        left = socket.socket(socket.AF_UNIX)
        left.bind(dead)
        left.close()  # the file stays, as after a daemon that was killed
        self.host(f"unix://{dead}")
        self.assertEqual(self.failure_of(self.ask).msg, "Docker daemon is not running")

    @unittest.skipIf(os.geteuid() == 0, "root is never denied")
    def test_a_socket_we_may_not_use(self):
        os.chmod(self.daemon.path, 0)
        self.assertEqual(self.failure_of(self.ask).msg, "No permission on the docker socket")

    def test_a_daemon_that_does_not_answer_in_time(self):
        started = time.monotonic()
        failure = self.failure_of(lambda: Docker._api("/images/slow:1/json", timeout=0.2))
        self.assertEqual(failure.msg, "Docker did not answer in time")
        self.assertLess(time.monotonic() - started, 2)

    def test_an_answer_that_comes_a_byte_at_a_time_is_not_waited_for_either(self):
        # The timeout is for the whole answer: a socket's own starts over at every byte that comes
        started = time.monotonic()
        failure = self.failure_of(lambda: Docker._api("/images/dribbled:1/json", timeout=0.2))
        self.assertEqual(failure.msg, "Docker did not answer in time")
        self.assertLess(time.monotonic() - started, 1)

    def test_an_answer_cut_short_is_not_an_answer_when_nothing_said_how_long_it_was(self):
        # No Content-Length: the body is "all there is until the connection ends", and our hanging up ends it.
        # What came until then is not what docker said
        failure = self.failure_of(lambda: Docker._api("/images/endless:1/json", timeout=0.2))
        self.assertEqual(failure.msg, "Docker did not answer in time")

    def test_something_that_does_not_speak_http(self):
        self.assertEqual(self.failure_of(lambda: Docker.image_exists("garbage:1")).msg, "Unknown docker failure")

    def test_none_of_them_is_a_no(self):
        # The point of it all: with nobody to ask, nothing gets created, pulled or built over what may be there
        self.host(f"unix://{self.dir}/nothing.sock")
        rec = RunRecorder()
        with mock.patch.object(Docker, "_run", rec):
            for ask in (
                lambda: containerctl.ensure_image("ubuntu:24.04"),
                lambda: Docker.ensure_network("proxy"),
                lambda: Docker.inspect(containerctl.DockerContainerInspect, "web"),
            ):
                self.failure_of(ask)
        self.assertEqual(rec.calls, [])


class WhereTheDaemonIsTests(ApiBase):
    """The CLI and we must ask the same daemon: DOCKER_HOST as the CLI reads it, else where the CLI says it goes"""

    CONTEXT = ["context", "inspect", "--format", "{{.Endpoints.docker.Host}}"]

    def test_without_docker_host_the_cli_is_asked_once(self):
        self.host(None)
        rec = RunRecorder(stdout=f"unix://{self.daemon.path}\n")
        with mock.patch.object(Docker, "_run", rec):
            self.assertIsNone(Docker.inspect(containerctl.DockerContainerInspect, "a"))
            self.assertIsNone(Docker.inspect(containerctl.DockerContainerInspect, "b"))
        self.assertEqual(rec.calls, [self.CONTEXT])
        self.assertEqual(self.daemon.asked, ["/containers/a/json", "/containers/b/json"])

    def test_docker_host_wins_and_is_read_every_time(self):
        with mock.patch.object(Docker, "_run", RunRecorder()) as rec:
            self.assertIsNone(Docker.inspect(containerctl.DockerContainerInspect, "a"))
            os.environ["DOCKER_HOST"] = f"unix://{self.dir}/moved.sock"
            self.assertEqual(self.failure_of(lambda: Docker.image_exists("a:b")).msg, "Docker daemon is not running")
        self.assertEqual(rec.calls, [])

    def test_a_cli_that_cannot_say_where_is_a_failure_and_is_asked_again(self):
        self.host(None)
        rec = RunRecorder(returncode=127, stderr="docker: command not found on PATH")
        with mock.patch.object(Docker, "_run", rec):
            self.assertEqual(self.failure_of(lambda: Docker.image_exists("a:b")).msg, "docker is not installed")
            rec.returncode, rec.stdout = 0, f"unix://{self.daemon.path}"
            self.assertFalse(Docker.image_exists("a:b"))
        self.assertEqual(rec.calls, [self.CONTEXT] * 2)

    def test_a_daemon_that_is_not_on_a_unix_socket_is_said_not_guessed_around(self):
        for endpoint in ("tcp://10.0.0.1:2376", "ssh://host"):
            self.host(endpoint)
            failure = self.failure_of(lambda: Docker.image_exists("a:b"))
            self.assertEqual(failure.msg, "Docker is not on a unix socket")
            self.assertIn(endpoint, failure.detail)
        self.assertEqual(self.daemon.asked, [])


if __name__ == "__main__":
    unittest.main()

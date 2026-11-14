"""
The real docker, not a stand-in: the places where we go by what docker *answers* (the engine API's 200 and 404 on
its socket, the JSON of an inspect and of the listings, the CLI's words for a daemon that is not there), which no
fake can vouch for. Every other test file plays docker and so pins our side only.

Skipped without docker, like the rest of the collection's integration suites. With CCTL_REAL_DOCKER=1 (CI) a missing
docker is a failure instead: there, a skip would be a check that silently stopped checking.

Nothing is pulled and no container is run: an inspect of what isn't there, one of each kind made (an image built
`FROM scratch`, a container of it created and never started, a volume, a network), found and removed, the four
listings, and a daemon that isn't there (DOCKER_HOST pointed at no socket: needs the client only).
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import containerctl
from containerctl import Docker
from tests.helper import reset_errors

REQUIRED = os.environ.get("CCTL_REAL_DOCKER") == "1"
MISSING = f"cctl-test-{os.getpid()}-missing"  # a name nothing has
NO_DAEMON = {"DOCKER_HOST": "unix:///nonexistent/cctl-test.sock"}


def need(what: str, there: bool) -> None:
    if not there:
        if REQUIRED:
            raise AssertionError(f"CCTL_REAL_DOCKER=1 but {what}")
        raise unittest.SkipTest(what)


def need_client() -> None:
    need("no docker on PATH", shutil.which("docker") is not None)


def need_daemon() -> None:
    need_client()
    answer = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"], capture_output=True, timeout=30)
    need("the docker daemon does not answer", answer.returncode == 0)


class WhatIsNotThereTests(unittest.TestCase):
    """Docker._object reads "there is no such thing" off the engine API's 404: is that still how it says it?"""

    @classmethod
    def setUpClass(cls):
        need_daemon()

    def setUp(self):
        reset_errors()

    def test_an_inspect_of_nothing_is_none_for_each_kind(self):
        for kind in (
            containerctl.DockerContainerInspect,
            containerctl.DockerImageInspect,
            containerctl.DockerVolumeInspect,
            containerctl.DockerNetworkInspect,
        ):
            with self.subTest(kind=kind.TYPE):  # DockerError here: the API no longer answers 404 at kind.API
                self.assertIsNone(Docker.inspect(kind, MISSING))

    def test_an_image_that_is_not_there_does_not_exist(self):
        self.assertFalse(Docker.image_exists(f"{MISSING}:nope"))

    def test_a_network_is_made_once_found_then_gone(self):
        name = f"cctl-test-{os.getpid()}-net"
        self.addCleanup(subprocess.run, ["docker", "network", "rm", name], capture_output=True)
        self.assertTrue(Docker.ensure_network(name))  # absent: created
        self.assertTrue(Docker.ensure_network(name))  # there: left alone ("already exists" would be a False)
        self.assertEqual(Docker.inspect(containerctl.DockerNetworkInspect, name).Name, name)
        self.assertIn(name, [network.Name for network in Docker.list_networks().required()])
        self.assertTrue(Docker.remove_network(name))
        self.assertIsNone(Docker.inspect(containerctl.DockerNetworkInspect, name))


class WhatIsThereTests(unittest.TestCase):
    """The other half: a 404 is only "no such thing" if what is there answers 200 at the same path. A path the
    daemon no longer serves is a 404 too, and would pass every test above."""

    PREFIX = f"cctl-test-{os.getpid()}"

    @classmethod
    def setUpClass(cls):
        need_daemon()

    def setUp(self):
        reset_errors()

    def docker(self, *args: str) -> None:
        """A docker command that must work: what is asked about has to be there for the question to mean anything"""
        done = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, f"docker {' '.join(args)}: {done.stderr}")

    def later(self, *args: str) -> None:
        self.addCleanup(subprocess.run, ["docker", *args], capture_output=True, timeout=120)

    def test_a_volume_that_is_there_is_found_then_gone(self):
        name = f"{self.PREFIX}-vol"
        self.later("volume", "rm", name)
        self.docker("volume", "create", name)
        self.assertEqual(Docker.inspect(containerctl.DockerVolumeInspect, name).Name, name)
        self.assertTrue(Docker.remove_volume(name))
        self.assertIsNone(Docker.inspect(containerctl.DockerVolumeInspect, name))
        self.assertEqual(containerctl.get_errors(), [])

    def test_an_image_and_a_container_of_it_are_found_then_gone(self):
        tag, name = f"{self.PREFIX}-img:probe", f"{self.PREFIX}-ctr"
        self.later("rmi", "-f", tag)
        self.later("rm", "-f", name)  # cleanups run last first: the container goes before its image
        with tempfile.TemporaryDirectory() as context:  # FROM scratch: nothing to pull
            with open(os.path.join(context, "Dockerfile"), "w") as f:
                f.write("FROM scratch\nCOPY Dockerfile /Dockerfile\n")
            self.docker("build", "-q", "-t", tag, context)
        self.assertTrue(Docker.image_exists(tag))
        image = Docker.inspect(containerctl.DockerImageInspect, tag)
        self.assertIn(tag, image.RepoTags)

        self.docker("create", "--name", name, tag, "/nothing")  # created, never started: there is nothing to run
        info = Docker.inspect(containerctl.DockerContainerInspect, name)
        self.assertEqual(info.name_clean, name)
        self.assertEqual(Docker.inspect(containerctl.DockerContainerInspect, info.Id).Id, info.Id)  # by ID as by name
        self.assertIsNone(Docker.inspect(containerctl.DockerNetworkInspect, name))  # a name is one kind of thing

        self.assertTrue(Docker.remove_container(name))
        self.assertIsNone(Docker.inspect(containerctl.DockerContainerInspect, name))
        self.assertTrue(Docker.remove_image(tag))
        self.assertFalse(Docker.image_exists(tag))
        self.assertEqual(containerctl.get_errors(), [])  # every answer was one we could read


class ListingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        need_daemon()

    def test_the_four_listings_come_back_and_every_row_is_one_we_can_read(self):
        for key, listing in Docker.listings(timeout=30).items():
            with self.subTest(listing=key):
                self.assertIsNone(listing.failure)
                self.assertEqual(listing.skipped, [])


class NoDaemonTests(unittest.TestCase):
    """A daemon that isn't there is never "no such image", by the socket (the questions) or by the CLI's words
    for it (a listing), which must be ones we know"""

    @classmethod
    def setUpClass(cls):
        need_client()

    def setUp(self):
        reset_errors()
        patch = mock.patch.dict(os.environ, NO_DAEMON)
        patch.start()
        self.addCleanup(patch.stop)

    def test_a_listing_says_the_daemon_is_not_running(self):
        failure = Docker.list_containers().failure
        self.assertIsNotNone(failure)
        self.assertEqual(failure.msg, "Docker daemon is not running", failure.detail)

    def test_does_it_exist_is_not_answered_by_no(self):
        for ask in (
            lambda: Docker.image_exists(f"{MISSING}:nope"),
            lambda: Docker.ensure_network(MISSING),
            lambda: Docker.inspect(containerctl.DockerContainerInspect, MISSING),
        ):
            with self.assertRaises(containerctl.DockerError) as raised:
                ask()
            self.assertEqual(str(raised.exception), "Docker daemon is not running", raised.exception.failure.detail)


if __name__ == "__main__":
    unittest.main()

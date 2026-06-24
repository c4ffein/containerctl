"""Tests for the JSON -> dataclass parsers (Docker `--format {{json .}}` output)."""

import unittest

from containerctl import (
    Container,
    ContainerInspect,
    Image,
    ImageInspect,
    Network,
    NetworkInspect,
    Volume,
    VolumeInspect,
    get_errors,
)
from tests.helper import reset_errors


class ContainerModelTests(unittest.TestCase):
    def setUp(self):
        reset_errors()

    def _valid(self):
        return {"ID": "abc123", "Names": "web", "Image": "nginx", "Status": "Up 2h"}

    def test_from_dict_minimal(self):
        c = Container.from_dict(self._valid())
        self.assertIsNotNone(c)
        self.assertEqual(c.ID, "abc123")
        self.assertEqual(c.Names, "web")

    def test_ignores_unknown_fields(self):
        data = self._valid()
        data["SomeFutureDockerField"] = "whatever"
        c = Container.from_dict(data)
        self.assertIsNotNone(c)

    def test_missing_required_returns_none_and_logs(self):
        data = self._valid()
        del data["Image"]
        self.assertIsNone(Container.from_dict(data))
        errs = get_errors()
        self.assertEqual(len(errs), 1)
        self.assertEqual(errs[0]["cat"], "docker.container_parse")

    def test_invalid_id_returns_none_and_logs(self):
        data = self._valid()
        data["ID"] = "abc;rm -rf /"
        self.assertIsNone(Container.from_dict(data))
        self.assertEqual(get_errors()[0]["cat"], "docker.container_parse")

    def test_to_dict_drops_none_platform(self):
        c = Container.from_dict(self._valid())
        self.assertNotIn("Platform", c.to_dict())

    def test_to_dict_keeps_set_platform(self):
        data = self._valid()
        data["Platform"] = "linux/amd64"
        c = Container.from_dict(data)
        self.assertEqual(c.to_dict()["Platform"], "linux/amd64")


class OtherListModelTests(unittest.TestCase):
    def setUp(self):
        reset_errors()

    def test_image_requires_id_repo_tag(self):
        self.assertIsNotNone(Image.from_dict({"ID": "img1", "Repository": "r", "Tag": "t", "Size": "1MB"}))
        self.assertIsNone(Image.from_dict({"Repository": "r", "Tag": "t"}))

    def test_volume_requires_name_driver(self):
        self.assertIsNotNone(Volume.from_dict({"Name": "vol1", "Driver": "local"}))
        self.assertIsNone(Volume.from_dict({"Name": "vol1"}))

    def test_volume_invalid_name_logs(self):
        self.assertIsNone(Volume.from_dict({"Name": "bad name", "Driver": "local"}))
        self.assertEqual(get_errors()[0]["cat"], "docker.volume_parse")

    def test_network_requires_id_name_driver(self):
        self.assertIsNotNone(Network.from_dict({"ID": "net1", "Name": "bridge", "Driver": "bridge"}))
        self.assertIsNone(Network.from_dict({"ID": "net1", "Name": "bridge"}))


class InspectModelTests(unittest.TestCase):
    def setUp(self):
        reset_errors()

    def test_container_inspect_properties(self):
        data = {
            "Id": "abc123",
            "Name": "/web",
            "Created": "2026-01-01",
            "State": {"Running": True},
            "Image": "sha256",
            "Config": {"Labels": {"com.example": "1"}},
        }
        ci = ContainerInspect.from_dict(data)
        self.assertIsNotNone(ci)
        self.assertEqual(ci.name_clean, "web")
        self.assertEqual(ci.labels, {"com.example": "1"})

    def test_container_inspect_labels_default_empty(self):
        ci = ContainerInspect.from_dict(
            {"Id": "abc", "Name": "/x", "Created": "t", "State": {}, "Image": "i", "Config": {}},
        )
        self.assertEqual(ci.labels, {})

    def test_container_inspect_missing_required(self):
        self.assertIsNone(ContainerInspect.from_dict({"Id": "abc"}))

    def test_image_inspect_allows_sha_id(self):
        # Image refs may contain ':' (e.g. sha256:...), unlike plain ids.
        ii = ImageInspect.from_dict(
            {"Id": "sha256:deadbeef", "Created": "t", "Architecture": "amd64", "Os": "linux", "Size": 10},
        )
        self.assertIsNotNone(ii)

    def test_volume_inspect_requires_mountpoint(self):
        self.assertIsNone(VolumeInspect.from_dict({"Name": "v", "Driver": "local"}))
        ok = VolumeInspect.from_dict({"Name": "v", "Driver": "local", "Mountpoint": "/m", "Scope": "local"})
        self.assertIsNotNone(ok)

    def test_network_inspect_minimal(self):
        ni = NetworkInspect.from_dict(
            {"Id": "n1", "Name": "bridge", "Created": "t", "Scope": "local", "Driver": "bridge"},
        )
        self.assertIsNotNone(ni)
        self.assertTrue(ni.EnableIPv4)
        self.assertFalse(ni.Internal)


if __name__ == "__main__":
    unittest.main()

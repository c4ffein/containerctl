"""Tests for the JSON -> dataclass parsers (Docker `--format {{json .}}` output)."""

import unittest

import containerctl
from containerctl import (
    DockerContainerInspect,
    DockerContainerRow,
    DockerImageInspect,
    DockerImageRow,
    DockerNetworkInspect,
    DockerNetworkRow,
    DockerVolumeInspect,
    DockerVolumeRow,
    get_errors,
)
from tests.helper import reset_errors


class ContainerModelTests(unittest.TestCase):
    def setUp(self):
        reset_errors()

    def _valid(self):
        return {"ID": "abc123", "Names": "web", "Image": "nginx", "Status": "Up 2h"}

    def test_from_dict_minimal(self):
        c = DockerContainerRow.from_dict(self._valid())
        self.assertIsNotNone(c)
        self.assertEqual(c.ID, "abc123")
        self.assertEqual(c.Names, "web")

    def test_ignores_unknown_fields(self):
        data = self._valid()
        data["SomeFutureDockerField"] = "whatever"
        c = DockerContainerRow.from_dict(data)
        self.assertIsNotNone(c)

    def test_missing_required_is_refused(self):
        data = self._valid()
        del data["Image"]
        self.assertRaises(ValueError, DockerContainerRow.from_dict, data)
        self.assertEqual(get_errors(), [])  # the listing decides what a skipped row becomes

    def test_invalid_id_is_refused(self):
        data = self._valid()
        data["ID"] = "abc;rm -rf /"
        self.assertRaises(ValueError, DockerContainerRow.from_dict, data)

    def test_project_is_read_off_the_cctl_name(self):
        self.assertEqual(DockerContainerRow.from_dict({**self._valid(), "Names": "cctl-prj-demo"}).Project, "demo")
        self.assertEqual(DockerContainerRow.from_dict({**self._valid(), "Names": "cctl-svc-web"}).Project, "")
        self.assertEqual(DockerContainerRow.from_dict(self._valid()).Project, "")


class OtherListModelTests(unittest.TestCase):
    def setUp(self):
        reset_errors()

    def test_image_requires_id_repo_tag(self):
        self.assertIsNotNone(DockerImageRow.from_dict({"ID": "img1", "Repository": "r", "Tag": "t", "Size": "1MB"}))
        self.assertRaises(ValueError, DockerImageRow.from_dict, {"Repository": "r", "Tag": "t"})

    def test_volume_requires_name_driver(self):
        self.assertIsNotNone(DockerVolumeRow.from_dict({"Name": "vol1", "Driver": "local"}))
        self.assertRaises(ValueError, DockerVolumeRow.from_dict, {"Name": "vol1"})

    def test_volume_invalid_name_logs(self):
        self.assertRaises(ValueError, DockerVolumeRow.from_dict, {"Name": "bad name", "Driver": "local"})

    def test_network_requires_id_name_driver(self):
        self.assertIsNotNone(DockerNetworkRow.from_dict({"ID": "net1", "Name": "bridge", "Driver": "bridge"}))
        self.assertRaises(ValueError, DockerNetworkRow.from_dict, {"ID": "net1", "Name": "bridge"})


class DeserializeTests(unittest.TestCase):
    """The dataclass is the schema: required = no default, types = the annotations, unknown keys dropped"""

    VALID = {"ID": "abc123", "Names": "web", "Image": "nginx", "Status": "Up 2h"}

    def refused(self, cls, data):
        with self.assertRaises(ValueError) as raised:
            cls.from_dict(data)
        return str(raised.exception)

    def test_only_a_json_object_is_a_row(self):
        for data in (5, None, "x", [1], True):
            self.assertEqual(self.refused(DockerContainerRow, data), f"Not an object: {type(data).__name__}")

    def test_required_fields_are_the_ones_without_a_default(self):
        self.assertEqual(self.refused(DockerContainerRow, {"Names": "x"}), "Missing fields: ['ID', 'Image', 'Status']")
        self.assertEqual(
            self.refused(DockerImageRow, {"ID": "a", "Repository": "r", "Tag": "t"}), "Missing fields: ['Size']"
        )

    def test_a_value_must_be_of_its_fields_type(self):
        self.assertEqual(
            self.refused(DockerContainerRow, {**self.VALID, "Names": None}), "Wrong types: Names (NoneType)"
        )
        self.assertEqual(
            self.refused(DockerContainerRow, {**self.VALID, "ID": 12, "State": ["up"]}),
            "Wrong types: ID (int), State (list)",
        )

    def test_a_union_takes_any_of_its_types(self):
        for platform in (None, "linux/amd64", {"os": "linux"}):
            self.assertEqual(DockerContainerRow.from_dict({**self.VALID, "Platform": platform}).Platform, platform)
        self.assertEqual(self.refused(DockerContainerRow, {**self.VALID, "Platform": 1}), "Wrong types: Platform (int)")

    def test_the_key_field_is_checked_per_class(self):
        self.assertEqual(self.refused(DockerContainerRow, {**self.VALID, "ID": "a;b"}), "Invalid ID: a;b")
        self.assertEqual(
            self.refused(DockerVolumeRow, {"Name": "bad name", "Driver": "local"}), "Invalid Name: bad name"
        )
        self.assertNotIn("KEY", DockerVolumeRow.__dataclass_fields__)


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
        ci = DockerContainerInspect.from_dict(data)
        self.assertIsNotNone(ci)
        self.assertEqual(ci.name_clean, "web")
        self.assertEqual(ci.labels, {"com.example": "1"})

    def test_container_inspect_labels_default_empty(self):
        ci = DockerContainerInspect.from_dict(
            {"Id": "abc", "Name": "/x", "Created": "t", "State": {}, "Image": "i", "Config": {}},
        )
        self.assertEqual(ci.labels, {})

    def test_container_inspect_missing_required(self):
        self.assertIsNone(DockerContainerInspect.from_dict({"Id": "abc"}))

    def test_image_inspect_allows_sha_id(self):
        # Image refs may contain ':' (e.g. sha256:...), unlike plain ids.
        ii = DockerImageInspect.from_dict(
            {"Id": "sha256:deadbeef", "Created": "t", "Architecture": "amd64", "Os": "linux", "Size": 10},
        )
        self.assertIsNotNone(ii)

    def test_volume_inspect_requires_mountpoint(self):
        self.assertIsNone(DockerVolumeInspect.from_dict({"Name": "v", "Driver": "local"}))
        ok = DockerVolumeInspect.from_dict({"Name": "v", "Driver": "local", "Mountpoint": "/m", "Scope": "local"})
        self.assertIsNotNone(ok)

    def test_inspect_checks_types_and_logs_what_is_wrong(self):
        data = {"Id": "n1", "Name": "bridge", "Created": "t", "Scope": "local", "Driver": 7}
        self.assertIsNone(DockerNetworkInspect.from_dict(data))
        (event,) = containerctl.get_errors()
        self.assertEqual((event.cat, event.msg), ("docker.network_inspect_parse", "Wrong types: Driver (int)"))

    def test_inspect_takes_null_where_docker_sends_it(self):
        data = {"Id": "n1", "Name": "b", "Created": "t", "Scope": "local", "Driver": "d", "Options": None}
        self.assertIsNotNone(DockerNetworkInspect.from_dict(data))

    def test_inspect_rejects_an_unsafe_key(self):
        data = {"Id": "a;b", "Name": "/x", "Created": "t", "State": {}, "Image": "i", "Config": {}}
        self.assertIsNone(DockerContainerInspect.from_dict(data))
        self.assertEqual(containerctl.get_errors()[0].msg, "Invalid Id: a;b")

    def test_network_inspect_minimal(self):
        ni = DockerNetworkInspect.from_dict(
            {"Id": "n1", "Name": "bridge", "Created": "t", "Scope": "local", "Driver": "bridge"},
        )
        self.assertIsNotNone(ni)
        self.assertTrue(ni.EnableIPv4)
        self.assertFalse(ni.Internal)


if __name__ == "__main__":
    unittest.main()

"""Is a pulled image behind its tag on its registry? What is asked (through the daemon), what is kept of the
answers, and what the rest reads of them. Docker is played here: tests/test_real_docker.py asks the real one."""

import argparse
import dataclasses
import io
import json
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import containerctl
from containerctl import Docker, DockerError, DockerImageRow, Failure, ProjectConfig, RegistryCheck, ServiceConfig
from tests.helper import ApiRecorder, listings, reset_errors

REMOTE = "sha256:" + "a" * 64
OLDER = "sha256:" + "b" * 64
LOCAL = "sha256:" + "1" * 64


def image(digests=(f"nginx@{REMOTE}",), id_=LOCAL) -> containerctl.DockerImageInspect:
    return containerctl.DockerImageInspect(
        Id=id_, Created="", Architecture="amd64", Os="linux", Size=1, RepoDigests=list(digests)
    )


class RegistryFileBase(unittest.TestCase):
    def setUp(self):
        reset_errors()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.file = Path(self._tmp.name) / "cache" / "registry.json"  # its directory is not there yet
        patch = mock.patch.object(containerctl, "REGISTRY_FILE", self.file)
        patch.start()
        self.addCleanup(patch.stop)


class RegistryDigestTests(unittest.TestCase):
    """Docker.registry_digest: the daemon's /distribution, read"""

    def setUp(self):
        reset_errors()

    def ask(self, status=200, body="{}", ref="nginx:1.27", **kwargs):
        self.api = ApiRecorder(status=status, body=body)
        with mock.patch.object(Docker, "_api", self.api):
            return Docker.registry_digest(ref, **kwargs)

    def test_the_digest_of_the_descriptor_is_the_answer(self):
        body = json.dumps({"Descriptor": {"mediaType": "application/vnd.oci.image.index.v1+json", "digest": REMOTE}})
        self.assertEqual(self.ask(body=body, ref="ghcr.io/you/app:1.0"), REMOTE)
        self.assertEqual(self.api.calls, ["/distribution/ghcr.io/you/app:1.0/json"])

    def test_the_network_is_in_it_so_the_timeout_is_its_own(self):
        seen = {}

        def api(path, timeout=None):
            seen["timeout"] = timeout
            return 200, json.dumps({"Descriptor": {"digest": REMOTE}})

        with mock.patch.object(Docker, "_api", api):
            Docker.registry_digest("nginx:1.27")
        self.assertEqual(seen["timeout"], containerctl.REGISTRY_TIMEOUT)

    def test_no_digest_is_never_an_answer(self):
        denied = json.dumps({"message": "pull access denied for you/private"})
        for status, body in (
            (401, denied),
            (404, '{"message": "manifest unknown"}'),
            (500, "not json"),
            (200, "{}"),
            (200, '{"Descriptor": {"digest": "not-a-digest"}}'),
            (200, '{"Descriptor": "sha256:abc"}'),
            (200, "[]"),
        ):
            with self.assertRaises(DockerError) as raised:
                self.ask(status, body)
            self.assertEqual(str(raised.exception), "The registry did not say")
            self.assertIn(f"HTTP {status}", raised.exception.failure.detail)
        with self.assertRaises(DockerError) as raised:
            self.ask(401, denied)
        self.assertIn("pull access denied for you/private", raised.exception.failure.detail)

    def test_what_is_not_a_ref_is_never_part_of_a_path(self):
        for ref in ("../containers/json", "-x", "nginx:1.27?all=1", ""):
            with self.assertRaises(DockerError):
                self.ask(ref=ref)
            self.assertEqual(self.api.calls, [])


class CheckRegistryTests(RegistryFileBase):
    def check(self, local, remote=REMOTE):
        registry = mock.Mock(side_effect=[remote] if not isinstance(remote, Exception) else remote)
        with (
            mock.patch.object(Docker, "inspect", return_value=local) as inspect,
            mock.patch.object(Docker, "registry_digest", registry),
        ):
            result = containerctl.check_registry("nginx:1.27")
        inspect.assert_called_once_with(containerctl.DockerImageInspect, "nginx:1.27")
        return result, registry

    def test_pulled_as_what_the_registry_has_now_is_current(self):
        result, registry = self.check(image())
        self.assertEqual((result.behind, result.status, result.local, result.remote), (False, "current", LOCAL, REMOTE))
        registry.assert_called_once_with("nginx:1.27")
        self.assertLess(abs(result.checked - time.time()), 5)

    def test_pulled_as_anything_else_is_behind(self):
        result, _ = self.check(image([f"nginx@{OLDER}"]))
        self.assertEqual((result.behind, result.status), (True, "behind"))
        result, _ = self.check(image([f"nginx@{OLDER}", f"mirror.local/nginx@{REMOTE}"]))  # the same image, twice
        self.assertFalse(result.behind)

    def test_what_is_not_here_or_was_built_here_is_not_asked_about(self):
        result, registry = self.check(None)
        self.assertEqual((result.behind, result.status), (None, "not pulled"))
        registry.assert_not_called()
        result, registry = self.check(image([]))
        self.assertEqual((result.behind, result.status), (None, "not from a registry"))
        registry.assert_not_called()

    def test_a_registry_that_does_not_say_is_a_note_and_a_line_of_the_log(self):
        silent = DockerError(
            Failure("The registry did not say", "is it public?", "GET /distribution/x: HTTP 401: denied")
        )
        result, _ = self.check(image(), silent)
        self.assertEqual((result.behind, result.status), (None, "unknown: The registry did not say"))
        self.assertEqual(
            [(e.cat, e.msg) for e in containerctl.get_errors()],
            [("image.registry", "nginx:1.27: GET /distribution/x: HTTP 401: denied")],
        )


class RegistryFileTests(RegistryFileBase):
    def test_what_is_written_is_read_back_and_nothing_is_left_beside_it(self):
        checks = {
            "nginx:1.27": RegistryCheck("nginx:1.27", 1700000000.5, LOCAL, REMOTE, True),
            "redis:7": RegistryCheck("redis:7", 1700000001.0, note="not pulled"),
        }
        containerctl.write_registry_checks(checks)
        self.assertEqual(containerctl.read_registry_checks(), checks)
        self.assertEqual([path.name for path in self.file.parent.iterdir()], ["registry.json"])
        containerctl.write_registry_checks({})  # over the one that is there
        self.assertEqual(containerctl.read_registry_checks(), {})

    def test_a_file_that_is_not_ours_is_an_empty_one(self):
        self.assertEqual(containerctl.read_registry_checks(), {})  # no file, no directory
        self.file.parent.mkdir()
        for content in ("", "not json", "{}", '[{"ref": "x"}]', '[{"ref": 1, "checked": 2.0}]', "[3]", '{"a": 1}'):
            self.file.write_text(content)
            self.assertEqual(containerctl.read_registry_checks(), {}, content)
        self.assertEqual(containerctl.get_errors(), [])

    def test_asking_keeps_the_answers_and_forgets_what_is_no_longer_asked_about(self):
        containerctl.write_registry_checks({"gone:1": RegistryCheck("gone:1", 1.0, LOCAL, REMOTE, True)})
        answers = {"nginx:1.27": RegistryCheck("nginx:1.27", 2.0, LOCAL, REMOTE, False)}
        with mock.patch.object(containerctl, "check_registry", side_effect=lambda ref: answers[ref]):
            self.assertEqual(containerctl.check_registries(["nginx:1.27"], {}), answers)
        self.assertEqual(containerctl.read_registry_checks(), answers)

    def test_an_interrupted_run_keeps_what_it_learned(self):
        before = {ref: RegistryCheck(ref, 1.0, LOCAL, REMOTE, True) for ref in ("a:1", "b:1")}
        containerctl.write_registry_checks(before)
        fresh = RegistryCheck("a:1", 2.0, LOCAL, REMOTE, False)
        with mock.patch.object(containerctl, "check_registry", side_effect=[fresh, KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                containerctl.check_registries(["a:1", "b:1"], {})
        self.assertEqual(containerctl.read_registry_checks(), {"a:1": fresh, "b:1": before["b:1"]})


class RegistryRefsAndIssuesTests(RegistryFileBase):
    def test_what_a_registry_can_be_asked_about(self):
        recipe = containerctl.ImageRecipeConfig(name="hermes", dockerfile=Path("h.Dockerfile"), context=Path("h"))
        projects = {
            "demo": ProjectConfig(name="demo", image="ubuntu:24.04"),
            "built": ProjectConfig(name="built", image="hermes"),  # a recipe's: asked about by its bases
            "pinned": ProjectConfig(name="pinned", image="ubuntu@sha256:" + "c" * 64),  # nothing to be behind
        }
        services = {
            "web": ServiceConfig(name="web", image="nginx"),
            "web2": ServiceConfig(name="web2", image="nginx:latest"),  # one ref, however it is written
            "tool": ServiceConfig(name="tool", image="ubuntu:24.04"),
        }
        self.assertEqual(
            containerctl.registry_refs(projects, services, {"hermes": recipe}),
            {
                "hermes:latest": ["recipe hermes", "project built"],  # a recipe is asked about, used or not
                "ubuntu:24.04": ["project demo", "service tool"],
                "nginx:latest": ["service web", "service web2"],
            },
        )

    def test_behind_is_an_issue_for_as_long_as_the_tag_is_on_that_image(self):
        refs = {"nginx:latest": ["service web", "project demo"], "redis:7": ["service cache"]}
        asked = time.time() - 3 * 86400 - 60
        checks = {
            "nginx:latest": RegistryCheck("nginx:latest", asked, LOCAL, REMOTE, True),
            "redis:7": RegistryCheck("redis:7", asked, LOCAL, REMOTE, False),
        }
        rows = [DockerImageRow("1" * 12, "nginx", "latest", "1MB"), DockerImageRow("1" * 12, "redis", "7", "1MB")]
        (issue,) = containerctl.registry_issues(refs, checks, rows, {})
        self.assertEqual((issue.level, issue.cat), (containerctl.WARNING, "image.behind"))
        self.assertEqual(issue.msg, "nginx:latest is behind its tag on the registry (asked 3d ago)")
        self.assertEqual(issue.hint, "docker pull nginx:latest, then recreate what runs it")
        self.assertEqual(issue.detail, "used by: service web, project demo")
        containerctl.registry_notes(rows, checks)
        self.assertEqual([row.Registry for row in rows], ["behind, 3d ago", "current, 3d ago"])

        pulled = [DockerImageRow("2" * 12, "nginx", "latest", "1MB")]  # since: another image, nothing known of it
        self.assertEqual(containerctl.registry_issues(refs, checks, pulled, {}), [])
        containerctl.registry_notes(pulled, checks)
        self.assertEqual(pulled[0].Registry, "")
        self.assertEqual(containerctl.registry_issues(refs, checks, [], {}), [])  # removed
        self.assertEqual(containerctl.registry_issues({}, checks, rows, {}), [])  # nothing uses it any more
        unknown = {"nginx:latest": RegistryCheck("nginx:latest", asked, note="unknown: The registry did not say")}
        self.assertEqual(containerctl.registry_issues(refs, unknown, rows, {}), [])

    def test_never_asked_or_not_for_a_while_is_one_issue_for_all_of_them(self):
        refs = {"nginx:latest": ["service web"], "redis:7": ["service cache"], "ubuntu:24.04": ["project demo"]}
        now = time.time()
        fresh = RegistryCheck("nginx:latest", now - containerctl.REGISTRY_STALE_AFTER + 60, LOCAL, REMOTE, True)
        old = RegistryCheck("redis:7", now - 20 * 86400, note="unknown: The registry did not say")
        self.assertIsNone(containerctl.stale_registry_issue({"nginx:latest": ["service web"]}, {"nginx:latest": fresh}))
        self.assertIsNone(containerctl.stale_registry_issue({}, {}))  # nothing pulled: nothing to ask
        issue = containerctl.stale_registry_issue(refs, {"nginx:latest": fresh, "redis:7": old})
        self.assertEqual((issue.level, issue.cat), (containerctl.WARNING, "image.unchecked"))
        self.assertEqual(issue.msg, "2 images not checked against the registry for over 2h")
        self.assertEqual(issue.hint, "containerctl image updates (or `u` on the Images tab)")
        self.assertEqual(issue.detail, "redis:7: asked 20d ago\nubuntu:24.04: never asked")
        issue = containerctl.stale_registry_issue({"redis:7": ["service cache"]}, {})
        self.assertEqual(issue.msg, "1 image not checked against the registry for over 2h")
        # Asked and not told is asked all the same: it is asked again when the answer would be old, not at once
        told_nothing = dataclasses.replace(old, checked=now)
        self.assertIsNone(containerctl.stale_registry_issue({"redis:7": ["service cache"]}, {"redis:7": told_nothing}))

    def test_how_long_ago(self):
        now = time.time()
        for seconds, said in ((0, "0s"), (59, "59s"), (60, "1m"), (3599, "59m"), (7200, "2h"), (86400 * 12, "12d")):
            self.assertEqual(containerctl.ago(now - seconds), said)
        self.assertEqual(containerctl.ago(now + 100), "0s")  # a clock that went back

    def test_the_issues_command_reads_the_file_and_asks_no_registry(self):
        containerctl.write_registry_checks(
            {"nginx:latest": RegistryCheck("nginx:latest", time.time(), LOCAL, REMOTE, True)}
        )
        rows = [DockerImageRow("1" * 12, "nginx", "latest", "1MB")]
        with (
            mock.patch.object(Docker, "listings", return_value=listings(images=rows)),
            mock.patch.object(containerctl, "read_projects", return_value=({}, [])),
            mock.patch.object(containerctl, "read_services", return_value=({"w": ServiceConfig("w", "nginx")}, [])),
            mock.patch.object(containerctl, "read_image_recipes", return_value=({}, [])),
            mock.patch.object(containerctl, "SECRETS_FILE", Path("/nonexistent/secrets.toml")),
            mock.patch.object(Docker, "registry_digest") as registry,
        ):
            issues = containerctl.current_issues()
        self.assertEqual([issue.cat for issue in issues], ["image.behind"])
        registry.assert_not_called()


class RecipeBasesTests(RegistryFileBase):
    """A recipe's build against the registries: by what it was built on (LABEL_BASES), since it has no tag there"""

    def setUp(self):
        super().setUp()
        self.recipe = containerctl.ImageRecipeConfig(
            name="dev", dockerfile=Path(self._tmp.name) / "dev.Dockerfile", context=Path("x")
        )
        self.recipe.dockerfile.write_text("FROM debian:12\nFROM node:22\n")

    def check(self, label, now):
        """`label` is what the build has (None: no such label, False: no such image); `now` what the registries say"""
        built = None
        if label is not False:
            built = image([], id_=LOCAL)
            built.Config = {"Labels": {} if label is None else {containerctl.LABEL_BASES: label}}

        def registry(ref):
            if isinstance(now[ref], Exception):
                raise now[ref]
            return now[ref]

        with (
            mock.patch.object(Docker, "inspect", return_value=built) as inspect,
            mock.patch.object(Docker, "registry_digest", side_effect=registry) as asked,
        ):
            result = containerctl.check_recipe(self.recipe)
        inspect.assert_called_once_with(containerctl.DockerImageInspect, "dev:latest")
        return result, [call.args[0] for call in asked.call_args_list]

    def test_built_on_what_the_registries_still_have_is_current(self):
        label = f"debian:12@{REMOTE} node:22@{OLDER}"
        result, asked = self.check(label, {"debian:12": REMOTE, "node:22": OLDER})
        self.assertEqual((result.behind, result.status, result.local), (False, "current", LOCAL))
        self.assertEqual(asked, ["debian:12", "node:22"])

    def test_a_base_replaced_since_the_build_is_behind_and_named(self):
        label = f"debian:12@{OLDER} node:22@{OLDER}"
        result, _ = self.check(label, {"debian:12": REMOTE, "node:22": OLDER})
        self.assertEqual((result.behind, result.status), (True, "built on an older debian:12"))
        result, _ = self.check(label, {"debian:12": REMOTE, "node:22": REMOTE})
        self.assertEqual(result.status, "built on an older debian:12, node:22")

    def test_what_keeps_us_from_knowing_is_said(self):
        self.assertEqual(self.check(False, {})[0].status, "not built")
        result, asked = self.check(None, {})  # built before the label, or by hand
        self.assertEqual((result.behind, result.status, asked), (None, "bases not recorded: rebuild once", []))
        self.recipe.dockerfile.write_text("FROM scratch\n")
        self.assertEqual(self.check(None, {})[0].status, "no base to check")
        self.assertEqual(self.check("", {})[0].status, "no base to check")  # recorded: there was none to
        silent = DockerError(Failure("The registry did not say", "", "GET /distribution/debian:12/json: HTTP 500"))
        result, _ = self.check(f"debian:12@{REMOTE}", {"debian:12": silent})
        self.assertEqual((result.behind, result.status), (None, "unknown: The registry did not say"))
        self.assertEqual(containerctl.get_errors()[-1].cat, "image.registry")

    def test_a_recipe_is_asked_about_as_one_and_its_issue_says_to_rebuild(self):
        recipes = {"dev": self.recipe}
        behind = RegistryCheck("dev:latest", time.time() - 3700, LOCAL, behind=True, note="built on an older debian:12")
        with (
            mock.patch.object(containerctl, "check_recipe", return_value=behind) as by_bases,
            mock.patch.object(
                containerctl, "check_registry", side_effect=lambda ref: RegistryCheck(ref, 1.0)
            ) as by_tag,
        ):
            checks = containerctl.check_registries(["dev:latest", "nginx:1.27"], recipes)
        by_bases.assert_called_once_with(self.recipe)
        by_tag.assert_called_once_with("nginx:1.27")
        refs = {"dev:latest": ["recipe dev", "project demo"]}
        rows = [DockerImageRow("1" * 12, "dev", "latest", "1MB")]
        (issue,) = containerctl.registry_issues(refs, checks, rows, recipes)
        self.assertEqual(issue.msg, "dev:latest was built on an older debian:12 (asked 1h ago)")
        self.assertEqual(issue.hint, "containerctl image build dev, then recreate what runs it")
        self.assertEqual(issue.detail, "used by: recipe dev, project demo")
        containerctl.registry_notes(rows, checks)
        self.assertEqual(rows[0].Registry, "behind, 1h ago")  # the column has no room for which base
        rebuilt = [DockerImageRow("2" * 12, "dev", "latest", "1MB")]
        self.assertEqual(containerctl.registry_issues(refs, checks, rebuilt, recipes), [])  # another image now


class CliImageUpdatesTests(RegistryFileBase):
    def run_cli(self, services, checks):
        with (
            mock.patch.object(containerctl, "load_projects", return_value={}),
            mock.patch.object(containerctl, "load_services", return_value=services),
            mock.patch.object(containerctl, "load_image_recipes", return_value={}),
            mock.patch.object(containerctl, "check_registry", side_effect=lambda ref: checks[ref]),
            redirect_stdout(io.StringIO()) as out,
        ):
            try:
                containerctl.cli_image_updates(argparse.Namespace())
                code = 0
            except SystemExit as e:
                code = e.code
        return code, out.getvalue()

    def test_nothing_to_ask(self):
        code, out = self.run_cli({}, {})
        self.assertEqual((code, out), (0, "No recipe, and no project or service: nothing to ask\n"))

    def test_every_image_its_answer_and_who_uses_it_exit_1_when_one_is_behind(self):
        services = {"web": ServiceConfig("web", "nginx:1.27"), "cache": ServiceConfig("cache", "redis:7")}
        checks = {
            "nginx:1.27": RegistryCheck("nginx:1.27", time.time(), LOCAL, REMOTE, True),
            "redis:7": RegistryCheck("redis:7", time.time(), LOCAL, REMOTE, False),
        }
        code, out = self.run_cli(services, checks)
        self.assertEqual(code, 1)
        lines = [line.split() for line in out.splitlines() if not line.startswith("-")]  # the rule under the headers
        self.assertEqual(
            lines[:3],
            [
                ["IMAGE", "REGISTRY", "USED", "BY"],
                ["nginx:1.27", "behind", "service", "web"],
                ["redis:7", "current", "service", "cache"],
            ],
        )
        self.assertIn("nginx:1.27: docker pull nginx:1.27, then recreate what runs it", out)
        self.assertEqual(sorted(containerctl.read_registry_checks()), ["nginx:1.27", "redis:7"])  # kept
        checks["nginx:1.27"] = dataclasses.replace(checks["nginx:1.27"], behind=None, note="not pulled")
        code, out = self.run_cli(services, checks)
        self.assertEqual(code, 0)  # not knowing is not being behind
        self.assertIn("not pulled", out)

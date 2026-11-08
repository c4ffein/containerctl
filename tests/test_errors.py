"""Tests for the in-process error registry used by the TUI 'errors' tab."""

import threading
import unittest
import unittest.mock

import containerctl
from containerctl import (
    ERROR,
    WARNING,
    Issue,
    clear_errors,
    collecting,
    fetch_issues,
    get_errors,
    get_errors_by_category,
    log_error,
)
from tests.helper import reset_errors


class ErrorRegistryTests(unittest.TestCase):
    def setUp(self):
        reset_errors()

    def test_log_and_get(self):
        log_error("cat.a", "boom", {"k": "v"})
        errors = get_errors()
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["cat"], "cat.a")
        self.assertEqual(errors[0]["msg"], "boom")
        self.assertEqual(errors[0]["ctx"], {"k": "v"})
        self.assertIn("ts", errors[0])

    def test_context_defaults_to_empty_dict(self):
        log_error("cat.a", "no ctx")
        self.assertEqual(get_errors()[0]["ctx"], {})

    def test_group_by_category(self):
        log_error("cat.a", "1")
        log_error("cat.b", "2")
        log_error("cat.a", "3")
        grouped = get_errors_by_category()
        self.assertEqual(set(grouped), {"cat.a", "cat.b"})
        self.assertEqual(len(grouped["cat.a"]), 2)

    def test_clear_returns_count_and_removes_only_category(self):
        log_error("cat.a", "1")
        log_error("cat.a", "2")
        log_error("cat.b", "3")
        removed = clear_errors("cat.a")
        self.assertEqual(removed, 2)
        self.assertEqual([e["cat"] for e in get_errors()], ["cat.b"])

    def test_fetch_issues_empty_state(self):
        rows = fetch_issues([])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Category"], "No issues")
        self.assertEqual(rows[0]["_category"], "")

    def test_fetch_issues_conditions_by_severity_then_events(self):
        log_error("docker.command", "docker ps failed")
        conditions = [
            Issue(WARNING, "image.versions", "hermes has 4 local versions", "docker rmi x"),
            Issue(ERROR, "project.config", "Failed to load p.toml"),
        ]
        rows = fetch_issues(conditions)
        self.assertEqual(
            [(r["Level"], r["Category"]) for r in rows],
            [(ERROR, "project.config"), (WARNING, "image.versions"), (ERROR, "docker.command")],
        )
        self.assertEqual(rows[1]["Message"], "hermes has 4 local versions  [docker rmi x]")
        self.assertEqual([r["Count"] for r in rows], ["", "", "1"])
        self.assertEqual([r["_category"] for r in rows], ["", "", "docker.command"])  # only events are clearable

    def test_every_occurrence_is_recorded_in_order(self):
        log_error("cat.a", "boom", {"first": 1})
        log_error("cat.a", "other")
        log_error("cat.a", "boom", {"second": 2})
        errors = get_errors()
        self.assertEqual([e["msg"] for e in errors], ["boom", "other", "boom"])
        self.assertEqual([e["ctx"] for e in errors if e["msg"] == "boom"], [{"first": 1}, {"second": 2}])
        self.assertLessEqual(errors[0]["ts"], errors[-1]["ts"])

    def test_registry_is_capped_keeping_the_newest(self):
        with unittest.mock.patch.object(containerctl, "ERROR_MAX_ENTRIES", 3):
            for i in range(5):
                log_error("cat.a", "boom", {"i": i})
        self.assertEqual([e["ctx"]["i"] for e in get_errors()], [2, 3, 4])

    def test_fetch_issues_counts_repeats_and_marks_latest(self):
        log_error("cat.a", "boom")
        log_error("cat.a", "boom")
        log_error("cat.a", "boom")
        log_error("cat.a", "other")
        log_error("cat.a", "other")
        rows = fetch_issues([])
        self.assertEqual(rows[0]["Count"], "5")  # every occurrence in the category
        self.assertEqual(rows[0]["Message"], "other (x2)")  # repeats of the latest message only

    def test_every_mutation_takes_the_registry_lock(self):
        # The TUI fetcher thread logs while the main thread clears/groups: all of them go through the lock.
        lock = unittest.mock.MagicMock()
        with unittest.mock.patch.object(containerctl, "_errors_lock", lock):
            log_error("cat.a", "1")
            get_errors_by_category()
            clear_errors("cat.a")
        self.assertEqual(lock.__enter__.call_count, 3)
        self.assertEqual(lock.__exit__.call_count, 3)

    def test_concurrent_loggers_and_clearers_keep_a_consistent_registry(self):
        def log(cat):
            for i in range(500):
                log_error(cat, "boom", {"i": i})

        def clear():
            for _ in range(200):
                clear_errors("cat.b")

        threads = [threading.Thread(target=log, args=("cat.a",)), threading.Thread(target=log, args=("cat.b",))]
        threads.append(threading.Thread(target=clear))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # cat.a is never cleared: all 500 land, in order; cat.b may or may not have survivors
        self.assertEqual([e["ctx"]["i"] for e in get_errors() if e["cat"] == "cat.a"], list(range(500)))
        self.assertEqual(clear_errors("cat.a"), 500)

    def test_clear_rebind_is_observable_through_module(self):
        # clear_errors() rebinds the module global; get_errors() must still
        # see the live list afterwards.
        log_error("cat.a", "1")
        clear_errors("cat.a")
        self.assertEqual(get_errors(), [])
        self.assertIs(get_errors(), containerctl._errors)


class CollectingTests(unittest.TestCase):
    """Inside collecting(), log_error yields error-level conditions instead of events: how the fetcher turns a
    config loader's complaints into something that shows while broken and vanishes when fixed."""

    def setUp(self):
        reset_errors()

    def test_inside_the_block_log_error_yields_conditions_not_events(self):
        with collecting() as issues:
            log_error("project.config", "Failed to load p.toml")
        self.assertEqual(issues, [Issue(ERROR, "project.config", "Failed to load p.toml")])
        self.assertEqual(get_errors(), [])

    def test_outside_the_block_events_are_back(self):
        with collecting():
            pass
        log_error("docker.command", "boom")
        self.assertEqual([e["cat"] for e in get_errors()], ["docker.command"])

    def test_blocks_nest_and_restore(self):
        with collecting() as outer:
            log_error("a", "1")
            with collecting() as inner:
                log_error("b", "2")
            log_error("c", "3")
        self.assertEqual([i.cat for i in outer], ["a", "c"])
        self.assertEqual([i.cat for i in inner], ["b"])

    def test_collection_is_per_thread(self):
        # The fetcher thread collects its config loads; an event logged by another thread meanwhile stays an event
        with collecting() as issues:
            t = threading.Thread(target=log_error, args=("docker.command", "boom"))
            t.start()
            t.join()
        self.assertEqual(issues, [])
        self.assertEqual([e["cat"] for e in get_errors()], ["docker.command"])


if __name__ == "__main__":
    unittest.main()

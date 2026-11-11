"""Tests for the in-process error log behind the TUI's Errors tab."""

import collections
import threading
import time
import unittest
import unittest.mock

import containerctl
from containerctl import Event, Issue, get_errors, log_error
from tests.helper import reset_errors


class ErrorLogTests(unittest.TestCase):
    def setUp(self):
        reset_errors()

    def test_log_and_get(self):
        log_error("cat.a", "boom", {"k": "v"})
        errors = get_errors()
        self.assertEqual(len(errors), 1)
        self.assertEqual((errors[0].cat, errors[0].msg, errors[0].ctx), ("cat.a", "boom", {"k": "v"}))
        self.assertGreater(errors[0].ts, 0)

    def test_context_defaults_to_empty_dict(self):
        log_error("cat.a", "no ctx")
        self.assertEqual(get_errors()[0].ctx, {})

    def test_every_occurrence_is_recorded_in_order(self):
        log_error("cat.a", "boom", {"first": 1})
        log_error("cat.a", "other")
        log_error("cat.a", "boom", {"second": 2})
        errors = get_errors()
        self.assertEqual([e.msg for e in errors], ["boom", "other", "boom"])
        self.assertEqual([e.ctx for e in errors if e.msg == "boom"], [{"first": 1}, {"second": 2}])
        self.assertLessEqual(errors[0].ts, errors[-1].ts)

    def test_the_log_is_a_ring_keeping_the_newest(self):
        with unittest.mock.patch.object(containerctl, "_errors", collections.deque(maxlen=3)):
            for i in range(5):
                log_error("cat.a", "boom", {"i": i})
            self.assertEqual([e.ctx["i"] for e in get_errors()], [2, 3, 4])
        self.assertEqual(containerctl._errors.maxlen, containerctl.EVENTS_MAX)  # the real one

    def test_get_returns_a_snapshot(self):
        log_error("cat.a", "1")
        snapshot = get_errors()
        log_error("cat.a", "2")
        self.assertEqual(len(snapshot), 1)

    def test_every_access_takes_the_lock(self):
        # The TUI fetcher thread logs while the main thread lists: both go through the lock.
        lock = unittest.mock.MagicMock()
        with unittest.mock.patch.object(containerctl, "_errors_lock", lock):
            log_error("cat.a", "1")
            get_errors()
        self.assertEqual(lock.__enter__.call_count, 2)
        self.assertEqual(lock.__exit__.call_count, 2)

    def test_concurrent_loggers_and_readers_keep_a_consistent_log(self):
        def log(cat):
            for i in range(500):
                log_error(cat, "boom", {"i": i})

        def read():
            for _ in range(200):
                for e in get_errors():
                    self.assertIn(e.cat, ("cat.a", "cat.b"))

        threads = [threading.Thread(target=log, args=("cat.a",)), threading.Thread(target=log, args=("cat.b",))]
        threads.append(threading.Thread(target=read))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for cat in ("cat.a", "cat.b"):
            self.assertEqual([e.ctx["i"] for e in get_errors() if e.cat == cat], list(range(500)), cat)

    def test_event_time_is_wall_clock(self):
        self.assertRegex(Event(time.time(), "c", "m", {}).time, r"^\d\d:\d\d:\d\d$")

    def test_clear_empties_the_log_and_counts_what_went(self):
        log_error("a", "1")
        log_error("b", "2")
        self.assertEqual(containerctl.clear_errors(), 2)
        self.assertEqual(get_errors(), [])
        self.assertEqual(containerctl.clear_errors(), 0)

    def test_issues_sort_errors_first_then_by_category(self):
        issues = [
            Issue(containerctl.WARNING, "b", "1"),
            Issue(containerctl.ERROR, "z", "2"),
            Issue(containerctl.WARNING, "a", "3"),
            Issue(containerctl.ERROR, "c", "4"),
        ]
        self.assertEqual([i.msg for i in containerctl.by_severity(issues)], ["4", "2", "3", "1"])

    def test_log_issues_is_the_one_shot_callers_sink(self):
        containerctl.log_issues([Issue(containerctl.ERROR, "project.config", "Failed to load p.toml", "edit p.toml")])
        self.assertEqual([(e.cat, e.msg) for e in get_errors()], [("project.config", "Failed to load p.toml")])


if __name__ == "__main__":
    unittest.main()

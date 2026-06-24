"""Tests for the in-process error registry used by the TUI 'errors' tab."""

import unittest

import containerctl
from containerctl import (
    clear_errors,
    fetch_errors,
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

    def test_fetch_errors_empty_state(self):
        rows = fetch_errors()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Category"], "No errors")

    def test_fetch_errors_truncates_latest_message(self):
        long_msg = "x" * 100
        log_error("cat.a", long_msg)
        rows = fetch_errors()
        self.assertEqual(rows[0]["Category"], "cat.a")
        self.assertEqual(rows[0]["Count"], "1")
        self.assertEqual(len(rows[0]["Latest"]), 50)

    def test_clear_rebind_is_observable_through_module(self):
        # clear_errors() rebinds the module global; get_errors() must still
        # see the live list afterwards.
        log_error("cat.a", "1")
        clear_errors("cat.a")
        self.assertEqual(get_errors(), [])
        self.assertIs(get_errors(), containerctl._errors)


if __name__ == "__main__":
    unittest.main()

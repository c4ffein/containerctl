"""Tests for the hand-rolled TUI: Screen buffering, UI widgets, App state and key handling.

No terminal: ``Term.size`` is pinned, ``Term.getch`` is replaced by a scripted key feed,
``Screen.flush`` output is captured, and Docker calls go through the usual recorders/mocks.
Rendered text is compared after stripping ANSI escapes.
"""

import io
import re
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import containerctl
from containerctl import (
    UI,
    App,
    Container,
    DataFetcher,
    Docker,
    Event,
    Image,
    Issue,
    Network,
    ProjectConfig,
    RepoStatus,
    Screen,
    ServiceConfig,
    Tab,
    Term,
    Volume,
    WorkspaceReport,
    log_error,
)
from tests.helper import RunRecorder, container, listings, reset_errors

WARNING, ERROR = containerctl.WARNING, containerctl.ERROR
DOWN = "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?"

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def plain(text: str) -> str:
    return ANSI.sub("", text)


def buffered(scr: Screen) -> str:
    return plain("".join(scr.buf))


def keys(*seq):
    """A ``Term.getch`` stand-in that replays ``seq`` then fails loudly if asked for more."""
    it = iter(seq)

    def getch(timeout=None):
        try:
            return next(it)
        except StopIteration:  # pragma: no cover - a test asked for more keys than scripted
            raise AssertionError("getch called more times than keys scripted")

    return getch


def tab(columns, id_key="ID", key="containers"):
    return Tab(name="T", key=key, columns=columns, id_key=id_key, actions={})


class ScreenTests(unittest.TestCase):
    def test_writeln_buffers_move_then_text(self):
        scr = Screen()
        scr.writeln(3, 5, "hi")
        self.assertEqual(scr.buf, ["\033[3;5H", "hi"])

    def test_flush_writes_once_and_empties_buffer(self):
        scr = Screen()
        scr.write("a")
        scr.clear_line()
        scr.write("b")
        out = io.StringIO()
        with redirect_stdout(out):
            scr.flush()
        self.assertEqual(out.getvalue(), "a" + Term.CLEAR_LINE + "b")
        self.assertEqual(scr.buf, [])

    def test_clear_drops_pending_output(self):
        scr = Screen()
        scr.write("junk")
        scr.clear()
        self.assertEqual(scr.buf, [])


class TermStyleTests(unittest.TestCase):
    def test_style_wraps_and_resets(self):
        self.assertEqual(Term.style("x", Term.BOLD, Term.RED), Term.BOLD + Term.RED + "x" + Term.RESET)

    def test_size_is_rows_then_cols(self):
        with mock.patch.object(containerctl.os, "get_terminal_size", return_value=(120, 40)):
            self.assertEqual(Term.size(), (40, 120))


class UIWidgetTests(unittest.TestCase):
    def setUp(self):
        reset_errors()
        self.scr = Screen()

    def test_tab_bar_marks_selected_and_dims_others(self):
        tabs = [tab([], key="a"), tab([], key="b")]
        tabs[0].name, tabs[1].name = "One", "Two"
        UI.tab_bar(self.scr, tabs, 1, 80, 1)
        raw = "".join(self.scr.buf)
        self.assertIn(Term.DIM + " One " + Term.RESET, raw)
        self.assertIn(Term.BOLD + Term.INVERT + " Two " + Term.RESET, raw)

    def test_tab_bar_colors_the_tabs_named_in_alerts(self):
        tabs = [tab([], key="issues"), tab([], key="errors")]
        UI.tab_bar(self.scr, tabs, 0, 80, 1)
        self.assertNotIn(Term.RED, "".join(self.scr.buf))
        self.assertNotIn(Term.YELLOW, "".join(self.scr.buf))
        self.scr.clear()
        UI.tab_bar(self.scr, tabs, 0, 80, 1, alerts={"issues": Term.YELLOW})
        self.assertIn(Term.YELLOW, "".join(self.scr.buf))
        self.assertNotIn(Term.RED, "".join(self.scr.buf))
        self.scr.clear()
        UI.tab_bar(self.scr, tabs, 1, 80, 1, alerts={"errors": Term.RED})
        self.assertIn(Term.RED, "".join(self.scr.buf))  # also on the selected tab

    def test_list_header_flex_column_takes_remaining_width(self):
        t = tab([("a", "AAAA", 6), ("b", "BBBBBBBBBBBBBBBBBBBB", 0)])
        UI.list_header(self.scr, t, 3, 20)
        text = buffered(self.scr)
        # width 20 -> 18 usable; 6 for A + 1 space; flex gets 11 and the header is clipped to it
        self.assertEqual(text, "AAAA   BBBBBBBBBBB ")

    def test_list_row_clips_values_and_inverts_selection(self):
        t = tab([("Names", "NAME", 4), ("Status", "STATUS", 0)])
        item = Container("a", "longname", "i", "Up 3 hours")
        UI.list_row(self.scr, t, item, 4, 30, selected=False)
        self.assertEqual(buffered(self.scr), "long Up 3 hours")
        self.assertNotIn(Term.INVERT, "".join(self.scr.buf))
        self.scr.clear()
        UI.list_row(self.scr, t, item, 4, 30, selected=True)
        self.assertIn(Term.INVERT, "".join(self.scr.buf))

    def test_list_row_colors_matching_status(self):
        t = tab([("Names", "NAME", 10), ("Status", "STATUS", 10)])
        colors = {"exited": Term.RED, "up": Term.GREEN}
        UI.list_row(self.scr, t, Container("a", "up-ish", "i", "Exited (0)"), 4, 30, False, colors)
        raw = "".join(self.scr.buf)
        self.assertIn(Term.RED + "Exited (0)" + Term.RESET, raw)
        self.assertNotIn(Term.GREEN, raw)  # only the Status column is coloured, even if NAME matches a key

    def test_flex_column_vanishes_on_a_terminal_too_narrow_for_the_fixed_ones(self):
        # 18 + 56 fixed columns on 76 columns: the flex one gets nothing, not a negative slice from the wrong end
        t = tab([("cat", "CATEGORY", 18), ("msg", "MESSAGE", 56), ("hint", "HINT", 0)])
        item = Issue(WARNING, "image.versions", "hermes has 4 local versions", "docker rmi hermes:cctl-local-000000")
        UI.list_row(self.scr, t, item, 4, 76, selected=False)
        self.assertEqual(buffered(self.scr), "image.versions     hermes has 4 local versions")
        self.scr.clear()
        UI.list_header(self.scr, t, 3, 76)
        self.assertEqual(buffered(self.scr).rstrip(), "CATEGORY           MESSAGE")
        self.scr.clear()
        UI.list_row(self.scr, t, item, 4, 100, selected=False)  # room again: the hint is back, clipped to 22
        self.assertTrue(buffered(self.scr).endswith("  docker rmi hermes:cctl"))

    def test_list_row_no_color_when_status_matches_nothing(self):
        t = tab([("Status", "STATUS", 10)])
        UI.list_row(self.scr, t, Container("a", "n", "i", "Weird"), 4, 30, False, {"up": Term.GREEN})
        self.assertNotIn(Term.GREEN, "".join(self.scr.buf))

    def test_render_colors_container_status(self):
        # end-to-end through App.render: the colour map wired for the containers tab must reach the row
        with mock.patch.object(Term, "size", return_value=(24, 80)):
            app = App()
            app.fetcher.data["containers"] = [Container("a", "c", "i", "Exited (1)")]
            with redirect_stdout(io.StringIO()) as out:
                app.render()
        self.assertIn(Term.RED + "Exited (1)", out.getvalue())

    def test_status_bar_pads_and_clips_to_width(self):
        UI.status_bar(self.scr, "abc", 24, 6)
        self.assertEqual(buffered(self.scr), "abc   ")
        self.scr.clear()
        UI.status_bar(self.scr, "abcdefghij", 24, 4)
        self.assertEqual(buffered(self.scr), "abcd")

    def test_message_applies_optional_style(self):
        UI.message(self.scr, "hello", 23, 80)
        self.assertEqual("".join(self.scr.buf)[-5:], "hello")
        self.scr.clear()
        UI.message(self.scr, "hello", 23, 80, Term.GREEN)
        self.assertTrue("".join(self.scr.buf).endswith(Term.GREEN + "hello" + Term.RESET))


class ConfirmAndInfoBoxTests(unittest.TestCase):
    def _confirm(self, *seq):
        with mock.patch.object(Term, "getch", keys(*seq)):
            with redirect_stdout(io.StringIO()) as out:
                result = UI.confirm("Sure?", 23)
        return result, plain(out.getvalue())

    def test_yes(self):
        result, out = self._confirm("y")
        self.assertTrue(result)
        self.assertIn("Sure? [y/N]", out)

    def test_no_enter_and_escape_decline(self):
        for key in ("n", "N", Term.KEY_ENTER, Term.KEY_ESC):
            self.assertFalse(self._confirm(key)[0], repr(key))

    def test_unrelated_keys_are_ignored_until_an_answer(self):
        self.assertTrue(self._confirm("x", "7", "Y")[0])

    def test_info_box_draws_title_lines_and_waits_for_a_key(self):
        getch = mock.Mock(return_value="x")
        with mock.patch.object(Term, "getch", getch):
            with redirect_stdout(io.StringIO()) as out:
                UI.info_box("Title", ["first line", "second"], 24, 80)
        text = plain(out.getvalue())
        self.assertIn("Title", text)
        self.assertIn("│ first line", text)
        self.assertIn("│ second    ", text)  # padded to the box width
        self.assertIn("[Press any key to close]", text)
        getch.assert_called_once()

    def test_a_warning_box_only_closes_on_enter(self):
        getch = mock.Mock(side_effect=["d", "y", "x", Term.KEY_ENTER, "z"])
        with mock.patch.object(Term, "getch", getch):
            with redirect_stdout(io.StringIO()) as out:
                UI.info_box("Title", ["line"], 24, 80, ok=True)
        self.assertIn("[Press Enter: ok]", plain(out.getvalue()))
        self.assertEqual(getch.call_count, 4)

    def test_info_box_clips_lines_to_terminal_width(self):
        with mock.patch.object(Term, "getch", keys("x")):
            with redirect_stdout(io.StringIO()) as out:
                UI.info_box("T", ["x" * 200], 24, 40)
        # every positioned write (one per box row) must fit inside the 40 columns
        rows = [plain(chunk) for chunk in re.split(r"\x1b\[\d+;\d+H", out.getvalue()) if chunk]
        self.assertGreater(len(rows), 3)
        for row in rows:
            self.assertLessEqual(len(row), 40, row)


class AppBase(unittest.TestCase):
    ROWS, COLS = 24, 80

    def setUp(self):
        reset_errors()
        self._size = mock.patch.object(Term, "size", return_value=(self.ROWS, self.COLS))
        self._size.start()
        self.app = App()  # the fetcher is never started: tests poke app.fetcher.data directly

    def tearDown(self):
        self._size.stop()

    def goto(self, key):
        self.app.current_tab = next(i for i, t in enumerate(self.app.tabs) if t.key == key)
        self.app.selected_index = 0
        self.app.scroll_offset = 0

    def feed(self, key):
        self.app.handle_key(key)

    def rendered(self):
        with redirect_stdout(io.StringIO()) as out:
            self.app.render()
        return plain(out.getvalue())

    def containers(self, n):
        self.app.fetcher.data["containers"] = [Container(f"id{i:010d}", f"c{i}", "img", "Up") for i in range(n)]


class AppItemsTests(AppBase):
    def test_tabs_are_in_documented_order(self):
        self.assertEqual(
            [t.key for t in self.app.tabs],
            ["containers", "images", "volumes", "networks", "projects", "services", "issues", "errors"],
        )

    def test_items_come_from_fetcher_cache(self):
        self.containers(2)
        self.assertEqual([c.Names for c in self.app.items()], ["c0", "c1"])

    def test_items_cache_copy_is_not_the_live_list(self):
        self.containers(1)
        self.app.items().append("x")
        self.assertEqual(len(self.app.fetcher.data["containers"]), 1)

    def test_an_empty_tab_has_no_items_and_nothing_selected(self):
        """No placeholder rows: what an empty list says is drawn by render, and no key acts on it"""
        for key in ("projects", "services", "issues", "errors"):
            self.goto(key)
            self.assertEqual(self.app.items(), [], key)
            self.assertIsNone(self.app.selected_id(), key)

    def test_issues_tab_shows_what_the_fetcher_detected(self):
        self.goto("issues")
        self.app.fetcher.data["issues"] = [Issue(WARNING, "image.versions", "hermes has 4 versions", "docker rmi x")]
        self.assertEqual([(i.cat, i.hint) for i in self.app.items()], [("image.versions", "docker rmi x")])
        self.assertEqual(self.app.selected_id(), "image.versions")

    def test_errors_tab_lists_the_log_newest_first(self):
        self.goto("errors")
        log_error("docker.command", "docker ps")
        log_error("image.build", "boom")
        self.assertEqual([e.cat for e in self.app.items()], ["image.build", "docker.command"])
        self.assertIsInstance(self.app.items()[0], Event)
        self.assertEqual(self.app.selected_id(), "image.build")

    def test_tab_alerts_follow_issue_levels_and_the_log(self):
        self.assertEqual(self.app.tab_alerts(), {})
        self.app.fetcher.data["issues"] = [Issue(WARNING, "a", "b")]
        self.assertEqual(self.app.tab_alerts(), {"issues": Term.YELLOW})
        self.app.fetcher.data["issues"] = [Issue(WARNING, "a", "b"), Issue(ERROR, "c", "d")]
        self.assertEqual(self.app.tab_alerts(), {"issues": Term.RED})  # one broken thing outranks the warnings
        log_error("x", "y")
        self.assertEqual(self.app.tab_alerts(), {"issues": Term.RED, "errors": Term.RED})
        self.app.fetcher.data["issues"] = []
        self.assertEqual(self.app.tab_alerts(), {"errors": Term.RED})

    def test_banner_only_when_all_four_listings_failed_for_one_reason(self):
        self.assertEqual(self.app.banner(), "")
        self.app.fetcher.failures = dict.fromkeys(containerctl.LISTINGS, "Docker daemon is not running")
        self.assertEqual(self.app.banner(), "Docker daemon is not running")
        self.app.fetcher.failures["volumes"] = "Unknown docker failure"  # two stories: the Issues tab tells them
        self.assertEqual(self.app.banner(), "")
        self.app.fetcher.failures = {"volumes": "Unknown docker failure"}  # docker is up, one listing is not
        self.assertEqual(self.app.banner(), "")

    def test_selected_id_uses_each_tabs_id_key(self):
        self.containers(1)
        self.assertEqual(self.app.selected_id(), "id0000000000")
        self.goto("volumes")
        self.app.fetcher.data["volumes"] = [Volume("vol", "local")]
        self.assertEqual(self.app.selected_id(), "vol")

    def test_selected_item_none_when_out_of_range(self):
        self.containers(1)
        self.app.selected_index = 5
        self.assertIsNone(self.app.selected_item())
        self.assertIsNone(self.app.selected_id())


class AppRenderTests(AppBase):
    def test_empty_tabs_say_what_is_missing(self):
        for key, text in (
            ("containers", "No containers found"),
            ("projects", f"No projects configured: add TOML to {containerctl.PROJECTS_DIR}/"),
            ("services", f"No services configured: add TOML to {containerctl.SERVICES_DIR}/"),
            ("issues", "No issues"),
            ("errors", "No errors"),
        ):
            self.goto(key)
            out = self.rendered()
            self.assertIn(text[: self.COLS - 2], out, key)
            self.assertIn("0 items", out, key)

    def test_a_failed_listing_says_why_in_its_panel_and_only_there(self):
        self.app.fetcher.failures = {"images": "Unknown docker failure"}
        self.containers(1)
        self.assertNotIn("Unknown docker failure", self.rendered())  # containers listed fine: no banner either
        self.goto("images")
        with redirect_stdout(io.StringIO()) as out:
            self.app.render()
        self.assertIn(Term.RED + "  Unknown docker failure", out.getvalue())
        self.assertNotIn("No images found", plain(out.getvalue()))

    def test_a_tab_whose_listing_skipped_rows_says_its_rows_are_not_all(self):
        self.containers(2)
        self.app.fetcher.notes = {"containers": "1 unreadable row left out: see Issues"}
        with redirect_stdout(io.StringIO()) as out:
            self.app.render()
        self.assertIn(Term.YELLOW + "── 1 unreadable row left out: see Issues ──", out.getvalue())
        self.assertIn("c0", plain(out.getvalue()))  # the rows that are fine still list
        self.goto("images")
        self.assertNotIn("unreadable", self.rendered())  # that tab's rows are all there
        self.goto("containers")
        self.app.fetcher.failures = dict.fromkeys(containerctl.LISTINGS, "Docker daemon is not running")
        self.assertNotIn("unreadable", self.rendered())  # docker down outranks it

    def test_docker_down_is_a_banner_on_every_tab(self):
        self.app.fetcher.failures = dict.fromkeys(containerctl.LISTINGS, "Docker daemon is not running")
        for key in ("containers", "projects", "errors"):
            self.goto(key)
            out = self.rendered()
            self.assertIn("── Docker daemon is not running ──", out, key)
        self.app.fetcher.failures = {}
        self.assertNotIn("Docker daemon is not running", self.rendered())

    def test_render_shows_tabs_header_rows_and_status_bar(self):
        self.containers(2)
        text = self.rendered()
        for name in ("Containers", "Images", "Issues"):
            self.assertIn(name, text)
        self.assertIn("NAME", text)
        self.assertIn("c0", text)
        self.assertIn("c1", text)
        self.assertIn("[s]tart [S]top", text)
        self.assertIn("2 items", text)

    def test_render_empty_state_message(self):
        self.assertIn("No containers found", self.rendered())
        self.assertIn("0 items", self.rendered())

    def test_render_clamps_selection_when_list_shrinks(self):
        self.containers(3)
        self.app.selected_index = 2
        self.containers(1)
        self.rendered()
        self.assertEqual(self.app.selected_index, 0)

    def test_render_shows_pending_message_and_keeps_it_until_next_key(self):
        self.app.set_message("Started abc", Term.GREEN)
        self.assertIn("Started abc", self.rendered())
        self.feed("x")
        self.assertNotIn("Started abc", self.rendered())

    def test_render_only_shows_visible_window(self):
        self.containers(50)
        self.app.scroll_offset = 40
        text = self.rendered()
        self.assertIn("c40", text)
        self.assertNotIn("c39 ", text)
        self.assertIn("50 items", text)

    def test_status_bar_keys_follow_tab(self):
        expectations = {
            "projects": "[e]nter [d]estroy",
            "services": "[u]p [d]own [r]ecreate",
            "networks": "[e]dit [d]elete",
            "issues": "[q]uit",
            "errors": "[q]uit",
            "images": "[d]elete [q]uit",
        }
        for key, hint in expectations.items():
            self.goto(key)
            self.assertIn(hint, self.rendered(), key)

    def test_render_flushes_in_one_write(self):
        self.containers(1)
        with mock.patch.object(containerctl.sys.stdout, "write") as write:
            self.app.render()
        write.assert_called_once()


class AppNavigationTests(AppBase):
    def test_q_quits(self):
        self.feed("q")
        self.assertFalse(self.app.running)

    def test_down_up_bounded(self):
        self.containers(2)
        self.feed("i")
        self.assertEqual(self.app.selected_index, 0)
        self.feed("k")
        self.feed(Term.KEY_DOWN)
        self.feed("k")
        self.assertEqual(self.app.selected_index, 1)
        self.feed(Term.KEY_UP)
        self.assertEqual(self.app.selected_index, 0)

    def test_scrolling_follows_selection(self):
        list_height = self.ROWS - 6
        self.containers(list_height + 5)
        for _ in range(list_height + 2):
            self.feed("k")
        self.assertEqual(self.app.selected_index, list_height + 2)
        self.assertEqual(self.app.scroll_offset, 3)
        for _ in range(list_height + 2):
            self.feed("i")
        self.assertEqual((self.app.selected_index, self.app.scroll_offset), (0, 0))

    def test_g_and_G_jump(self):
        self.containers(40)
        self.feed("G")
        self.assertEqual(self.app.selected_index, 39)
        self.assertEqual(self.app.scroll_offset, 40 - (self.ROWS - 6))
        self.feed("g")
        self.assertEqual((self.app.selected_index, self.app.scroll_offset), (0, 0))

    def test_tab_cycling_wraps_and_resets_selection(self):
        self.containers(3)
        self.feed("k")
        self.feed("l")
        self.assertEqual((self.app.current_tab, self.app.selected_index), (1, 0))
        self.feed(Term.KEY_RIGHT)
        self.feed(Term.KEY_TAB)
        self.assertEqual(self.app.current_tab, 3)
        self.feed("j")
        self.assertEqual(self.app.current_tab, 2)
        self.app.current_tab = 0
        self.feed(Term.KEY_LEFT)
        self.assertEqual(self.app.current_tab, len(self.app.tabs) - 1)
        self.feed("l")
        self.assertEqual(self.app.current_tab, 0)

    def test_number_keys_select_tabs(self):
        self.feed("7")
        self.assertEqual(self.app.current().key, "issues")
        self.feed("8")
        self.assertEqual(self.app.current().key, "errors")
        self.feed("1")
        self.assertEqual(self.app.current().key, "containers")

    def test_action_keys_without_selection_are_noops(self):
        rec = RunRecorder()
        with mock.patch.object(Docker, "_run", rec):
            for key in "sSrdeLo":
                self.feed(key)
        self.assertEqual(rec.calls, [])


class AppSwitchGraceTests(AppBase):
    """Its row gone, the cursor fell onto a neighbour by itself: a key landing right after may not be for it. It
    does nothing, a box to ok says so, and pressing it again is the user's business."""

    def setUp(self):
        super().setUp()
        self.containers(2)
        self.rendered()  # c0 is under the cursor
        self.rec = RunRecorder()
        self.now = 100.0
        for patch in (
            mock.patch.object(Docker, "_run", self.rec),
            mock.patch.object(containerctl.time, "monotonic", side_effect=lambda: self.now),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.app.fetcher.data["containers"] = self.app.fetcher.data["containers"][1:]  # c0 is gone
        self.rendered()  # a redraw: the cursor is on c1 now

    def press(self, key):
        """The box the key opened, if any; the loop's redraw follows"""
        with mock.patch.object(UI, "info_box") as box:
            self.feed(key)
        self.rendered()
        return box

    def test_a_key_right_after_does_nothing_and_a_box_says_so(self):
        self.now += 0.3
        box = self.press("s")
        self.assertEqual(self.rec.calls, [])
        title, lines = box.call_args.args[:2]
        self.assertEqual(title, "Selection changed")
        self.assertIn("the selection just moved to 'id0000000001'", lines[0])
        self.assertEqual(box.call_args.kwargs, {"ok": True})  # Enter alone closes it

    def test_once_oked_the_key_pressed_again_acts(self):
        self.press("s")
        self.press("s").assert_not_called()
        self.assertEqual(self.rec.calls, [["start", "id0000000001"]])  # once: the box retried nothing

    def test_a_key_after_the_grace_period_is_for_the_row_on_screen(self):
        self.now += App.SWITCH_GRACE
        self.press("s").assert_not_called()
        self.assertEqual(self.rec.last, ["start", "id0000000001"])

    def test_moving_the_cursor_by_hand_ends_the_wait(self):
        self.press("g").assert_not_called()
        self.press("s").assert_not_called()
        self.assertEqual(self.rec.last, ["start", "id0000000001"])

    def test_a_key_that_finds_its_row_gone_gets_the_same_box_and_one_ok_is_enough(self):
        self.now += 10
        self.app.fetcher.data["containers"] = [Container("id0000000009", "c9", "img", "Up")]  # c1 gone, no redraw
        box = self.press("s")  # for c1
        self.assertEqual(box.call_args.args[0], "Selection changed")
        self.assertIn("the row that key was for is gone", box.call_args.args[1][0])
        self.assertEqual(self.rec.calls, [])
        self.press("s").assert_not_called()  # c9 is on screen, the box was oked: no second box
        self.assertEqual(self.rec.last, ["start", "id0000000009"])


class AppContainerActionTests(AppBase):
    def setUp(self):
        super().setUp()
        self.containers(2)
        self.feed("k")  # select c1
        self.rec = RunRecorder()
        self._run = mock.patch.object(Docker, "_run", self.rec)
        self._run.start()

    def tearDown(self):
        self._run.stop()
        super().tearDown()

    def test_start_stop_restart_act_on_selected_id(self):
        self.feed("s")
        self.assertEqual(self.rec.last, ["start", "id0000000001"])
        self.feed("S")
        self.assertEqual(self.rec.last, ["stop", "id0000000001"])
        self.feed("r")
        self.assertEqual(self.rec.last, ["restart", "id0000000001"])
        self.assertEqual(self.app.message_text, "Restarted id0000000001")
        self.assertEqual(self.app.message_style, Term.GREEN)

    def test_failed_action_reports_in_red(self):
        self.rec.returncode = 1
        self.feed("s")
        self.assertEqual(self.app.message_text, "Failed to start id0000000001")
        self.assertEqual(self.app.message_style, Term.RED)

    def test_delete_requires_confirmation(self):
        with mock.patch.object(UI, "confirm", return_value=False) as confirm:
            self.feed("d")
        confirm.assert_called_once_with("Delete container 'c1'?", self.ROWS - 1)
        self.assertEqual(self.rec.calls, [])
        with mock.patch.object(UI, "confirm", return_value=True):
            self.feed("d")
        self.assertEqual(self.rec.last, ["rm", "-f", "id0000000001"])
        self.assertEqual(self.app.message_text, "Deleted c1")

    def test_origin_opens_info_box_with_investigation(self):
        result = {"source": "compose", "details": ["project: x"]}
        with mock.patch.object(Docker, "investigate_source", return_value=result) as inv:
            with mock.patch.object(UI, "info_box") as box:
                self.feed("o")
        inv.assert_called_once_with("id0000000001")
        box.assert_called_once_with("Source: compose", ["Container: c1", "project: x"], self.ROWS, self.COLS)

    def test_exec_leaves_and_reenters_tui_around_shell(self):
        order = []
        with (
            mock.patch.object(self.app, "exit_tui_temporarily", side_effect=lambda: order.append("exit")),
            mock.patch.object(Docker, "shell", side_effect=lambda cid: order.append(f"shell:{cid}")),
            mock.patch.object(self.app, "enter_tui", side_effect=lambda: order.append("enter")),
        ):
            self.feed("e")
        self.assertEqual(order, ["exit", "shell:id0000000001", "enter"])

    def test_logs_waits_for_enter_before_returning(self):
        with (
            mock.patch.object(self.app, "exit_tui_temporarily"),
            mock.patch.object(Docker, "logs") as logs,
            mock.patch("builtins.input", return_value="") as inp,
            mock.patch.object(self.app, "enter_tui"),
        ):
            self.feed("L")
        logs.assert_called_once_with("id0000000001", tail=100)
        inp.assert_called_once()


class AppOtherTabActionTests(AppBase):
    def setUp(self):
        super().setUp()
        self.rec = RunRecorder()
        self._run = mock.patch.object(Docker, "_run", self.rec)
        self._run.start()

    def tearDown(self):
        self._run.stop()
        super().tearDown()

    def test_image_delete_names_repo_and_tag_in_prompt(self):
        self.goto("images")
        self.app.fetcher.data["images"] = [Image("sha", "nginx", "1.27", "1MB")]
        with mock.patch.object(UI, "confirm", return_value=True) as confirm:
            self.feed("d")
        confirm.assert_called_once_with("Delete image 'nginx:1.27'?", self.ROWS - 1)
        self.assertEqual(self.rec.last, ["rmi", "-f", "sha"])

    def test_a_keypress_acts_on_one_snapshot_of_the_list(self):
        """The fetcher swaps the lists between two reads: the row named in the prompt is the row deleted"""
        self.goto("images")
        snapshots = [[Image("sha1", "nginx", "1.27", "1MB")], [Image("sha2", "redis", "7", "1MB")], []]

        def swapping(key):
            return snapshots.pop(0) if len(snapshots) > 1 else snapshots[0]

        with (
            mock.patch.object(self.app.fetcher, "get", side_effect=swapping),
            mock.patch.object(UI, "confirm", return_value=True) as confirm,
        ):
            self.app._handle_key("d")
        confirm.assert_called_once_with("Delete image 'nginx:1.27'?", self.ROWS - 1)
        self.assertEqual(self.rec.last, ["rmi", "-f", "sha1"])

    def test_a_key_acts_on_the_row_that_was_drawn_wherever_it_is_now(self):
        """The screen is only redrawn on a keypress: by then a fetch may have moved the rows under the cursor"""
        self.goto("images")
        nginx, redis = Image("sha1", "nginx", "1.27", "1MB"), Image("sha2", "redis", "7", "1MB")
        self.app.fetcher.data["images"] = [nginx, redis]
        self.rendered()  # nginx is under the cursor
        self.app.fetcher.data["images"] = [Image("sha0", "alpine", "3", "1MB"), nginx, redis]
        with mock.patch.object(UI, "confirm", return_value=True) as confirm:
            self.feed("d")
        confirm.assert_called_once_with("Delete image 'nginx:1.27'?", self.ROWS - 1)
        self.assertEqual(self.rec.last, ["rmi", "-f", "sha1"])
        self.assertEqual(self.app.selected_index, 1)  # the selection followed it

    def test_tags_of_one_image_are_rows_of_their_own(self):
        self.goto("images")
        old, new = Image("sha1", "nginx", "1.26", "1MB"), Image("sha1", "nginx", "1.27", "1MB")
        self.app.fetcher.data["images"] = [old, new]
        self.feed("k")
        self.rendered()  # 1.27 is under the cursor
        self.app.fetcher.data["images"] = [new, old]
        with mock.patch.object(UI, "confirm", return_value=False) as confirm:
            self.feed("d")
        confirm.assert_called_once_with("Delete image 'nginx:1.27'?", self.ROWS - 1)

    def test_a_key_for_a_row_that_is_gone_acts_on_nothing_and_says_so(self):
        self.goto("images")
        self.app.fetcher.data["images"] = [Image("sha1", "nginx", "1.27", "1MB")]
        self.rendered()
        self.app.fetcher.data["images"] = [Image("sha2", "redis", "7", "1MB")]  # same index, another image
        with mock.patch.object(UI, "confirm", return_value=True) as confirm, mock.patch.object(UI, "info_box") as box:
            self.feed("d")
        confirm.assert_not_called()
        self.assertEqual(self.rec.calls, [])
        self.assertEqual(box.call_args.args[0], "Selection changed")  # a modal: it can't be missed
        self.rendered()
        with mock.patch.object(UI, "confirm", return_value=True):  # redis is on screen now: the next key is for it
            self.feed("d")
        self.assertEqual(self.rec.last, ["rmi", "-f", "sha2"])

    def test_a_row_that_appeared_in_an_empty_list_is_not_acted_on_unseen(self):
        self.goto("images")
        self.rendered()  # "No images found"
        self.app.fetcher.data["images"] = [Image("sha1", "nginx", "1.27", "1MB")]
        with mock.patch.object(UI, "confirm", return_value=True) as confirm, mock.patch.object(UI, "info_box") as box:
            self.feed("d")
        confirm.assert_not_called()
        box.assert_called_once()
        self.assertEqual(self.rec.calls, [])

    def test_a_redraw_keeps_the_row_selected_not_the_position(self):
        self.goto("volumes")
        a, b, c = (Volume(name, "local") for name in "abc")
        self.app.fetcher.data["volumes"] = [a, b, c]
        self.feed("k")
        self.rendered()  # b is under the cursor
        self.app.fetcher.data["volumes"] = [Volume("0", "local"), a, b, c]
        self.rendered()  # a resize: no key in between
        self.assertEqual(self.app.selected_item().Name, "b")
        self.app.fetcher.data["volumes"] = [Volume("0", "local"), a, c]  # b is gone: c took its place
        self.rendered()
        self.assertEqual(self.app.selected_item().Name, "c")
        self.app.fetcher.data["volumes"] = [a]  # nothing at that position anymore: the last line
        self.rendered()
        self.assertEqual(self.app.selected_item().Name, "a")
        with mock.patch.object(UI, "confirm", return_value=True), mock.patch.object(UI, "info_box") as box:
            self.feed("d")  # right after the cursor fell: a box, nothing done
            self.assertEqual(self.rec.calls, [])
            self.feed("d")  # oked, pressed again
        box.assert_called_once()
        self.assertEqual(self.rec.last, ["volume", "rm", "a"])

    def test_moving_starts_from_the_row_that_was_drawn(self):
        self.goto("volumes")
        a, b, c = (Volume(name, "local") for name in "abc")
        self.app.fetcher.data["volumes"] = [a, b, c]
        self.rendered()  # a is under the cursor
        self.app.fetcher.data["volumes"] = [Volume("0", "local"), a, b, c]
        self.feed("k")
        self.assertEqual(self.app.selected_item().Name, "b")

    def test_volume_delete(self):
        self.goto("volumes")
        self.app.fetcher.data["volumes"] = [Volume("vol", "local", Mountpoint="/v")]
        with mock.patch.object(UI, "confirm", return_value=True):
            self.feed("d")
        self.assertEqual(self.rec.last, ["volume", "rm", "vol"])

    def test_network_edit_lists_connected_containers(self):
        self.goto("networks")
        self.app.fetcher.data["networks"] = [Network("net1", "cctl-net-x", "bridge", "local")]
        info = mock.Mock(Name="cctl-net-x", Containers={"abc": {"Name": "cctl-prj-x"}})
        with mock.patch.object(Docker, "inspect_network", return_value=info):
            with mock.patch.object(UI, "info_box") as box:
                self.feed("e")
        box.assert_called_once_with("Network: cctl-net-x", ["cctl-prj-x"], self.ROWS, self.COLS)

    def test_network_edit_failure_sets_red_message(self):
        self.goto("networks")
        self.app.fetcher.data["networks"] = [Network("net1", "n", "bridge", "local")]
        with mock.patch.object(Docker, "inspect_network", return_value=None):
            self.feed("e")
        self.assertEqual(self.app.message_style, Term.RED)

    def test_project_enter_uses_loaded_config(self):
        self.goto("projects")
        self.app.fetcher.data["projects"] = [containerctl.ProjectRow("demo", "running", "i", "-")]
        project = ProjectConfig(name="demo", image="i")
        with (
            mock.patch.object(containerctl, "read_projects", return_value=({"demo": project}, [])),
            mock.patch.object(self.app, "exit_tui_temporarily"),
            mock.patch.object(Docker, "enter_project") as enter,
            mock.patch.object(self.app, "enter_tui"),
        ):
            self.feed("e")
        enter.assert_called_once_with(project)

    def test_project_destroy_without_container_warns_instead_of_asking(self):
        self.goto("projects")
        self.app.fetcher.data["projects"] = [containerctl.ProjectRow("demo", "not created", "i", "-")]
        project = ProjectConfig(name="demo", image="i")
        with (
            mock.patch.object(containerctl, "read_projects", return_value=({"demo": project}, [])),
            mock.patch.object(Docker, "find_project_container", return_value=None),
            mock.patch.object(UI, "confirm") as confirm,
        ):
            self.feed("d")
        confirm.assert_not_called()
        self.assertEqual(self.app.message_style, Term.YELLOW)

    def _destroy(self, report, answer):
        self.goto("projects")
        self.app.fetcher.data["projects"] = [containerctl.ProjectRow("demo", "running", "i", "-")]
        project = ProjectConfig(name="demo", image="i")
        with (
            mock.patch.object(containerctl, "read_projects", return_value=({"demo": project}, [])),
            mock.patch.object(Docker, "find_project_container", return_value=container("cctl-prj-demo")),
            mock.patch.object(containerctl, "project_report", return_value=report) as check,
            mock.patch.object(UI, "confirm", return_value=answer) as confirm,
            mock.patch.object(Docker, "destroy_project", return_value=True) as destroy,
        ):
            self.feed("d")
        check.assert_called_once_with(project)
        return project, confirm, destroy

    def test_project_destroy_confirmed(self):
        project, confirm, destroy = self._destroy(WorkspaceReport(statuses=[RepoStatus(name="app")]), True)
        confirm.assert_called_once_with("Destroy project 'demo'?", self.ROWS - 1)
        destroy.assert_called_once_with(project)
        self.assertEqual(self.app.message_text, "Destroyed demo")

    def test_project_destroy_prompt_names_dirty_dirs_and_still_destroys_on_yes(self):
        report = WorkspaceReport(statuses=[RepoStatus(name="app", ahead=2)], extra=[], missing=["lib"])
        project, confirm, destroy = self._destroy(report, True)
        confirm.assert_called_once_with("Destroy project 'demo'? Workspace is DIRTY: app, lib.", self.ROWS - 1)
        destroy.assert_called_once_with(project)

    def test_project_destroy_declined_on_dirty_prompt_does_nothing(self):
        _, _, destroy = self._destroy(WorkspaceReport(statuses=[RepoStatus(name="app", untracked=True)]), False)
        destroy.assert_not_called()
        self.assertEqual(self.app.message_text, "")

    def test_service_down_confirmed(self):
        self.goto("services")
        self.app.fetcher.data["services"] = [containerctl.ServiceRow("web", "running", "i", "")]
        service = ServiceConfig(name="web", image="i")
        with (
            mock.patch.object(containerctl, "read_services", return_value=({"web": service}, [])),
            mock.patch.object(Docker, "find_service_container", return_value=container("cctl-svc-web")),
            mock.patch.object(UI, "confirm", return_value=True) as confirm,
            mock.patch.object(Docker, "service_down", return_value=True) as down,
        ):
            self.feed("d")
        confirm.assert_called_once_with("Stop service 'web'?", self.ROWS - 1)
        down.assert_called_once_with(service)

    def test_service_recreate_declined_does_nothing(self):
        self.goto("services")
        self.app.fetcher.data["services"] = [containerctl.ServiceRow("web", "running", "i", "")]
        with (
            mock.patch.object(
                containerctl, "read_services", return_value=({"web": ServiceConfig(name="web", image="i")}, [])
            ),
            mock.patch.object(UI, "confirm", return_value=False),
            mock.patch.object(Docker, "service_recreate") as recreate,
        ):
            self.feed("r")
        recreate.assert_not_called()

    def test_error_log_clear_asks_first(self):
        log_error("a.cat", "1")
        log_error("b.cat", "2")
        self.goto("errors")
        with mock.patch.object(UI, "confirm", return_value=False) as confirm:
            self.feed("c")
        confirm.assert_called_once_with("Clear the error log?", self.ROWS - 1)
        self.assertEqual(len(containerctl.get_errors()), 2)
        with mock.patch.object(UI, "confirm", return_value=True):
            self.feed("c")
        self.assertEqual(containerctl.get_errors(), [])
        self.assertEqual((self.app.message_text, self.app.message_style), ("Cleared 2 errors", Term.GREEN))
        self.assertEqual(self.app.tab_alerts(), {})  # red was the log holding something

    def test_error_log_clear_on_an_empty_log_asks_nothing(self):
        self.goto("errors")
        with mock.patch.object(UI, "confirm") as confirm:
            self.feed("c")
        confirm.assert_not_called()

    def test_enter_on_an_error_shows_its_message_and_context_in_full(self):
        log_error("docker.container_parse", "missing 'ID' " + "x" * 100, {"data": "{'Names': 'c0'}"})
        self.goto("errors")
        with mock.patch.object(UI, "info_box") as box:
            self.feed(Term.KEY_ENTER)
        title, lines, rows, cols = box.call_args.args
        self.assertRegex(title, r"^\d\d:\d\d:\d\d  docker\.container_parse$")
        self.assertEqual("".join(lines[:2]), "missing 'ID' " + "x" * 100)  # wrapped, not clipped
        self.assertEqual(lines[2:], ["", "{'data': \"{'Names': 'c0'}\"}"])
        self.assertEqual((rows, cols), (self.ROWS, self.COLS))

    def test_enter_on_an_issue_shows_its_message_and_hint_in_full(self):
        self.goto("issues")
        self.app.fetcher.data["issues"] = [Issue(ERROR, "project.config", "Failed to load " + "p" * 90, "edit p.toml")]
        with mock.patch.object(UI, "info_box") as box:
            self.feed(Term.KEY_ENTER)
        title, lines, _, _ = box.call_args.args
        self.assertEqual(title, "error  project.config")
        self.assertEqual("".join(lines[:-2]).replace(" ", ""), "Failedtoload" + "p" * 90)
        self.assertEqual(lines[-2:], ["", "edit p.toml"])

    def test_enter_on_an_issue_shows_what_docker_said(self):
        self.goto("issues")
        detail = "docker ps -a: weird\ndocker images: weird"
        self.app.fetcher.data["issues"] = [
            Issue(ERROR, "docker", "Unknown docker failure (containers, images)", "h", detail)
        ]
        with mock.patch.object(UI, "info_box") as box:
            self.feed(Term.KEY_ENTER)
        lines = box.call_args.args[1]
        self.assertEqual(lines[-4:], ["", "docker ps -a: weird", "", "docker images: weird"])

    def test_enter_on_an_empty_error_log_does_nothing(self):
        self.goto("errors")
        with mock.patch.object(UI, "info_box") as box:
            self.feed(Term.KEY_ENTER)
        box.assert_not_called()

    def test_an_action_docker_gave_no_answer_to_says_why_and_does_nothing(self):
        self.goto("projects")
        self.app.fetcher.data["projects"] = [containerctl.ProjectRow("demo", "unknown", "i", "-")]
        project = ProjectConfig(name="demo", image="i")
        rec = RunRecorder(returncode=1, stderr="Cannot connect to the Docker daemon")
        for key in ("d", "e"):  # destroy asks docker first; enter leaves the TUI, then asks
            with (
                mock.patch.object(containerctl, "read_projects", return_value=({"demo": project}, [])),
                mock.patch.object(Docker, "_run", rec),
                mock.patch.object(UI, "confirm") as confirm,
                redirect_stdout(io.StringIO()) as out,
            ):
                self.feed(key)
            confirm.assert_not_called()
            self.assertEqual(
                (self.app.message_text, self.app.message_style),
                ("Docker daemon is not running: start docker", Term.RED),
            )
            self.assertTrue(out.getvalue().endswith(Term.ALT_SCREEN + Term.HIDE_CURSOR), key)  # back in the TUI
        self.assertEqual({call[0] for call in rec.calls}, {"ps"})
        self.assertEqual([e.cat for e in containerctl.get_errors()], ["docker.command"] * 2)

    def test_a_key_that_is_no_key_is_ignored(self):
        self.feed("")  # stdin at EOF: not a tab number
        self.assertEqual(self.app.current_tab, 0)


class AppRunLoopTests(AppBase):
    def test_run_renders_handles_keys_until_quit_and_restores_terminal(self):
        with (
            mock.patch.object(containerctl.signal, "signal"),
            mock.patch.object(self.app.fetcher, "start"),
            mock.patch.object(self.app.fetcher, "stop") as stop,
            mock.patch.object(Term, "getch", keys("l", "q")),
            redirect_stdout(io.StringIO()) as out,
        ):
            self.app.run()
        self.assertEqual(self.app.current_tab, 1)
        self.assertFalse(self.app.running)
        stop.assert_called_once()
        self.assertTrue(out.getvalue().startswith(Term.ALT_SCREEN + Term.HIDE_CURSOR))
        self.assertTrue(out.getvalue().endswith(Term.SHOW_CURSOR + Term.MAIN_SCREEN))

    def test_a_fetch_that_lands_is_drawn_without_waiting_for_a_key(self):
        script = iter(["", "fetch", "", "q"])  # "" is a poll that timed out; "fetch" one during which a fetch landed

        def getch(timeout=None):
            self.assertEqual(timeout, App.REDRAW_POLL)
            step = next(script)
            if step == "fetch":
                self.containers(1)
                self.app.fetcher.version += 1
                return ""
            return step

        with (
            mock.patch.object(containerctl.signal, "signal"),
            mock.patch.object(self.app.fetcher, "start"),
            mock.patch.object(self.app.fetcher, "stop"),
            mock.patch.object(Term, "getch", getch),
            mock.patch.object(self.app, "render", wraps=self.app.render) as render,
            mock.patch.object(self.app, "handle_key", wraps=self.app.handle_key) as handle,
            redirect_stdout(io.StringIO()) as out,
        ):
            self.app.run()
        handled = [call.args[0] for call in handle.call_args_list]
        self.assertEqual(render.call_count, 2)  # at start, and once for the fetch: idle polls draw nothing
        self.assertIn("c0", plain(out.getvalue()))
        self.assertEqual(handled, ["q"])  # a timed-out poll is not a key (it would wipe the status message)

    def test_the_fetcher_counts_the_fetches_that_landed(self):
        with mock.patch.object(self.app.fetcher, "_fetch", side_effect=[RuntimeError("boom")]):
            self.app.fetcher._fetch_all()
        self.assertEqual(self.app.fetcher.version, 1)  # a crashed fetch changes what there is to show too

    def test_the_keyboard_is_raw_for_the_session_and_handed_back_around_what_runs_outside(self):
        calls = []
        with (
            mock.patch.object(Term, "raw_on", side_effect=lambda: calls.append("raw")),
            mock.patch.object(Term, "raw_off", side_effect=lambda: calls.append("cooked")),
            redirect_stdout(io.StringIO()),
        ):
            self.app.enter_tui()
            self.app.exit_tui_temporarily()  # a docker exec, an input(): the terminal as we found it
            self.app.enter_tui()
            self.app.exit_tui()
        self.assertEqual(calls, ["raw", "cooked", "raw", "cooked"])

    def test_raw_mode_is_left_alone_when_stdin_is_not_a_terminal(self):
        with mock.patch.object(containerctl.sys, "stdin", io.StringIO()):
            Term.raw_on()
            self.assertIsNone(Term._cooked)
            Term.raw_off()

    def test_ctrl_c_quits_and_restores_the_terminal(self):
        with (
            mock.patch.object(containerctl.signal, "signal"),
            mock.patch.object(self.app.fetcher, "start"),
            mock.patch.object(self.app.fetcher, "stop") as stop,
            mock.patch.object(Term, "getch", side_effect=KeyboardInterrupt),
            redirect_stdout(io.StringIO()) as out,
        ):
            self.app.run()  # no traceback: it returns
        stop.assert_called_once()
        self.assertTrue(out.getvalue().endswith(Term.SHOW_CURSOR + Term.MAIN_SCREEN))

    def test_run_restores_terminal_even_if_rendering_raises(self):
        with (
            mock.patch.object(containerctl.signal, "signal"),
            mock.patch.object(self.app.fetcher, "start"),
            mock.patch.object(self.app.fetcher, "stop") as stop,
            mock.patch.object(self.app, "render", side_effect=RuntimeError("boom")),
            redirect_stdout(io.StringIO()) as out,
        ):
            with self.assertRaises(RuntimeError):
                self.app.run()
        stop.assert_called_once()
        self.assertTrue(out.getvalue().endswith(Term.SHOW_CURSOR + Term.MAIN_SCREEN))


class DataFetcherTests(unittest.TestCase):
    def setUp(self):
        reset_errors()
        self.fetcher = DataFetcher(interval=0.01)
        self.containers = [
            container("cctl-prj-demo", id_="abc"),
            container("cctl-svc-web", "exited", id_="def", image="n"),
            container("other", id_="ghi", image="x"),
        ]
        self.projects = {"demo": ProjectConfig(name="demo", image="i")}
        self.services = {"web": ServiceConfig(name="web", image="n")}

    def _patches(self, projects=None, services=None, broken=(), failed=None, docker=None):
        """Config as read_* hands it back, docker as Docker.listings() does. `failed` maps listing keys to stderr;
        `docker` replaces the listings call altogether."""
        projects = self.projects if projects is None else projects
        services = self.services if services is None else services
        return (
            mock.patch.object(Docker, "listings", docker or (lambda: listings(failed, containers=self.containers))),
            mock.patch.object(containerctl, "read_projects", return_value=(projects, list(broken))),
            mock.patch.object(containerctl, "read_services", return_value=(services, [])),
            mock.patch.object(containerctl, "read_images", return_value=({}, [])),
            mock.patch.object(containerctl, "PROJECTS_DIR", Path("/nonexistent/projects")),
            mock.patch.object(containerctl, "SERVICES_DIR", Path("/nonexistent/services")),
            mock.patch.object(containerctl, "SECRETS_FILE", Path("/nonexistent/secrets.toml")),  # never the real one
        )

    def _fetch(self, **kwargs):
        patches = self._patches(**kwargs)
        for patch in patches:
            patch.start()
        try:
            self.fetcher._fetch_all()
        finally:
            for patch in patches:
                patch.stop()

    def log(self):
        return [(e.cat, e.msg) for e in containerctl.get_errors()]

    def test_docker_down_and_a_broken_config_are_issues_while_they_last(self):
        broken = [Issue(ERROR, "project.config", "Failed to load p.toml: bad toml", "edit p.toml")]
        down = dict.fromkeys(containerctl.LISTINGS, DOWN)
        self._fetch(broken=broken, failed=down)
        self.assertEqual(
            [(i.level, i.cat, i.msg) for i in self.fetcher.get("issues")],
            [
                (ERROR, "docker", "Docker daemon is not running (containers, images, volumes, networks)"),
                (ERROR, "project.config", "Failed to load p.toml: bad toml"),
            ],
        )
        self.assertEqual(
            self.fetcher.get_failures(), dict.fromkeys(containerctl.LISTINGS, "Docker daemon is not running")
        )
        self.assertEqual(self.fetcher.get("containers"), [])
        self._fetch()  # docker is back, the file is fixed: gone by themselves, nothing to clear
        self.assertEqual(self.fetcher.get("issues"), [])
        self.assertEqual(self.fetcher.get_failures(), {})
        self.assertEqual(len(self.fetcher.get("containers")), 3)

    def test_what_a_fetch_finds_wrong_is_state_and_never_reaches_the_error_log(self):
        """Docker down, a failed listing, a broken config: issues for as long as they last, even ten seconds.
        The stream of errors is for what happened once."""
        down = dict.fromkeys(containerctl.LISTINGS, DOWN)
        broken = [Issue(ERROR, "project.config", "Failed to load p.toml: bad toml", "edit p.toml")]
        for failed in (down, down, {**down, "volumes": "plugin exploded"}, None, {"images": DOWN}):
            self._fetch(failed=failed, broken=broken)
        self.assertEqual(self.log(), [])
        docker = self.fetcher.get("issues")[0]
        self.assertEqual(
            (docker.msg, docker.detail), ("Docker daemon is not running (images)", f"docker images: {DOWN}")
        )

    def test_a_partial_failure_blanks_its_panel_only(self):
        self._fetch(failed={"volumes": "plugin exploded"})
        self.assertEqual(self.fetcher.get_failures(), {"volumes": "Unknown docker failure"})
        self.assertEqual(len(self.fetcher.get("containers")), 3)
        self.assertEqual(
            [(i.msg, i.hint) for i in self.fetcher.get("issues")],
            [("Unknown docker failure (volumes)", "its details have what docker said")],
        )
        self.assertEqual(self.fetcher.get("issues")[0].detail, "docker volumes: plugin exploded")  # to teach from
        self.assertEqual(self.log(), [])

    def test_an_unreadable_row_is_a_warning_while_it_lasts(self):
        def docker():
            result = listings(containers=self.containers)
            result["containers"].skipped.append("docker ps: Missing fields: ['ID'] in {}")
            return result

        for _ in range(3):
            self._fetch(docker=docker)
        self.assertEqual(
            [(i.level, i.cat, i.msg, i.detail) for i in self.fetcher.get("issues")],
            [
                (
                    WARNING,
                    "docker.rows",
                    "1 unreadable row left out of the containers listing",
                    "docker ps: Missing fields: ['ID'] in {}",
                )
            ],
        )
        self.assertEqual(len(self.fetcher.get("containers")), 3)  # the rows that are fine still list
        self.assertEqual(self.fetcher.get_failures(), {})
        self.assertEqual(self.fetcher.get_notes(), {"containers": "1 unreadable row left out: see Issues"})
        self.assertEqual(self.log(), [])
        self._fetch()
        self.assertEqual((self.fetcher.get("issues"), self.fetcher.get_notes()), ([], {}))

    def test_projects_and_services_still_list_from_config_when_ps_failed(self):
        self._fetch(failed={"containers": DOWN})
        self.assertEqual([(p.Name, p.Status) for p in self.fetcher.get("projects")], [("demo", "unknown")])
        self.assertEqual([(s.Name, s.Status) for s in self.fetcher.get("services")], [("web", "unknown")])

    def test_a_crashing_fetch_blanks_the_view_and_says_why(self):
        self._fetch()
        self.assertTrue(self.fetcher.get("containers"))
        crashes = (
            PermissionError(13, "Permission denied"),  # docker on PATH but not executable
            OSError(8, "Exec format error"),  # the wrong binary at that path
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),  # output that isn't text
        )
        for exc in crashes:
            with self.subTest(exc=exc):
                reason = f"{type(exc).__name__}: {exc}"
                self._fetch(docker=mock.Mock(side_effect=exc))
                self._fetch(docker=mock.Mock(side_effect=exc))  # again: a bug of ours is loud every tick
                self.assertEqual(self.fetcher.get("containers"), [])  # blank, not stale
                self.assertEqual(
                    [(i.level, i.cat, i.msg) for i in self.fetcher.get("issues")],
                    [(ERROR, "fetcher", f"Fetch crashed: {reason}")],
                )
                panels = ["containers", "images", "volumes", "networks", "projects", "services"]
                self.assertEqual(self.fetcher.get_failures(), dict.fromkeys(panels, f"Fetch crashed: {reason}"))
                self.assertEqual(self.log()[-1], ("fetcher", reason))
        self.assertEqual(len(self.log()), 2 * len(crashes))

    def test_the_next_good_fetch_refills_the_view_and_the_log_keeps_the_crash(self):
        self._fetch(docker=mock.Mock(side_effect=PermissionError(13, "Permission denied")))
        self.assertEqual([cat for cat, _ in self.log()], ["fetcher"])
        self._fetch()
        self.assertTrue(self.fetcher.get("containers"))
        self.assertEqual((self.fetcher.get("issues"), self.fetcher.get_failures()), ([], {}))
        self.assertEqual([cat for cat, _ in self.log()], ["fetcher"])  # until cleared, like any error

    def test_a_crashing_fetch_does_not_kill_the_background_thread(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise PermissionError(13, "Permission denied")
            return listings(containers=self.containers)

        patches = self._patches(docker=flaky)
        for patch in patches:
            patch.start()
        try:
            self.fetcher.start()  # the first, blocking fetch crashes...
            self.assertEqual([cat for cat, _ in self.log()], ["fetcher"])
            deadline = time.monotonic() + 2
            while not self.fetcher.get("containers") and time.monotonic() < deadline:
                time.sleep(0.01)
            self.fetcher.stop()
        finally:
            for patch in patches:
                patch.stop()
        self.assertTrue(self.fetcher.get("containers"))  # ...and the thread went on to the next, fine one
        self.assertEqual(self.fetcher.get("issues"), [])

    def test_issues_are_detected_anew_each_fetch_errors_first(self):
        projects = {
            "demo": ProjectConfig(name="demo", image="i"),
            "lib": ProjectConfig(name="lib", image="i"),
            "a": ProjectConfig(name="a", image="i", shortcuts={"lib": "lib"}),
        }
        broken = [Issue(ERROR, "service.config", "Failed to load s.toml", "edit s.toml")]
        self._fetch(projects=projects, broken=broken)
        self.assertEqual([i.cat for i in self.fetcher.get("issues")], ["service.config", "project.shortcut"])
        self._fetch()  # fixed: the issues are simply gone, nothing to clear
        self.assertEqual(self.fetcher.get("issues"), [])
        self.assertEqual(self.log(), [])  # and none of it was ever an error

    def test_a_container_whose_config_is_gone_is_an_orphan(self):
        self._fetch(projects={}, services={})
        self.assertEqual(
            [(i.level, i.cat) for i in self.fetcher.get("issues")],
            [(WARNING, "project.orphan"), (WARNING, "service.orphan")],
        )

    def test_containers_get_project_column_from_name_prefix(self):
        self._fetch()
        by_name = {c.Names: c.Project for c in self.fetcher.get("containers")}
        self.assertEqual(by_name, {"cctl-prj-demo": "demo", "cctl-svc-web": "", "other": ""})

    def test_projects_status_from_matching_container(self):
        projects = {
            "demo": ProjectConfig(name="demo", image="i", repos=[containerctl.RepoSpec("https://x/r.git")]),
            "new": ProjectConfig(name="new", image="j"),
        }
        self._fetch(projects=projects)
        rows = {p.Name: p for p in self.fetcher.get("projects")}
        self.assertEqual(rows["demo"].Status, "running")
        self.assertEqual(rows["demo"].Repo, "https://x/r.git")
        self.assertEqual(rows["new"].Status, "not created")

    def test_services_status_and_ports_summary(self):
        services = {"web": ServiceConfig(name="web", image="n", ports=["1:1", "2:2", "3:3", "4:4", "5:5"])}
        self._fetch(services=services)
        row = self.fetcher.get("services")[0]
        self.assertEqual(row.Status, "exited")
        self.assertEqual(row.Ports, "1:1, 2:2, 3:3 +2")

    def test_get_returns_a_copy(self):
        self._fetch()
        self.fetcher.get("containers").clear()
        self.assertEqual(len(self.fetcher.get("containers")), 3)
        self.assertEqual(self.fetcher.get("nope"), [])
        self.fetcher.get_failures()["x"] = "y"
        self.assertEqual(self.fetcher.get_failures(), {})

    def test_background_thread_refreshes_and_stops(self):
        calls = []
        with mock.patch.object(self.fetcher, "_fetch_all", side_effect=lambda: calls.append(1)):
            self.fetcher.start()  # one blocking initial fetch
            self.fetcher._stop_event.wait(0.1)  # let the thread tick a few times
            self.fetcher.stop()
        self.assertGreaterEqual(len(calls), 2)
        self.assertFalse(self.fetcher._thread.is_alive())


if __name__ == "__main__":
    unittest.main()

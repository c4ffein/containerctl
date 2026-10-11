"""Tests for the hand-rolled TUI: Screen buffering, UI widgets, App state and key handling.

No terminal: ``Term.size`` is pinned, ``Term.getch`` is replaced by a scripted key feed,
``Screen.flush`` output is captured, and Docker calls go through the usual recorders/mocks.
Rendered text is compared after stripping ANSI escapes.
"""

import io
import re
import unittest
from contextlib import redirect_stdout
from unittest import mock

import containerctl
from containerctl import UI, App, DataFetcher, Docker, ProjectConfig, Screen, ServiceConfig, Tab, Term, log_error
from tests.helper import RunRecorder, reset_errors

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def plain(text: str) -> str:
    return ANSI.sub("", text)


def buffered(scr: Screen) -> str:
    return plain("".join(scr.buf))


def keys(*seq):
    """A ``Term.getch`` stand-in that replays ``seq`` then fails loudly if asked for more."""
    it = iter(seq)

    def getch():
        try:
            return next(it)
        except StopIteration:  # pragma: no cover - a test asked for more keys than scripted
            raise AssertionError("getch called more times than keys scripted")

    return getch


def tab(columns, id_key="ID", key="containers"):
    return Tab(name="T", key=key, fetch=lambda: [], columns=columns, id_key=id_key, actions={})


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

    def test_tab_bar_paints_errors_tab_red_only_when_errors_exist(self):
        errors_tab = tab([], key="errors")
        errors_tab.name = "Errors"
        UI.tab_bar(self.scr, [errors_tab], 0, 80, 1)
        self.assertNotIn(Term.RED, "".join(self.scr.buf))
        log_error("x", "boom")
        self.scr.clear()
        UI.tab_bar(self.scr, [errors_tab], 0, 80, 1)
        self.assertIn(Term.RED, "".join(self.scr.buf))

    def test_list_header_flex_column_takes_remaining_width(self):
        t = tab([("a", "AAAA", 6), ("b", "BBBBBBBBBBBBBBBBBBBB", 0)])
        UI.list_header(self.scr, t, 3, 20)
        text = buffered(self.scr)
        # width 20 -> 18 usable; 6 for A + 1 space; flex gets 11 and the header is clipped to it
        self.assertEqual(text, "AAAA   BBBBBBBBBBB ")

    def test_list_row_clips_values_and_inverts_selection(self):
        t = tab([("Names", "NAME", 4), ("Status", "STATUS", 0)])
        item = {"Names": "longname", "Status": "Up 3 hours"}
        UI.list_row(self.scr, t, item, 4, 30, selected=False)
        self.assertEqual(buffered(self.scr), "long Up 3 hours")
        self.assertNotIn(Term.INVERT, "".join(self.scr.buf))
        self.scr.clear()
        UI.list_row(self.scr, t, item, 4, 30, selected=True)
        self.assertIn(Term.INVERT, "".join(self.scr.buf))

    def test_list_row_colors_matching_status(self):
        t = tab([("Names", "NAME", 10), ("Status", "STATUS", 10)])
        colors = {"exited": Term.RED, "up": Term.GREEN}
        UI.list_row(self.scr, t, {"Names": "up-ish", "Status": "Exited (0)"}, 4, 30, False, colors)
        raw = "".join(self.scr.buf)
        self.assertIn(Term.RED + "Exited (0)" + Term.RESET, raw)
        self.assertNotIn(Term.GREEN, raw)  # only the Status column is coloured, even if NAME matches a key

    def test_list_row_no_color_when_status_matches_nothing(self):
        t = tab([("Status", "STATUS", 10)])
        UI.list_row(self.scr, t, {"Status": "Weird"}, 4, 30, False, {"up": Term.GREEN})
        self.assertNotIn(Term.GREEN, "".join(self.scr.buf))

    def test_render_colors_container_status(self):
        # end-to-end through App.render: the colour map wired for the containers tab must reach the row
        with mock.patch.object(Term, "size", return_value=(24, 80)):
            app = App()
            app.fetcher.data["containers"] = [{"ID": "a", "Names": "c", "Image": "i", "Status": "Exited (1)"}]
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
        self.app.fetcher.data["containers"] = [
            {"ID": f"id{i:010d}", "Names": f"c{i}", "Image": "img", "Status": "Up", "Project": ""} for i in range(n)
        ]


class AppItemsTests(AppBase):
    def test_tabs_are_in_documented_order(self):
        self.assertEqual(
            [t.key for t in self.app.tabs],
            ["containers", "images", "volumes", "networks", "projects", "services", "errors"],
        )

    def test_items_come_from_fetcher_cache(self):
        self.containers(2)
        self.assertEqual([c["Names"] for c in self.app.items()], ["c0", "c1"])

    def test_items_cache_copy_is_not_the_live_list(self):
        self.containers(1)
        self.app.items().append({"ID": "x"})
        self.assertEqual(len(self.app.fetcher.data["containers"]), 1)

    def test_projects_and_services_show_placeholder_when_empty(self):
        self.goto("projects")
        self.assertEqual(self.app.items()[0]["Name"], "No projects configured")
        self.assertIsNone(self.app.selected_id() and None)  # placeholder has a Name, see next assert
        self.assertEqual(self.app.selected_id(), "No projects configured")
        self.goto("services")
        self.assertEqual(self.app.items()[0]["Name"], "No services configured")

    def test_errors_tab_reads_registry_live(self):
        self.goto("errors")
        self.assertEqual(self.app.items()[0]["Category"], "No errors")
        self.assertEqual(self.app.selected_id(), "")  # placeholder carries an empty _category
        log_error("docker.command", "docker ps")
        self.assertEqual(self.app.selected_id(), "docker.command")

    def test_selected_id_uses_each_tabs_id_key(self):
        self.containers(1)
        self.assertEqual(self.app.selected_id(), "id0000000000")
        self.goto("volumes")
        self.app.fetcher.data["volumes"] = [{"Name": "vol", "Driver": "local"}]
        self.assertEqual(self.app.selected_id(), "vol")

    def test_selected_item_none_when_out_of_range(self):
        self.containers(1)
        self.app.selected_index = 5
        self.assertIsNone(self.app.selected_item())
        self.assertIsNone(self.app.selected_id())


class AppRenderTests(AppBase):
    def test_render_shows_tabs_header_rows_and_status_bar(self):
        self.containers(2)
        text = self.rendered()
        for name in ("Containers", "Images", "Errors"):
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
            "errors": "[c]lear",
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
        self.assertEqual(self.app.current().key, "errors")
        self.feed("1")
        self.assertEqual(self.app.current().key, "containers")

    def test_action_keys_without_selection_are_noops(self):
        rec = RunRecorder()
        with mock.patch.object(Docker, "_run", rec):
            for key in "sSrdeLo":
                self.feed(key)
        self.assertEqual(rec.calls, [])


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
        self.app.fetcher.data["images"] = [{"ID": "sha", "Repository": "nginx", "Tag": "1.27", "Size": "1MB"}]
        with mock.patch.object(UI, "confirm", return_value=True) as confirm:
            self.feed("d")
        confirm.assert_called_once_with("Delete image 'nginx:1.27'?", self.ROWS - 1)
        self.assertEqual(self.rec.last, ["rmi", "-f", "sha"])

    def test_volume_delete(self):
        self.goto("volumes")
        self.app.fetcher.data["volumes"] = [{"Name": "vol", "Driver": "local", "Mountpoint": "/v"}]
        with mock.patch.object(UI, "confirm", return_value=True):
            self.feed("d")
        self.assertEqual(self.rec.last, ["volume", "rm", "vol"])

    def test_network_edit_lists_connected_containers(self):
        self.goto("networks")
        self.app.fetcher.data["networks"] = [{"ID": "net1", "Name": "cctl-net-x", "Driver": "bridge", "Scope": "local"}]
        info = mock.Mock(Name="cctl-net-x", Containers={"abc": {"Name": "cctl-prj-x"}})
        with mock.patch.object(Docker, "inspect_network", return_value=info):
            with mock.patch.object(UI, "info_box") as box:
                self.feed("e")
        box.assert_called_once_with("Network: cctl-net-x", ["cctl-prj-x"], self.ROWS, self.COLS)

    def test_network_edit_failure_sets_red_message(self):
        self.goto("networks")
        self.app.fetcher.data["networks"] = [{"ID": "net1", "Name": "n", "Driver": "bridge", "Scope": "local"}]
        with mock.patch.object(Docker, "inspect_network", return_value=None):
            self.feed("e")
        self.assertEqual(self.app.message_style, Term.RED)

    def test_project_enter_uses_loaded_config(self):
        self.goto("projects")
        self.app.fetcher.data["projects"] = [{"Name": "demo", "Status": "running", "Image": "i", "Repo": "-"}]
        project = ProjectConfig(name="demo", image="i")
        with (
            mock.patch.object(containerctl, "load_projects", return_value={"demo": project}),
            mock.patch.object(self.app, "exit_tui_temporarily"),
            mock.patch.object(Docker, "enter_project") as enter,
            mock.patch.object(self.app, "enter_tui"),
        ):
            self.feed("e")
        enter.assert_called_once_with(project)

    def test_project_destroy_without_container_warns_instead_of_asking(self):
        self.goto("projects")
        self.app.fetcher.data["projects"] = [{"Name": "demo", "Status": "not created", "Image": "i", "Repo": "-"}]
        project = ProjectConfig(name="demo", image="i")
        with (
            mock.patch.object(containerctl, "load_projects", return_value={"demo": project}),
            mock.patch.object(Docker, "find_project_container", return_value=None),
            mock.patch.object(UI, "confirm") as confirm,
        ):
            self.feed("d")
        confirm.assert_not_called()
        self.assertEqual(self.app.message_style, Term.YELLOW)

    def test_project_destroy_confirmed(self):
        self.goto("projects")
        self.app.fetcher.data["projects"] = [{"Name": "demo", "Status": "running", "Image": "i", "Repo": "-"}]
        project = ProjectConfig(name="demo", image="i")
        with (
            mock.patch.object(containerctl, "load_projects", return_value={"demo": project}),
            mock.patch.object(Docker, "find_project_container", return_value={"State": "running"}),
            mock.patch.object(UI, "confirm", return_value=True),
            mock.patch.object(Docker, "destroy_project", return_value=True) as destroy,
        ):
            self.feed("d")
        destroy.assert_called_once_with(project)
        self.assertEqual(self.app.message_text, "Destroyed demo")

    def test_service_down_confirmed(self):
        self.goto("services")
        self.app.fetcher.data["services"] = [{"Name": "web", "Status": "running", "Image": "i", "Ports": ""}]
        service = ServiceConfig(name="web", image="i")
        with (
            mock.patch.object(containerctl, "load_services", return_value={"web": service}),
            mock.patch.object(Docker, "find_service_container", return_value={"State": "running"}),
            mock.patch.object(UI, "confirm", return_value=True) as confirm,
            mock.patch.object(Docker, "service_down", return_value=True) as down,
        ):
            self.feed("d")
        confirm.assert_called_once_with("Stop service 'web'?", self.ROWS - 1)
        down.assert_called_once_with(service)

    def test_service_recreate_declined_does_nothing(self):
        self.goto("services")
        self.app.fetcher.data["services"] = [{"Name": "web", "Status": "running", "Image": "i", "Ports": ""}]
        with (
            mock.patch.object(
                containerctl, "load_services", return_value={"web": ServiceConfig(name="web", image="i")}
            ),
            mock.patch.object(UI, "confirm", return_value=False),
            mock.patch.object(Docker, "service_recreate") as recreate,
        ):
            self.feed("r")
        recreate.assert_not_called()

    def test_errors_clear_selected_category_after_confirm(self):
        log_error("a.cat", "1")
        log_error("a.cat", "2")
        log_error("b.cat", "3")
        self.goto("errors")
        with mock.patch.object(UI, "confirm", return_value=True) as confirm:
            self.feed("c")
        confirm.assert_called_once_with("Clear all 'a.cat' errors?", self.ROWS - 1)
        self.assertEqual([e["cat"] for e in containerctl.get_errors()], ["b.cat"])
        self.assertEqual(self.app.message_text, "Cleared 2 errors from a.cat")

    def test_errors_clear_ignores_placeholder(self):
        self.goto("errors")
        with mock.patch.object(UI, "confirm") as confirm:
            self.feed("c")
        confirm.assert_not_called()


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
            {"ID": "abc", "Names": "cctl-prj-demo", "Image": "i", "Status": "Up", "State": "running"},
            {"ID": "def", "Names": "cctl-svc-web", "Image": "n", "Status": "Exited", "State": "exited"},
            {"ID": "ghi", "Names": "other", "Image": "x", "Status": "Up", "State": "running"},
        ]

    def _fetch(self, projects=None, services=None):
        with (
            mock.patch.object(Docker, "containers", return_value=[dict(c) for c in self.containers]),
            mock.patch.object(Docker, "images", return_value=[]),
            mock.patch.object(Docker, "volumes", return_value=[]),
            mock.patch.object(Docker, "networks", return_value=[]),
            mock.patch.object(containerctl, "load_projects", return_value=projects or {}),
            mock.patch.object(containerctl, "load_services", return_value=services or {}),
        ):
            self.fetcher._fetch_all()

    def test_containers_get_project_column_from_name_prefix(self):
        self._fetch()
        by_name = {c["Names"]: c["Project"] for c in self.fetcher.get("containers")}
        self.assertEqual(by_name, {"cctl-prj-demo": "demo", "cctl-svc-web": "", "other": ""})

    def test_projects_status_from_matching_container(self):
        projects = {
            "demo": ProjectConfig(name="demo", image="i", repos=[containerctl.RepoSpec("https://x/r.git")]),
            "new": ProjectConfig(name="new", image="j"),
        }
        self._fetch(projects=projects)
        rows = {p["Name"]: p for p in self.fetcher.get("projects")}
        self.assertEqual(rows["demo"]["Status"], "running")
        self.assertEqual(rows["demo"]["Repo"], "https://x/r.git")
        self.assertEqual(rows["new"]["Status"], "not created")

    def test_services_status_and_ports_summary(self):
        services = {"web": ServiceConfig(name="web", image="n", ports=["1:1", "2:2", "3:3", "4:4", "5:5"])}
        self._fetch(services=services)
        row = self.fetcher.get("services")[0]
        self.assertEqual(row["Status"], "exited")
        self.assertEqual(row["Ports"], "1:1, 2:2, 3:3 +2")

    def test_get_returns_a_copy(self):
        self._fetch()
        self.fetcher.get("containers").clear()
        self.assertEqual(len(self.fetcher.get("containers")), 3)
        self.assertEqual(self.fetcher.get("nope"), [])

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

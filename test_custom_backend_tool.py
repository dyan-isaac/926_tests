import pytest
from unittest.mock import Mock, MagicMock
from matplotlib.backend_tools import *
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent


# ---------- ToolBase and ToolToggleBase ----------

class TestToolBaseToggle:
    def test_toolbase_name_toolmanager_figure(self):
        tm = Mock()
        tool = ToolBase(tm, "tool")
        fig = Figure()
        tool.set_figure(fig)
        assert tool.name == "tool"
        assert tool.toolmanager is tm
        assert tool.figure == fig

    def test_tooltoggle_toggle_behavior(self):
        tool = ToolToggleBase(Mock(), "toggle")
        assert not tool.toggled
        tool.trigger(None, None)
        assert tool.toggled
        tool.trigger(None, None)
        assert not tool.toggled

    def test_tooltoggle_radio_group(self):
        tool = ToolToggleBase(Mock(), "toggle")
        assert tool.radio_group is None

    def test_tooltoggle_cursor_property(self):
        tool = ToolToggleBase(Mock(), "toggle")
        assert tool.cursor is None


# ---------- View Stack Tools ----------

class TestViewsPositionTools:
    def setup_method(self):
        self.tool = ToolViewsPositions(Mock(), "viewpos")
        self.fig = Figure()
        self.tool.set_figure(self.fig)
        self.tool.add_figure(self.fig)

    def test_push_and_home(self):
        self.tool.push_current()
        self.tool.home()
        assert self.fig in self.tool.views
        assert self.fig in self.tool.positions

    def test_back_forward_flow(self):
        self.tool.push_current()
        self.tool.back()
        self.tool.forward()

    # def test_clear_resets_stack(self):
    #     self.tool.clear(self.fig)
    #     assert self.tool.views[self.fig].elements == []

    def test_update_view_calls_draw_idle(self):
        self.tool.push_current()
        canvas = self.fig.canvas = MagicMock()
        self.tool.update_view()
        canvas.draw_idle.assert_called()


# ---------- Zoom and Pan Tool Behavior ----------

class TestZoomPanToolBehavior:
    def test_zoom_scroll_behavior(self):
        tool = ToolZoom(Mock(), "zoom")
        fig = Figure()
        tool.set_figure(fig)
        event = Mock()
        event.inaxes = fig.add_subplot()
        event.button = "up"
        event.x, event.y = 100, 100
        tool.scroll_zoom(event)

    def test_pan_mouse_press_release(self):
        tool = ToolPan(Mock(), "pan")
        fig = Figure()
        ax = fig.add_subplot()
        ax.can_pan = lambda: True
        ax.start_pan = lambda *a: None
        ax.end_pan = lambda: None
        fig.canvas = MagicMock()
        tool.set_figure(fig)
        event = Mock()
        event.button = 1
        event.x, event.y = 10, 10
        event.inaxes = ax
        ax.in_axes = lambda e: True
        tool._press(event)
        tool._release(event)


# ---------- Cursor and Pointer Tools ----------

class TestPointerCursorTools:
    def test_cursor_position_message(self):
        tm = Mock()
        tm.messagelock.locked.return_value = False
        tm.message_event = Mock()
        tool = ToolCursorPosition(tm, "cursor")
        fig = Figure()
        tool.set_figure(fig)
        tool.send_message(Mock(x=1, y=1, inaxes=None))
        tm.message_event.assert_called_once()

    # def test_set_cursor_triggers_event_connection(self):
    #     tm = Mock()
    #     tool = ToolSetCursor(tm, "cursor_set")
    #     fig = Figure()
    #     fig.canvas = MagicMock()
    #     tool.set_figure(fig)
    #     assert tool._id_drag is not None


# ---------- Basic Trigger Tools ----------

class TestSimpleToolTriggers:
    def test_tool_quit(self):
        fig = Figure()
        tool = ToolQuit(Mock(), "quit")
        tool.set_figure(fig)
        tool.trigger(None, None)

    def test_tool_quit_all(self):
        ToolQuitAll(Mock(), "quit_all").trigger(None, None)

    # def test_tool_grid(self):
    #     tool = ToolGrid(Mock(), "grid")
    #     fig = Figure()
    #     tool.set_figure(fig)
    #     event = Mock()
    #     event.key = "g"
    #     tool.trigger(None, event)
    #
    # def test_tool_minor_grid(self):
    #     tool = ToolMinorGrid(Mock(), "minor_grid")
    #     fig = Figure()
    #     tool.set_figure(fig)
    #     event = Mock()
    #     event.key = "G"
    #     tool.trigger(None, event)

    def test_tool_fullscreen(self):
        fig = Figure()
        fig.canvas.manager = MagicMock()
        tool = ToolFullScreen(Mock(), "fullscreen")
        tool.set_figure(fig)
        tool.trigger(None, None)

    def test_tool_help_text(self):
        tm = Mock()
        tm.get_tool_keymap.return_value = ["ctrl+h"]
        tool = ToolHelpBase(tm, "help")
        tm.tools = {"help": tool}
        tool.description = "Help tool"
        txt = tool._get_help_text()
        html = tool._get_help_html()
        assert "Help tool" in txt
        assert "<table>" in html


# ---------- Tool Manager and Container ----------

class TestToolUtilityFunctions:
    def test_add_tools_to_manager(self):
        manager = Mock()
        add_tools_to_manager(manager)
        assert manager.add_tool.call_count >= len(default_tools)

    def test_add_tools_to_container(self):
        container = Mock()
        add_tools_to_container(container)
        assert container.add_tool.call_count >= 6

    def test_toolcopy_message(self):
        tm = Mock()
        tm.message_event = Mock()
        tool = ToolCopyToClipboardBase(tm, "copy")
        tool.trigger()
        tm.message_event.assert_called_once()

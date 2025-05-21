import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import pytest
from unittest import mock
from matplotlib.testing.widgets import click_and_drag
from numpy.testing import assert_allclose
import numpy as np


def test_range_slider_reset():
    fig, ax = plt.subplots()
    slider = widgets.RangeSlider(ax=ax, label='', valmin=0.0, valmax=10.0, valinit=(2.0, 8.0))
    slider.set_val((3.0, 7.0))
    slider.reset()
    assert slider.val == (2.0, 8.0)


def test_textbox_submit_event():
    fig, ax = plt.subplots()
    textbox = widgets.TextBox(ax, 'Test')
    on_submit = mock.Mock()
    textbox.on_submit(on_submit)
    textbox.set_val('New Value')
    assert on_submit.call_count == 1
    assert textbox.text == 'New Value'


def test_radio_buttons_clear():
    fig, ax = plt.subplots()
    radio = widgets.RadioButtons(ax, ('Option 1', 'Option 2', 'Option 3'))
    radio.set_active(1)
    radio.clear()
    assert radio.value_selected == 'Option 1'
    assert radio.index_selected == 0


def test_check_buttons_toggle():
    fig, ax = plt.subplots()
    check = widgets.CheckButtons(ax, ('Option 1', 'Option 2', 'Option 3'), (True, False, True))
    check.set_active(1)
    assert check.get_status() == [True, True, True]
    check.set_active(1)
    assert check.get_status() == [True, False, True]

def test_rectangle_selector_toggle_state():
    fig, ax = plt.subplots()
    tool = widgets.RectangleSelector(ax, lambda e1, e2: None, interactive=True)
    tool.add_state('move')
    assert 'move' in tool._state
    tool.remove_state('move')
    assert 'move' not in tool._state


def test_range_slider_set_val():
    fig, ax = plt.subplots()
    range_slider = widgets.RangeSlider(ax, label="Range Slider", valmin=0, valmax=10, valinit=(2, 8))

    # Set a new range
    range_slider.set_val((3, 7))
    assert range_slider.val == (3, 7)

    # Verify the range is within bounds
    range_slider.set_val((-1, 12))
    assert range_slider.val == (0, 10)  # Should snap to min and max values


def test_check_buttons_initial_state():
    fig, ax = plt.subplots()
    check_buttons = widgets.CheckButtons(ax, labels=["Option 1", "Option 2"], actives=[True, False])

    # Verify initial states
    assert check_buttons.get_status() == [True, False]


def test_radio_buttons_active_state():
    fig, ax = plt.subplots()
    radio_buttons = widgets.RadioButtons(ax, labels=["Option A", "Option B", "Option C"], active=1)

    # Verify initial active state
    assert radio_buttons.value_selected == "Option B"

    # Change active state
    radio_buttons.set_active(2)
    assert radio_buttons.value_selected == "Option C"


def test_textbox_text_change():
    fig, ax = plt.subplots()
    textbox = widgets.TextBox(ax, label="Input")
    on_change = mock.Mock()
    textbox.on_text_change(on_change)

    # Simulate text change
    textbox.set_val("New Text")
    assert textbox.text == "New Text"
    on_change.assert_called_once_with("New Text")


def test_cursor_visibility():
    fig, ax = plt.subplots()
    cursor = widgets.Cursor(ax, horizOn=True, vertOn=False)

    # Verify initial visibility
    assert cursor.visible is True

    # Change visibility
    cursor.visible = False
    assert cursor.visible is False

@pytest.mark.parametrize("valinit, new_val, expected", [
    (5, 7, 7),  # Normal case
])
def test_slider_set_val(valinit, new_val, expected):
    fig, ax = plt.subplots()
    slider = widgets.Slider(ax, label="Test Slider", valmin=0, valmax=10, valinit=valinit)

    # Set a new value
    slider.set_val(new_val)
    assert slider.val == expected


@pytest.mark.parametrize("valinit, new_range, expected", [
    ((2, 8), (3, 7), (3, 7)),  # Normal range
    ((2, 8), (-1, 12), (0, 10)),  # Exceeding bounds
    ((2, 8), (5, 5), (5, 5)),  # Single-point range
])
def test_range_slider_set_val(valinit, new_range, expected):
    fig, ax = plt.subplots()
    range_slider = widgets.RangeSlider(ax, label="Range Slider", valmin=0, valmax=10, valinit=valinit)

    # Set a new range
    range_slider.set_val(new_range)
    assert range_slider.val == expected


@pytest.mark.parametrize("active_index, expected", [
    (0, "Option 1"),  # First option
    (1, "Option 2"),  # Second option
    (2, "Option 3"),  # Third option
])
def test_radio_buttons_active_state(active_index, expected):
    fig, ax = plt.subplots()
    radio_buttons = widgets.RadioButtons(ax, labels=["Option 1", "Option 2", "Option 3"], active=active_index)

    # Verify active state
    assert radio_buttons.value_selected == expected


@pytest.mark.parametrize("initial_states, toggle_index, expected_states", [
    ([True, False, True], 1, [True, True, True]),  # Toggle off to on
    ([True, True, True], 1, [True, False, True]),  # Toggle on to off
    ([False, False, False], 0, [True, False, False]),  # Toggle off to on
])
def test_check_buttons_toggle(initial_states, toggle_index, expected_states):
    fig, ax = plt.subplots()
    check_buttons = widgets.CheckButtons(ax, labels=["Option 1", "Option 2", "Option 3"], actives=initial_states)

    # Toggle a button
    check_buttons.set_active(toggle_index)
    assert check_buttons.get_status() == expected_states


@pytest.mark.parametrize("text, expected", [
    ("Hello", "Hello"),  # Normal text
    ("", ""),  # Empty text
    ("12345", "12345"),  # Numeric text
])
def test_textbox_text_change(text, expected):
    fig, ax = plt.subplots()
    textbox = widgets.TextBox(ax, label="Input")
    textbox.set_val(text)

    # Verify text change
    assert textbox.text == expected
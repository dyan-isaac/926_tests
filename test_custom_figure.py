import pytest
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np


@pytest.mark.parametrize("figsize", [
    (4, 3),  # standard
    (0.1, 0.1),  # small
    (20, 15),  # large
    (10, 0.01),  # extreme aspect
    (0.01, 10)
])
def test_set_size_inches_edge_cases(figsize):
    fig = Figure()
    fig.set_size_inches(*figsize)
    assert fig.get_size_inches()[0] == pytest.approx(figsize[0])
    assert fig.get_size_inches()[1] == pytest.approx(figsize[1])


@pytest.mark.parametrize("dpi", [1, 72, 150, 300, 1200])
def test_dpi_setting_and_scaling(dpi):
    fig = Figure()
    fig.set_dpi(dpi)
    assert fig.get_dpi() == dpi


@pytest.mark.parametrize("frameon", [True, False])
def test_figure_background_visibility(frameon):
    fig = Figure(frameon=frameon)
    assert fig.get_frameon() is frameon


@pytest.mark.parametrize("color", ['red', '#00FF00', (0.5, 0.5, 0.5)])
def test_set_face_and_edge_color(color):
    fig = Figure()
    fig.set_facecolor(color)
    fig.set_edgecolor(color)
    assert fig.get_facecolor() == fig.patch.get_facecolor()
    assert fig.get_edgecolor() == fig.patch.get_edgecolor()


@pytest.mark.parametrize("rotation", [0, 45, 90, 120, 180])
def test_suptitle_rotation(rotation):
    fig = Figure()
    text = fig.suptitle("Title", rotation=rotation)
    assert text.get_rotation() == rotation


@pytest.mark.parametrize("input_type", [
    111,
    (1, 1, 1),
    (2, 2, 3),
    (3, 3, 3)
])
def test_add_subplot_various_formats(input_type):
    fig = Figure()
    if isinstance(input_type, tuple):
        ax = fig.add_subplot(*input_type)
    else:
        ax = fig.add_subplot(input_type)
    assert isinstance(ax, Axes)



def test_add_artist_directly():
    fig = Figure()
    from matplotlib.patches import Circle
    c = Circle((0.5, 0.5), 0.1)
    fig.add_artist(c)
    assert c in fig.artists


def test_autofmt_xdate_layout_adjustment():
    fig = Figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    fig.autofmt_xdate()
    assert fig.stale is True


@pytest.mark.parametrize("text, x, y", [
    ("label", 0.1, 0.1),
    ("", 0.5, 0.5),
    ("x" * 1000, 0.9, 0.9)
])
def test_text_addition(text, x, y):
    fig = Figure()
    txt = fig.text(x, y, text)
    assert txt.get_text() == text


@pytest.mark.parametrize("nrows,ncols", [(1, 1), (2, 2), (3, 1), (1, 3)])
def test_subplots_creation(nrows, ncols):
    fig = Figure()
    axs = fig.subplots(nrows=nrows, ncols=ncols)

    if nrows == 1 and ncols == 1:
        assert isinstance(axs, Axes)
    else:
        axs_array = np.array(axs).reshape((nrows, ncols))
        assert axs_array.shape == (nrows, ncols)




def test_clear_removes_axes():
    fig = Figure()
    fig.add_subplot(111)
    fig.clear()
    assert not fig.axes


@pytest.mark.parametrize("key", ["x", "y", "title"])
def test_align_label_groups_initialization(key):
    fig = Figure()
    assert key in fig._align_label_groups
    assert hasattr(fig._align_label_groups[key], 'get_siblings')


def test_get_children_includes_patch():
    fig = Figure()
    children = fig.get_children()
    assert fig.patch in children



def test_add_axes_invalid_type():
    fig = Figure()
    with pytest.raises(TypeError):
        fig.add_axes("not a rect")


@pytest.mark.parametrize("visible", [True, False])
def test_patch_visibility_toggle(visible):
    fig = Figure()
    fig.set_frameon(visible)
    assert fig.get_frameon() is visible

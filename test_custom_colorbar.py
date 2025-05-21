import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize, NoNorm, LogNorm
import pytest

# ------------------ Alpha Handling Tests ------------------

@pytest.mark.parametrize("alpha", [
    np.array([[0.1, 0.5], [0.9, 1.0]]),
    np.array([[0.0, 0.0], [1.0, 1.0]]),
    np.array([[0.5, 0.5], [0.5, 0.5]]),
])
def test_colorbar_set_alpha_array(alpha):
    fig, ax = plt.subplots()
    pc = ax.pcolormesh([[0, 1], [2, 3]], alpha=alpha)
    cb = fig.colorbar(pc)
    assert cb.alpha is None


# ------------------ Scale Setting Tests ------------------

@pytest.mark.parametrize("scale", ["log", "linear"])
def test_set_scale_log_and_linear(scale):
    fig, ax = plt.subplots()
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 100), cmap="viridis")
    cb = Colorbar(ax, sm, orientation="vertical")
    if scale == "log":
        cb._set_scale("log", base=10)
        assert cb.ax.yaxis.get_scale() == "log"
    else:
        cb._set_scale("linear")
        assert cb.ax.yaxis.get_scale() == "linear"


# ------------------ View Setting and Getting Tests ------------------

@pytest.mark.parametrize("new_view", [
    (0, 10),
    (-5, 5),
    (100, 200),
])
def test_get_and_set_view(new_view):
    fig, ax = plt.subplots()
    pc = ax.pcolormesh([[1, 2], [3, 4]])
    cb = fig.colorbar(pc)
    cb._set_view(new_view)
    assert cb.norm.vmin == new_view[0] and cb.norm.vmax == new_view[1]


# ------------------ NoNorm and Normalization Tests ------------------

def test_colorbar_nonnormalized_values():
    fig, ax = plt.subplots()
    norm = NoNorm()
    mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    cb = Colorbar(ax, mappable, orientation="vertical")
    assert cb.norm.vmin == 0 and cb.norm.vmax == 1


@pytest.mark.parametrize("vmin, vmax", [
    (0, 1),
    (-1, 1),
    (10, 20),
])
def test_colorbar_normalized_values(vmin, vmax):
    fig, ax = plt.subplots()
    norm = Normalize(vmin=vmin, vmax=vmax)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    cb = Colorbar(ax, mappable, orientation="vertical")
    assert cb.norm.vmin == vmin and cb.norm.vmax == vmax


# ------------------ Locator and Formatter Set/Get Tests ------------------

def test_locator_set_get():
    fig, ax = plt.subplots()
    pc = ax.pcolormesh(np.random.rand(5, 5))
    cb = fig.colorbar(pc)
    locator = cb.locator
    cb.locator = locator
    assert cb.locator is locator


def test_formatter_set_get():
    fig, ax = plt.subplots()
    pc = ax.pcolormesh(np.random.rand(5, 5))
    cb = fig.colorbar(pc)
    formatter = cb.formatter
    cb.formatter = formatter
    assert cb.formatter is formatter


# ------------------ Additional Edge Cases ------------------

@pytest.mark.parametrize("cmap", ["viridis", "plasma", "inferno", "magma", "cividis"])
def test_colorbar_different_cmaps(cmap):
    fig, ax = plt.subplots()
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    cb = Colorbar(ax, sm, orientation="vertical")
    assert cb.mappable.get_cmap().name == cmap


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_colorbar_orientation(orientation):
    fig, ax = plt.subplots()
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="viridis")
    cb = Colorbar(ax, sm, orientation=orientation)
    assert cb.orientation == orientation


@pytest.mark.parametrize("norm", [Normalize(0, 1), LogNorm(1, 10)])
def test_colorbar_with_various_norms(norm):
    fig, ax = plt.subplots()
    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    cb = Colorbar(ax, sm, orientation="vertical")
    assert isinstance(cb.norm, (Normalize, LogNorm))


@pytest.mark.parametrize("value_range", [
    np.linspace(0, 1, 5),
    np.linspace(10, 20, 10),
])
def test_colorbar_with_varied_data(value_range):
    fig, ax = plt.subplots()
    sm = plt.cm.ScalarMappable(norm=Normalize(value_range.min(), value_range.max()), cmap="viridis")
    cb = Colorbar(ax, sm, orientation="vertical")
    assert cb.norm.vmin == value_range.min()
    assert cb.norm.vmax == value_range.max()

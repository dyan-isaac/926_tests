import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize, NoNorm
import pytest


def test_colorbar_set_alpha_array():
    fig, ax = plt.subplots()
    alpha = np.array([[0.1, 0.5], [0.9, 1.0]])
    pc = ax.pcolormesh([[0, 1], [2, 3]], alpha=alpha)
    cb = fig.colorbar(pc)
    assert cb.alpha is None  # array alpha should trigger alpha=None


# def test_set_ticks_with_labels_minor():
#     fig, ax = plt.subplots()
#     pc = ax.pcolormesh(np.random.rand(10, 10))
#     cb = fig.colorbar(pc)
#     cb.set_ticks([0.2, 0.4, 0.6], labels=["low", "med", "high"], minor=True)
#     assert len(cb.ax.yaxis.get_minorticklocs()) > 0


def test_set_scale_log_and_linear():
    fig, ax = plt.subplots()
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 100), cmap="viridis")
    cb = Colorbar(ax, sm, orientation="vertical")
    cb._set_scale("log", base=10)
    assert cb.ax.yaxis.get_scale() == "log"
    cb._set_scale("linear")
    assert cb.ax.yaxis.get_scale() == "linear"


# def test_colorbar_cla_resets_state():
#     fig, ax = plt.subplots()
#     pc = ax.pcolormesh([[1, 2], [3, 4]])
#     cb = fig.colorbar(pc)
#     assert hasattr(cb.ax, "drag_pan")
#     cb.ax.cla()
#     assert not hasattr(cb.ax, "drag_pan")


def test_get_and_set_view():
    fig, ax = plt.subplots()
    pc = ax.pcolormesh([[1, 2], [3, 4]])
    cb = fig.colorbar(pc)
    original = cb._get_view()
    new_view = (0, 10)
    cb._set_view(new_view)
    assert cb.norm.vmin == 0 and cb.norm.vmax == 10


# def test_set_view_from_bbox():
#     fig, ax = plt.subplots()
#     pc = ax.pcolormesh([[1, 2], [3, 4]])
#     cb = fig.colorbar(pc)
#     bbox = cb.ax.bbox
#     cb._set_view_from_bbox(bbox)
#     assert isinstance(cb.norm.vmin, float)
#     assert isinstance(cb.norm.vmax, float)


# def test_drag_pan_updates_norm():
#     fig, ax = plt.subplots()
#     pc = ax.pcolormesh([[1, 2], [3, 4]])
#     cb = fig.colorbar(pc)
#     cb.ax._get_pan_points = lambda button, key, x, y: np.array([[0, 5], [0, 15]])
#     cb.drag_pan(button=1, key=None, x=0, y=0)
#     assert cb.norm.vmin == 5 and cb.norm.vmax == 15


def test_colorbar_nonnormalized_values():
    fig, ax = plt.subplots()
    norm = NoNorm()
    mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    cb = Colorbar(ax, mappable, orientation="vertical")
    assert cb.norm.vmin == 0 and cb.norm.vmax == 1


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

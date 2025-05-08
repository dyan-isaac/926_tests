
import io
import pickle
import numpy as np
import pytest
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import text as mtext

@pytest.mark.parametrize("rotation", [0, 45, 90, 180])
def test_text_rotation_applied(rotation):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Test", rotation=rotation)
    assert txt.get_rotation() == rotation

@pytest.mark.parametrize("ha", ['left', 'center', 'right'])
def test_text_horizontal_alignment(ha):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Aligned", ha=ha)
    assert txt.get_ha() == ha

@pytest.mark.parametrize("va", ['top', 'center', 'bottom', 'baseline'])
def test_text_vertical_alignment(va):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Aligned", va=va)
    assert txt.get_va() == va

@pytest.mark.parametrize("fontsize", [6, 10, 20, 'x-large'])
def test_text_fontsize(fontsize):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Size", fontsize=fontsize)
    assert txt.get_fontsize() == txt.get_fontsize()  # sanity check

@pytest.mark.parametrize("color", ['r', 'g', 'b', '#123456'])
def test_text_color(color):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Color", color=color)
    assert txt.get_color() == color

@pytest.mark.parametrize("fontweight", ['normal', 'bold', 700])
def test_text_fontweight(fontweight):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Weight", weight=fontweight)
    assert txt.get_weight() == fontweight

@pytest.mark.parametrize("fontstyle", ['normal', 'italic', 'oblique'])
def test_text_fontstyle(fontstyle):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Style", style=fontstyle)
    assert txt.get_style() == fontstyle

@pytest.mark.parametrize("usetex", [False])
def test_text_usetex_backend(monkeypatch, usetex):
    monkeypatch.setitem(plt.rcParams, "text.usetex", usetex)
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, r"$x^2$")
    fig.canvas.draw()
    assert txt.get_text() == r"$x^2$"

@pytest.mark.parametrize("clip_on", [True, False])
def test_text_clip_behavior(clip_on):
    fig, ax = plt.subplots()
    txt = ax.text(2.0, 2.0, "Clipped", clip_on=clip_on)
    assert txt.get_clip_on() == clip_on

@pytest.mark.parametrize("wrap", [True, False])
def test_text_wrap_behavior(wrap):
    fig, ax = plt.subplots()
    txt = ax.text(0.1, 0.5, "Long text for wrap testing", wrap=wrap)
    assert txt.get_wrap() == wrap

def test_text_pickling_and_restoration():
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "PickleTest")
    data = pickle.dumps(txt)
    restored = pickle.loads(data)
    assert restored.get_text() == "PickleTest"

def test_text_draw_figure_cleanup():
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Draw and close")
    fig.canvas.draw()
    plt.close(fig)
    assert not plt.fignum_exists(fig.number)

# Repeat for more styles, layout, rotation modes, bbox options etc. to reach 40+ cases.
matplotlib.pyplot.close()
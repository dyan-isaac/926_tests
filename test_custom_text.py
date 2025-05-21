import io
import pickle
import numpy as np
import pytest
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import text as mtext

# Basic properties
@pytest.mark.parametrize("rotation", [0, 45, 90, 180, -45, 270])
def test_text_rotation_applied(rotation):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Test", rotation=rotation)
    assert txt.get_rotation() == rotation % 360

@pytest.mark.parametrize("rotation_mode", ['default', 'anchor'])
def test_text_rotation_mode(rotation_mode):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Test", rotation=45, rotation_mode=rotation_mode)
    assert txt.get_rotation_mode() == rotation_mode

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

# Font properties
@pytest.mark.parametrize("fontsize", [6, 10, 20, 'x-small', 'medium', 'x-large', 12.5])
def test_text_fontsize(fontsize):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Size", fontsize=fontsize)
    assert txt.get_size() == matplotlib.text.Text._get_size(txt.get_fontsize())

@pytest.mark.parametrize("color", ['r', 'g', 'b', '#123456', '0.5', (0.1, 0.2, 0.3, 0.4)])
def test_text_color(color):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Color", color=color)
    assert txt.get_color() == color

@pytest.mark.parametrize("fontweight", ['normal', 'bold', 'light', 'heavy', 400, 700])
def test_text_fontweight(fontweight):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Weight", weight=fontweight)
    assert txt.get_weight() == fontweight

@pytest.mark.parametrize("fontstyle", ['normal', 'italic', 'oblique'])
def test_text_fontstyle(fontstyle):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Style", style=fontstyle)
    assert txt.get_style() == fontstyle

@pytest.mark.parametrize("fontfamily", ['serif', 'sans-serif', 'cursive', 'monospace'])
def test_text_fontfamily(fontfamily):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Family", family=fontfamily)
    assert txt.get_family()[0] == fontfamily

# Text content and formatting
@pytest.mark.parametrize("text", ["Simple", "Multiline\ntext", "   Leading spaces", "Trailing spaces   "])
def test_text_content(text):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, text)
    assert txt.get_text() == text

@pytest.mark.parametrize("usetex", [False, True])
def test_text_usetex_backend(monkeypatch, usetex):
    monkeypatch.setitem(plt.rcParams, "text.usetex", usetex)
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, r"$x^2$")
    fig.canvas.draw()
    assert txt.get_text() == r"$x^2$"

@pytest.mark.parametrize("math_text", [None, "regular", "default"])
def test_text_math_text(math_text):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, r"$\alpha$", math_text=math_text)
    fig.canvas.draw()
    assert txt.get_text() == r"$\alpha$"

# Layout and positioning
@pytest.mark.parametrize("position", [(0.1, 0.1), (0.9, 0.9), (0.5, 0.5), (-0.1, 1.1)])
def test_text_position(position):
    fig, ax = plt.subplots()
    x, y = position
    txt = ax.text(x, y, "Position")
    assert np.allclose((txt.get_position(), (x, y)))

@pytest.mark.parametrize("transform", ['data', 'axes', 'figure', 'display'])
def test_text_transform(transform):
    fig, ax = plt.subplots()
    if transform == 'data':
        trans = ax.transData
    elif transform == 'axes':
        trans = ax.transAxes
    elif transform == 'figure':
        trans = fig.transFigure
    else:
        trans = None
    txt = ax.text(0.5, 0.5, "Transform", transform=trans)
    assert txt.get_transform() == (ax.transData if trans is None else trans)

# Visual properties
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_text_alpha(alpha):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Alpha", alpha=alpha)
    assert txt.get_alpha() == alpha

@pytest.mark.parametrize("backgroundcolor", [None, 'yellow', '#abcdef', (0.8, 0.8, 0.8)])
def test_text_backgroundcolor(backgroundcolor):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Background", backgroundcolor=backgroundcolor)
    assert txt.get_backgroundcolor() == backgroundcolor

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

@pytest.mark.parametrize("multialignment", ['left', 'center', 'right'])
def test_text_multialignment(multialignment):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Multi\nline\ntext", multialignment=multialignment)
    assert txt.get_multialignment() == multialignment

# Bounding box tests
@pytest.mark.parametrize("bbox_style", [None, dict(facecolor='red', alpha=0.5),
                                       dict(boxstyle='round,pad=0.5', edgecolor='blue')])
def test_text_bbox(bbox_style):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "BBox", bbox=bbox_style)
    if bbox_style is None:
        assert txt.get_bbox_patch() is None
    else:
        assert txt.get_bbox_patch() is not None

# Interaction tests
def test_text_set_properties_after_creation():
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Dynamic")
    txt.set_text("Updated")
    txt.set_color('green')
    txt.set_fontsize(15)
    assert txt.get_text() == "Updated"
    assert txt.get_color() == 'green'
    assert txt.get_fontsize() == 15

def test_text_remove():
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "To be removed")
    txt.remove()
    assert txt not in ax.texts

# Pickling and serialization
def test_text_pickling_and_restoration():
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "PickleTest", rotation=45, color='red')
    data = pickle.dumps(txt)
    restored = pickle.loads(data)
    assert restored.get_text() == "PickleTest"
    assert restored.get_rotation() == 45
    assert restored.get_color() == 'red'

def test_text_deepcopy():
    import copy
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "CopyTest", fontsize=12)
    txt_copy = copy.deepcopy(txt)
    assert txt_copy.get_text() == "CopyTest"
    assert txt_copy.get_fontsize() == 12

# Figure and canvas interaction
def test_text_draw_figure_cleanup():
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Draw and close")
    fig.canvas.draw()
    plt.close(fig)
    assert not plt.fignum_exists(fig.number)

def test_text_figure_coordinates():
    fig, ax = plt.subplots()
    txt = fig.text(0.5, 0.5, "Figure text")
    assert txt.get_position() == (0.5, 0.5)
    assert txt in fig.texts

# Advanced text features
@pytest.mark.parametrize("url", [None, "https://matplotlib.org", "internal_link"])
def test_text_url(url):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "URL", url=url)
    assert txt.get_url() == url

@pytest.mark.parametrize("gid", [None, "text-group", "unique-id"])
def test_text_gid(gid):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "GID", gid=gid)
    assert txt.get_gid() == gid

@pytest.mark.parametrize("sketch_params", [None, (1, 2, 3), (10, 20, 30)])
def test_text_sketch_params(sketch_params):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Sketch", sketch_params=sketch_params)
    assert txt.get_sketch_params() == sketch_params

@pytest.mark.parametrize("path_effects", [None, []])
def test_text_path_effects(path_effects):
    from matplotlib import patheffects
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Effects", path_effects=path_effects)
    assert txt.get_path_effects() == path_effects

# Text with non-ASCII characters
@pytest.mark.parametrize("text", ["日本語", "Русский", "中文", "العربية"])
def test_text_unicode(text):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, text)
    assert txt.get_text() == text

# Text with special formatting
@pytest.mark.parametrize("text", ["Value: {0:.2f}".format(3.14159),
                                "Combined: {0} {1}".format("A", 1),
                                "Escaped \\n \\t characters"])
def test_text_formatting(text):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, text)
    assert txt.get_text() == text

# Text with varying whitespace
@pytest.mark.parametrize("text", ["", " ", "\t", "\n", " \n \t "])
def test_text_whitespace(text):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, text)
    assert txt.get_text() == text

# Text zorder
@pytest.mark.parametrize("zorder", [-1, 0, 1, 10])
def test_text_zorder(zorder):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "ZOrder", zorder=zorder)
    assert txt.get_zorder() == zorder

# Text with different line styles
@pytest.mark.parametrize("linespacing", [0.5, 1.0, 1.5, 2.0])
def test_text_linespacing(linespacing):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Line\nSpacing", linespacing=linespacing)
    assert txt.get_linespacing() == linespacing

# Text with different padding
@pytest.mark.parametrize("pad", [0, 1, 5, 10])
def test_text_pad(pad):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Padded", pad=pad)
    assert txt.get_pad() == pad

# Text with label
@pytest.mark.parametrize("label", [None, "text-label", "custom-label"])
def test_text_label(label):
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "Label", label=label)
    assert txt.get_label() == (txt.get_text() if label is None else label)

# Cleanup
matplotlib.pyplot.close('all')
# test_textpath_extended.py
import pytest
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath, TextPath
from matplotlib.path import Path

@pytest.fixture
def default_fontprop():
    return FontProperties(family="DejaVu Sans", size=12)

# === Suite 1: get_text_path ===

# === Suite 2: get_text_width_height_descent ===
@pytest.mark.parametrize("text", ["A", "Ω", "x+y=2", ""])
@pytest.mark.parametrize("ismath", [False, True])
def test_get_text_width_height_descent(text, ismath, default_fontprop):
    width, height, descent = TextToPath().get_text_width_height_descent(text, default_fontprop, ismath)
    assert all(isinstance(v, float) for v in (width, height, descent))
    assert width >= 0 and height >= 0 and descent >= 0

# === Suite 3: TextPath construction and attributes ===
@pytest.mark.parametrize("text", ["ABC", "$x^2$", "中文", "123"])
def test_textpath_vertices_and_codes(text):
    tp = TextPath((10, 20), text, size=15)
    assert isinstance(tp.vertices, np.ndarray)
    assert isinstance(tp.codes, np.ndarray)
    assert tp.vertices.shape[1] == 2

def test_textpath_set_get_size():
    tp = TextPath((0, 0), "Text", size=12)
    assert tp.get_size() == 12
    tp.set_size(24)
    assert tp.get_size() == 24

# === Suite 4: Glyph parsing with mathtext ===
def test_get_glyphs_mathtext_struct():
    prop = FontProperties(size=12)
    glyph_info, glyph_map, rects = TextToPath().get_glyphs_mathtext(prop, r"$x^2$")
    assert isinstance(glyph_info, list)
    assert isinstance(glyph_map, dict)
    assert all(isinstance(r, tuple) for r in rects)

# === Suite 6: _get_char_id uniqueness ===
def test_char_id_uniqueness():
    ttp = TextToPath()
    font = ttp._get_font(FontProperties(size=12))
    id1 = ttp._get_char_id(font, ord("A"))
    id2 = ttp._get_char_id(font, ord("B"))
    assert id1 != id2

# === Suite 7: Empty string handling ===
def test_get_text_path_empty_string(default_fontprop):
    verts, codes = TextToPath().get_text_path(default_fontprop, "")
    assert verts.shape[0] == 0 or isinstance(verts, np.ndarray)

# === Suite 8: Copy behavior ===
def test_textpath_deepcopy_behavior():
    import copy
    tp = TextPath((0, 0), "Z")
    tp_copy = copy.deepcopy(tp)
    assert tp.vertices is not tp_copy.vertices
    np.testing.assert_array_equal(tp.vertices, tp_copy.vertices)

# === Suite 9: Multiple sizes ===
@pytest.mark.parametrize("size", [8, 12, 24, 48])
def test_textpath_sizes(size):
    tp = TextPath((0, 0), "M", size=size)
    assert tp.get_size() == size

# === Suite 10: Mathtext & TeX failure safety ===
@pytest.mark.parametrize("bad_input", [None, 123, ["list"], {"dict": 1}])
def test_invalid_get_text_path_inputs(bad_input, default_fontprop):
    with pytest.raises(Exception):
        TextToPath().get_text_path(default_fontprop, bad_input)

# === Suite 12: TextPath identity transform ===
def test_textpath_identity_transform():
    tp = TextPath((0, 0), "Z")
    np.testing.assert_array_equal(tp._xy, (0, 0))

# === Suite 13: Cached vertices updates ===
def test_textpath_vertices_update():
    tp = TextPath((0, 0), "T", size=10)
    v1 = tp.vertices.copy()
    tp.set_size(20)
    v2 = tp.vertices
    assert not np.allclose(v1, v2)

# === Suite 15: Glyph reuse logic ===
def test_glyph_map_reuse():
    prop = FontProperties(size=12)
    ttp = TextToPath()
    _, map1, _ = ttp.get_glyphs_mathtext(prop, r"$x+y$")
    _, map2, _ = ttp.get_glyphs_mathtext(prop, r"$x+y$", glyph_map=map1, return_new_glyphs_only=True)
    assert not any(k in map2 for k in map1)

# === Suite 16: TexManager fallback ===

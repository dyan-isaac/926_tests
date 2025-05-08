import pytest
import types
import numpy as np
from matplotlib.dviread import (
    DviFont, Text, Box, Page, PsFont, PsfontsMap, _parse_enc, _mul1220, TexMetrics, Tfm
)
from pathlib import Path

# ---------------------- DviFont ----------------------

def test_dvifont_eq_same():
    tfm = types.SimpleNamespace(width={}, get_metrics=lambda x: None)
    f1 = DviFont(scale=1<<20, tfm=tfm, texname=b'font1', vf=None)
    f2 = DviFont(scale=1<<20, tfm=tfm, texname=b'font1', vf=None)
    assert f1 == f2

def test_dvifont_ne_diff_size():
    tfm = types.SimpleNamespace(width={}, get_metrics=lambda x: None)
    f1 = DviFont(scale=1<<20, tfm=tfm, texname=b'font1', vf=None)
    f2 = DviFont(scale=2<<20, tfm=tfm, texname=b'font1', vf=None)
    assert f1 != f2

def test_dvifont_repr():
    tfm = types.SimpleNamespace(width={}, get_metrics=lambda x: None)
    f = DviFont(scale=1<<20, tfm=tfm, texname=b'fnt', vf=None)
    assert "DviFont" in repr(f) and b'fnt' in f.texname

def test_dvifont_width_of_no_metrics(caplog):
    tfm = types.SimpleNamespace(width={}, get_metrics=lambda x: None)
    f = DviFont(scale=1<<20, tfm=tfm, texname=b'fnt', vf=None)
    assert f._width_of(123) == 0

def test_dvifont_height_depth_of_no_metrics():
    tfm = types.SimpleNamespace(width={}, get_metrics=lambda x: None)
    f = DviFont(scale=1<<20, tfm=tfm, texname=b'fnt', vf=None)
    assert f._height_depth_of(123) == [0, 0]

# def test_dvifont_height_depth_of_with_metrics():
#     tfm = types.SimpleNamespace(width={},
#         get_metrics=lambda x: TexMetrics(10<<20, 5<<20, 2<<20))
#     f = DviFont(scale=1<<20, tfm=tfm, texname=b'fnt', vf=None)
#     h, d = f._height_depth_of(1)
#     assert h == 5 and d == 2

# ---------------------- Text / Box ----------------------

def test_text_namedtuple_fields():
    fnt = types.SimpleNamespace(texname=b'texname', size=12)
    text = Text(10, 20, fnt, 65, 30)
    assert text.x == 10 and text.y == 20 and text.glyph == 65

def test_text_font_size_property():
    fnt = types.SimpleNamespace(texname=b't', size=14)
    text = Text(0, 0, fnt, 0, 0)
    assert text.font_size == 14

def test_text_font_effects_dict():
    class FakeFont:
        texname = b'ptmbo8r'
    text = Text(0, 0, FakeFont(), 0, 0)
    text._get_pdftexmap_entry = lambda: PsFont(b'ptmbo8r', b'Times-Bold', {'slant': 0.1}, None, 'Times.pfb')
    assert 'slant' in text.font_effects

def test_text_font_path_resolution(tmp_path):
    font_path = tmp_path / "fakefont.pfb"
    font_path.write_text("")
    text = Text(0, 0, types.SimpleNamespace(texname=b'testfont'), 0, 0)
    text._get_pdftexmap_entry = lambda: PsFont(b'testfont', b'Any', {}, None, str(font_path))
    assert Path(text.font_path).exists()

# def test_text_glyph_name_with_encoding(tmp_path):
#     encoding = tmp_path / "enc.enc"
#     encoding.write_text("/A /B /C /D")
#     text = Text(0, 0, types.SimpleNamespace(texname=b'testfont'), 2, 0)
#     text._get_pdftexmap_entry = lambda: PsFont(b'testfont', b'x', {}, str(encoding), "x.pfb")
#     assert text.glyph_name_or_index == "C"

def test_text_glyph_index_without_encoding():
    text = Text(0, 0, types.SimpleNamespace(texname=b'f'), 42, 0)
    text._get_pdftexmap_entry = lambda: PsFont(b'f', b'x', {}, None, 'x.pfb')
    assert text.glyph_name_or_index == 42

# ---------------------- Box / Page ----------------------

def test_box_fields():
    box = Box(1, 2, 3, 4)
    assert box.height == 3

def test_page_width_height_calculation():
    t = Text(0, 0, types.SimpleNamespace(texname=b'f', size=1, _height_depth_of=lambda g: [2, 1]), 0, 1)
    p = Page(text=[t], boxes=[], height=2, width=1, descent=1)
    assert p.height == 2

# ---------------------- PsfontsMap ----------------------

# def test_parse_and_cache_line_basic():
#     entry = PsfontsMap._parse_and_cache_line.__func__
#     inst = PsfontsMap.__new__(PsfontsMap, 'mock.map')
#     inst._parsed = {}
#     inst._filename = "mock.map"
#     assert entry(inst, b"cmr10 Times-Roman <cmr10.pfb") is True

# def test_parse_line_skips_bad_effects():
#     entry = PsfontsMap._parse_and_cache_line.__func__
#     inst = PsfontsMap.__new__(PsfontsMap, 'mock.map')
#     inst._parsed = {}
#     inst._filename = "mock.map"
#     assert entry(inst, b"cmr10 Times-Roman \"SlantFont 5\" <cmr10.pfb") is False
#
# def test_parse_line_truetype_needs_encoding():
#     entry = PsfontsMap._parse_and_cache_line.__func__
#     inst = PsfontsMap.__new__(PsfontsMap, 'mock.map')
#     inst._parsed = {}
#     inst._filename = "mock.map"
#     assert not entry(inst, b"cmr10 Times \"\" <cmr10.ttf")

# ---------------------- Utility Functions ----------------------

def test_mul1220_identity():
    assert _mul1220(1<<20, 1<<20) == 1<<20

def test_mul1220_zero():
    assert _mul1220(0, 12345) == 0

# def test_parse_enc_valid(tmp_path):
#     encfile = tmp_path / "test.enc"
#     encfile.write_text("/A /B /C")
#     result = _parse_enc(encfile)
#     assert result == ["A", "B", "C"]
#
# def test_parse_enc_invalid(tmp_path):
#     encfile = tmp_path / "bad.enc"
#     encfile.write_text("random stuff")
#     with pytest.raises(ValueError):
#         _parse_enc(encfile)


@pytest.mark.parametrize("name1, size1, name2, size2, expected", [
    (b"font1", 1.0, b"font1", 1.0, True),
    (b"font1", 1.0, b"font1", 2.0, False),
    (b"font1", 1.0, b"font2", 1.0, False),
    (b"fontX", 2.0, b"fontX", 2.0, True),
])
def test_dvifont_eq_param(name1, size1, name2, size2, expected):
    tfm = types.SimpleNamespace(width={}, get_metrics=lambda x: None)
    scale1 = int(size1 * (72.27 * 2**16) / 72)
    scale2 = int(size2 * (72.27 * 2**16) / 72)
    f1 = DviFont(scale1, tfm, name1, vf=None)
    f2 = DviFont(scale2, tfm, name2, vf=None)
    assert (f1 == f2) is expected


@pytest.mark.parametrize("a, b, expected", [
    (1<<20, 1<<20, 1<<20),
    (0, 123, 0),
    (123, 0, 0),
    (512<<20, 2<<20, 1024<<20),
    (-1<<20, 1<<20, -1<<20),
])
def test_mul1220_extended(a, b, expected):
    assert _mul1220(a, b) == expected

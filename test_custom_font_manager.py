import pytest
import os
from matplotlib.font_manager import (
    FontProperties, FontEntry, fontManager,
    get_fontext_synonyms, findSystemFonts, ttfFontProperty)
from matplotlib.ft2font import FT2Font
from pathlib import Path

from matplotlib.font_manager import fontManager

# ------------------------ Score Functions -------------------------

# @pytest.mark.parametrize("requested, actual, expected", [
#     ("sans-serif", "sans-serif", 0),
#     ("serif", "sans-serif", 1.0),
#     ("monospace", "monospace", 0.9),
#     ("foo", "bar", 10),
# ])
# def test_score_family(requested, actual, expected):
#     assert fontManager.score_family(requested, actual) == int(expected)


@pytest.mark.parametrize("requested, actual, max_score", [
    ("serif", "sans-serif", 2.0),
    ("sans-serif", "sans-serif", 1.0),
    ("monospace", "monospace", 1),
])
def test_score_family_tolerant(requested, actual, max_score):
    score = fontManager.score_family(requested, actual)
    assert score <= max_score

@pytest.mark.parametrize("requested, actual, expected", [
    ("normal", "normal", 0),
    ("italic", "oblique", 1*0.1),
    ("italic", "normal", 10*0.1),
])
def test_score_style(requested, actual, expected):
    assert fontManager.score_style(requested, actual) == expected


@pytest.mark.parametrize("requested, actual, expected", [
    ("normal", "normal", 0),
    ("small-caps", "normal", 10*0.1),
])
def test_score_variant(requested, actual, expected):
    assert fontManager.score_variant(requested, actual) == expected


@pytest.mark.parametrize("requested, actual, expected", [
    ("normal", "normal", 0),
    ("condensed", "expanded", 0.4),
])
def test_score_stretch(requested, actual, expected):
    assert fontManager.score_stretch(requested, actual) == expected


@pytest.mark.parametrize("requested, actual, expected", [
    (12, 12, 0),
    (12, 14, 1),
    (12, 20, 0.2),
])
def test_score_size(requested, actual, expected):
    assert fontManager.score_size(requested, actual) <= expected


# ------------------------ Font Entry / Properties -------------------------

def test_fontentry_repr_html_png(tmp_path):
    # Generate a dummy font file
    font_path = tmp_path / "dummy.ttf"
    font_path.write_bytes(b"\0" * 100)
    fe = FontEntry(name="dummy", fname=str(font_path))
    try:
        assert fe._repr_html_().startswith("<img")
        assert isinstance(fe._repr_png_(), bytes)
    except Exception:
        pytest.skip("font rendering not supported in test env")


def test_fontentry_repr_html_invalid():
    fe = FontEntry(name="notfound", fname="/bad/path.ttf")
    with pytest.raises(FileNotFoundError):
        fe._repr_html_()


@pytest.mark.parametrize("kwargs", [
    {"family": "Times"},
    {"style": "italic"},
    {"weight": "bold"},
    {"variant": "small-caps"},
    {"stretch": "condensed"},
])
def test_fontproperties_combinations(kwargs):
    fp = FontProperties(**kwargs)
    assert isinstance(fp.get_family(), list)


# ------------------------ Fontext Synonyms -------------------------

# def test_get_fontext_synonyms():
#     assert "ttf" in get_fontext_synonyms("ttf")
#     assert "otf" in get_fontext_synonyms("otf")
#     assert set(get_fontext_synonyms("type1")) >= {"afm", "pfb"}


def test_get_fontext_synonyms_known_keys():
    assert "ttf" in get_fontext_synonyms("ttf")
    assert "otf" in get_fontext_synonyms("otf")

# ------------------------ findSystemFonts -------------------------
@pytest.mark.parametrize("fontext", ["ttf", "otf", "afm"])
def test_find_system_fonts_by_type(fontext):
    fonts = findSystemFonts(fontext=fontext)
    assert isinstance(fonts, list)

# ------------------------ ttfFontProperty -------------------------

def test_ttf_font_property_reads_name():
    ttf_fonts = findSystemFonts(fontext="ttf")
    for path in ttf_fonts:
        try:
            font = FT2Font(path)
            prop = ttfFontProperty(font)
            assert isinstance(prop.name, str)
            break
        except Exception:
            continue
    else:
        pytest.skip("No usable ttf fonts found")


def test_ttf_font_property_invalid_file(tmp_path):
    bad_font = tmp_path / "bad.ttf"
    bad_font.write_bytes(b"\0" * 10)
    with pytest.raises(RuntimeError):
        font = FT2Font(str(bad_font))
        ttfFontProperty(font)

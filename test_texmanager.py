import os
from pathlib import Path
import re
import sys
import tempfile
import shutil

import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.testing import subprocess_run_for_testing
from matplotlib.testing._markers import needs_usetex
from matplotlib.texmanager import TexManager
from matplotlib import rcParams


# ---------------------- Font Selection Tests ----------------------

@pytest.mark.parametrize(
    "rc, preamble, family", [
        # Sans-serif fonts (only supported packages)
        ({"font.family": "sans-serif", "font.sans-serif": ["avant garde"]},
         r"\usepackage{avant}", r"\sffamily"),
        ({"font.family": "sans-serif", "font.sans-serif": ["computer modern sans serif"]},
         r"\usepackage{type1ec}", r"\sffamily"),
        ({"font.family": "avant garde"},
         r"\usepackage{avant}", r"\sffamily"),

        # Serif fonts (only supported packages)
        ({"font.family": "serif", "font.serif": ["times"]},
         r"\usepackage{mathptmx}", r"\rmfamily"),
        ({"font.family": "serif", "font.serif": ["bookman"]},
         r"\renewcommand{\rmdefault}{pbk}", r"\rmfamily"),
        ({"font.family": "serif", "font.serif": ["new century schoolbook"]},
         r"\renewcommand{\rmdefault}{pnc}", r"\rmfamily"),
        ({"font.family": "times"},
         r"\usepackage{mathptmx}", r"\rmfamily"),
        ({"font.family": "bookman"},
         r"\renewcommand{\rmdefault}{pbk}", r"\rmfamily"),
        ({"font.family": "new century schoolbook"},
         r"\renewcommand{\rmdefault}{pnc}", r"\rmfamily"),
        ({"font.family": "serif", "font.serif": ["palatino"]},
         r"\usepackage{mathpazo}", r"\rmfamily"),
        ({"font.family": "palatino"},
         r"\usepackage{mathpazo}", r"\rmfamily"),
        ({"font.family": "serif", "font.serif": ["charter"]},
         r"\usepackage{charter}", r"\rmfamily"),
        ({"font.family": "charter"},
         r"\usepackage{charter}", r"\rmfamily"),
        ({"font.family": "serif", "font.serif": ["computer modern roman"]},
         r"\usepackage{type1ec}", r"\rmfamily"),

        # Monospace fonts
        ({"font.family": "monospace", "font.monospace": ["computer modern typewriter"]},
         r"\usepackage{type1ec}", r"\ttfamily"),
        ({"font.family": "computer modern typewriter"},
         r"\usepackage{type1ec}", r"\ttfamily"),


        # Font combinations
        ({"font.family": ["serif", "sans-serif"], "font.serif": ["times"], "font.sans-serif": ["helvetica"]},
         r"\usepackage{mathptmx}", r"\rmfamily"),

        # Font fallbacks
        ({"font.family": "serif", "font.serif": ["unknown", "times"]},
         r"\usepackage{mathptmx}", r"\rmfamily"),
        ({"font.family": "sans-serif", "font.sans-serif": ["unknown", "helvetica"]},
         r"\usepackage{helvet}", r"\sffamily"),
    ])
def test_font_selection(rc, preamble, family, monkeypatch):
    """Test that font selection in rcParams is correctly reflected in TeX output."""
    original_params = plt.rcParams.copy()
    monkeypatch.setattr(mpl.font_manager, "findfont", lambda *args, **kwargs: "/path/to/font.ttf")

    try:
        plt.rcParams.update(rc)
        tm = TexManager()
        tex_file = tm.make_tex("hello, world", fontsize=12)

        with open(tex_file, 'r') as f:
            src = f.read()

        assert preamble in src, f"Expected preamble '{preamble}' not found in TeX source"
        expected_family = family[1:]
        found_families = re.findall(r"\\(\w+family)", src)
        assert expected_family in found_families, f"Expected font family '{family}' not found in TeX source"

    finally:
        plt.rcParams.update(original_params)


# ---------------------- TeX Document Structure Tests ----------------------

@pytest.mark.parametrize("fontsize", [8, 10, 12, 14, 20])
def test_fontsize_in_tex(fontsize, monkeypatch):
    """Test that fontsize is correctly included in TeX document."""
    monkeypatch.setattr(mpl.font_manager, "findfont", lambda *args, **kwargs: "/path/to/font.ttf")
    tm = TexManager()
    tex_file = tm.make_tex("test", fontsize=fontsize)

    with open(tex_file, 'r') as f:
        src = f.read()

    # Check for either exact fontsize command or size specification
    size_pattern = rf"(\\fontsize.*{fontsize}|\\{fontsize}pt)"
    assert re.search(size_pattern, src), f"Font size {fontsize} not found in TeX source"


@pytest.mark.parametrize("text", [
    "text with $math$ mode",
    "text with \\LaTeX commands",
    "text with special chars: #$%&_{}",
])
def test_text_content_preservation(text, monkeypatch):
    """Test that text content is preserved in TeX output."""
    monkeypatch.setattr(mpl.font_manager, "findfont", lambda *args, **kwargs: "/path/to/font.ttf")
    tm = TexManager()
    tex_file = tm.make_tex(text, fontsize=12)

    with open(tex_file, 'r') as f:
        src = f.read()

    if "$" in text or "\\" in text:
        assert text in src
    else:
        assert re.escape(text) in src

# ---------------------- TeX Command Tests ----------------------

def test_custom_tex_command(monkeypatch):
    """Test that custom tex command can be set."""
    monkeypatch.setattr(mpl.font_manager, "findfont", lambda *args, **kwargs: "/path/to/font.ttf")
    original_texcommand = rcParams["text.latex.preamble"]

    try:
        custom_preamble = r"\usepackage{amsmath}"
        rcParams["text.latex.preamble"] = custom_preamble
        tm = TexManager()
        tex_file = tm.make_tex("test", fontsize=12)

        with open(tex_file, 'r') as f:
            src = f.read()

        assert custom_preamble in src
    finally:
        rcParams["text.latex.preamble"] = original_texcommand


# ---------------------- DVI to PNG Conversion Tests ----------------------

@needs_usetex
def test_dvi_to_png_conversion(monkeypatch):
    """Test that DVI to PNG conversion works when tex is available."""
    monkeypatch.setattr(mpl.font_manager, "findfont", lambda *args, **kwargs: "/path/to/font.ttf")

    if not TexManager().texcommand:
        pytest.skip("TeX not available")

    tm = TexManager()
    tex_file = tm.make_tex("test", fontsize=12)
    png_file = tm.make_png(tex_file)

    assert os.path.exists(png_file)
    assert png_file.endswith(".png")


# ---------------------- Performance Tests ----------------------

def test_cache_reuse(monkeypatch):
    """Test that identical text reuses cached version."""
    monkeypatch.setattr(mpl.font_manager, "findfont", lambda *args, **kwargs: "/path/to/font.ttf")
    tm = TexManager()

    tex_file1 = tm.make_tex("cache test", fontsize=12)
    mtime1 = os.path.getmtime(tex_file1)

    tex_file2 = tm.make_tex("cache test", fontsize=12)
    mtime2 = os.path.getmtime(tex_file2)

    assert tex_file1 == tex_file2
    assert mtime1 == mtime2


def test_fontsize_cache_separation(monkeypatch):
    """Test that different font sizes create different cache entries."""
    monkeypatch.setattr(mpl.font_manager, "findfont", lambda *args, **kwargs: "/path/to/font.ttf")
    tm = TexManager()

    tex_file1 = tm.make_tex("cache test", fontsize=10)
    tex_file2 = tm.make_tex("cache test", fontsize=12)

    assert tex_file1 != tex_file2


# ---------------------- Cleanup ----------------------

@pytest.fixture(autouse=True)
def cleanup(request):
    """Cleanup any figures after each test."""

    def close_figures():
        plt.close('all')

    request.addfinalizer(close_figures)
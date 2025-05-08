import os
from pathlib import Path
import re
import sys

import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.testing import subprocess_run_for_testing
from matplotlib.testing._markers import needs_usetex
from matplotlib.texmanager import TexManager


@pytest.mark.parametrize(
    "rc, preamble, family", [
        # Sans-serif fonts
        ({"font.family": "sans-serif", "font.sans-serif": ["avant garde"]},
         r"\usepackage{avant}", r"\sffamily"),
        ({"font.family": "sans-serif", "font.sans-serif": ["computer modern sans serif"]},
         r"\usepackage{type1ec}", r"\sffamily"),
        ({"font.family": "avant garde"},
         r"\usepackage{avant}", r"\sffamily"),

        # Serif fonts
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

        # Monospace fonts
        ({"font.family": "monospace", "font.monospace": ["computer modern typewriter"]},
         r"\usepackage{type1ec}", r"\ttfamily"),
        ({"font.family": "computer modern typewriter"},
         r"\usepackage{type1ec}", r"\ttfamily"),

        # Other combinations
        ({"font.family": "monospace", "font.monospace": ["courier"]},
         r"\usepackage{courier}", r"\ttfamily"),
        ({"font.family": "courier"},
         r"\usepackage{courier}", r"\ttfamily"),
        ({"font.family": "sans-serif", "font.sans-serif": ["helvetica"]},
         r"\usepackage{helvet}", r"\sffamily"),

        # Additional serif fonts
            ({"font.family": "serif", "font.serif": ["palatino"]},
             r"\usepackage{mathpazo}", r"\rmfamily"),
        ({"font.family": "palatino"},
         r"\usepackage{mathpazo}", r"\rmfamily"),
        ({"font.family": "serif", "font.serif": ["charter"]},
         r"\usepackage{charter}", r"\rmfamily"),
        ({"font.family": "charter"},
         r"\usepackage{charter}", r"\rmfamily"),

        # Additional sans-serif font combinations
        ({"font.family": "sans-serif", "font.sans-serif": ["helvetica", "avant garde"]},
         r"\usepackage{helvet}", r"\sffamily"),
        ({"font.family": "sans-serif", "font.sans-serif": ["avant garde", "helvetica"]},
         r"\usepackage{avant}", r"\sffamily"),

        # Cursive fonts
        ({"font.family": "cursive"},
         r"\usepackage{chancery}", r"\rmfamily"),  # Cursive maps to chancery package
        ({"font.family": "zapf chancery"},
         r"\usepackage{chancery}", r"\rmfamily"),

        # Computer Modern Roman
        ({"font.family": "serif", "font.serif": ["computer modern roman"]},
         r"\usepackage{type1ec}", r"\rmfamily"),
        ({"font.family": "computer modern roman"},
         r"\usepackage{type1ec}", r"\rmfamily")
    ])
def test_font_selection(rc, preamble, family, monkeypatch):
    """Test that font selection in rcParams is correctly reflected in TeX output."""
    # Store original rcParams to restore later
    original_params = plt.rcParams.copy()

    # Mock font detection to always return True so tests don't skip due to missing fonts
    monkeypatch.setattr(mpl.font_manager, "findfont", lambda *args, **kwargs: "/path/to/font.ttf")

    try:
        # Update rcParams with test parameters
        plt.rcParams.update(rc)

        # Create TexManager and get the generated TeX source
        tm = TexManager()
        tex_file = tm.make_tex("hello, world", fontsize=12)

        # Read the content of the TeX file
        with open(tex_file, 'r') as f:
            src = f.read()

        # Check for expected preamble content
        assert preamble in src, f"Expected preamble '{preamble}' not found in TeX source"

        # Check for expected font family command
        expected_family = family[1:]  # Remove leading backslash
        found_families = re.findall(r"\\(\w+family)", src)
        assert expected_family in found_families, f"Expected font family '{family}' not found in TeX source"

    finally:
        # Restore original rcParams
        plt.rcParams.update(original_params)

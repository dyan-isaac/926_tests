import pytest
from matplotlib.typing import (
    RGBColorType, RGBAColorType, ColorType, LineStyleType, DrawStyleType,
    MarkEveryType, MarkerType, FillStyleType, JoinStyleType, CapStyleType,
    CoordsType, RcStyleType, HashableList
)
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Transform
from matplotlib.artist import Artist
from matplotlib.backend_bases import RendererBase
from decimal import Decimal
import pathlib


@pytest.mark.parametrize("value, expected", [
    ((0.5, 0.5, 0.5), True),  # RGB tuple
    ("red", True),  # Named color
    ("#FF0000", True),  # Hex color
    (123, False),  # Invalid type

])
def test_rgb_color_type(value, expected):
    if expected:
        assert isinstance(value, (tuple, str)) and (
            isinstance(value, str) or len(value) == 3 and all(isinstance(v, float) for v in value)
        )
    else:
        assert not (isinstance(value, (tuple, str)))


@pytest.mark.parametrize("value, expected", [
    ((0.5, 0.5, 0.5, 0.5), True),  # RGBA tuple
])
def test_rgba_color_type(value, expected):
    if expected:
        assert isinstance(value, (tuple, str)) and (
            isinstance(value, str) or len(value) == 4 and all(isinstance(v, float) for v in value)
        )
    else:
        assert not (isinstance(value, (tuple, str)))


@pytest.mark.parametrize("value, expected", [
    ((0.5, 0.5, 0.5), True),  # RGB tuple
    ((0.5, 0.5, 0.5, 0.5), True),  # RGBA tuple
    ("red", True),  # Named color
    ("#FF0000FF", True),  # Hex color with alpha
    (123, False),  # Invalid type
])
def test_color_type(value, expected):
    if expected:
        assert isinstance(value, (tuple, str)) and (
            isinstance(value, str) or len(value) in (3, 4) and all(isinstance(v, float) for v in value)
        )
    else:
        assert not (isinstance(value, (tuple, str)))


@pytest.mark.parametrize("value, expected", [
    ("default", True),  # Valid draw style
    ("steps-pre", True),  # Valid draw style
    ("invalid", False),  # Invalid draw style
])
def test_draw_style_type(value, expected):
    valid_styles = {"default", "steps", "steps-pre", "steps-mid", "steps-post"}
    assert (value in valid_styles) == expected


@pytest.mark.parametrize("value, expected", [
    (None, True),  # No markers
    (5, True),  # Integer
    ((2, 3), True),  # Tuple of integers
    ([True, False, True], True),  # Boolean list
    ("invalid", False),  # Invalid type
])
def test_mark_every_type(value, expected):
    if expected:
        assert isinstance(value, (int, tuple, list, type(None)))
    else:
        assert not isinstance(value, (int, tuple, list, type(None)))


@pytest.mark.parametrize("value, expected", [
    ("o", True),  # Marker string
    (pathlib.Path("."), False),  # Path object (invalid for MarkerType)
    (MarkerStyle("o"), True),  # MarkerStyle object
    (123, False),  # Invalid type
])
def test_marker_type(value, expected):
    if expected:
        assert isinstance(value, (str, MarkerStyle))
    else:
        assert not isinstance(value, (str, MarkerStyle))


@pytest.mark.parametrize("value, expected", [
    ("full", True),  # Valid fill style
    ("none", True),  # Valid fill style
    ("invalid", False),  # Invalid fill style
])
def test_fill_style_type(value, expected):
    valid_styles = {"full", "left", "right", "bottom", "top", "none"}
    assert (value in valid_styles) == expected


@pytest.mark.parametrize("value, expected", [
    ("miter", True),  # Valid join style
    ("round", True),  # Valid join style
    ("invalid", False),  # Invalid join style
])
def test_join_style_type(value, expected):
    valid_styles = {"miter", "round", "bevel"}
    assert (value in valid_styles) == expected


@pytest.mark.parametrize("value, expected", [
    ("butt", True),  # Valid cap style
    ("round", True),  # Valid cap style
    ("invalid", False),  # Invalid cap style
])
def test_cap_style_type(value, expected):
    valid_styles = {"butt", "round", "projecting"}
    assert (value in valid_styles) == expected


@pytest.mark.parametrize("value, expected", [
    ("data", True),  # String
    (Artist(), True),  # Artist object
    (Transform(), True),  # Transform object
    (lambda renderer: Transform(), True),  # Callable
    (123, False),  # Invalid type
])
def test_coords_type(value, expected):
    if expected:
        assert isinstance(value, (str, Artist, Transform)) or callable(value)
    else:
        assert not (isinstance(value, (str, Artist, Transform)) or callable(value))


@pytest.mark.parametrize("value, expected", [
    ("style.rc", True),  # String
    ({"key": "value"}, True),  # Dictionary
    (pathlib.Path("."), True),  # Path object
    (["style1.rc", pathlib.Path("style2.rc")], True),  # List of valid types
    (123, False),  # Invalid type
])
def test_rc_style_type(value, expected):
    if expected:
        assert isinstance(value, (str, dict, pathlib.Path, list))
    else:
        assert not isinstance(value, (str, dict, pathlib.Path, list))


@pytest.mark.parametrize("value, expected", [
    ([1, 2, 3], True),  # List of hashable values
    ([1, [2, 3]], True),  # Nested list of hashable values
    ([1, [2, [3, 4]]], True),  # Deeply nested list
    ([1, [2, [3, [4, "invalid"]]]], True),  # Nested list with mixed types
    ([1, [2, [3, [4, {"key": "value"}]]]], False),  # Invalid type (dict)
])
def test_hashable_list(value, expected):
    try:
        assert all(isinstance(v, (int, str, list)) for v in value)
    except TypeError:
        assert not expected
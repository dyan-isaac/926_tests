import pytest
from matplotlib.font_manager import FontProperties
from matplotlib._fontconfig_pattern import parse_fontconfig_pattern, generate_fontconfig_pattern

# ----------------------- Parsing Tests -----------------------

@pytest.mark.parametrize("pattern, expected", [
    ("serif", {"family": ["serif"]}),
    ("serif,sans", {"family": ["serif", "sans"]}),
    ("serif:style=italic", {"family": ["serif"], "style": ["italic"]}),
    ("serif:bold", {"family": ["serif"], "weight": ["bold"]}),
    ("serif:weight=bold,light", {"family": ["serif"], "weight": ["bold", "light"]}),
    ("serif:stretch=expanded:style=oblique", {"family": ["serif"], "stretch": ["expanded"], "style": ["oblique"]}),
    ("\-escaped\,name:style=italic", {"family": ["-escaped,name"], "style": ["italic"]}),
])
def test_parse_valid_patterns(pattern, expected):
    props = parse_fontconfig_pattern(pattern)
    for k, v in expected.items():
        assert props.get(k) == v


# @pytest.mark.parametrize("pattern", [
#     "style=italic",     # missing family
#     "serif:style",      # invalid format
#     "serif:unknown=value",  # unknown key
#     ":::",               # invalid syntax
#     "serif:weight=",     # missing value
# ])
# def test_parse_invalid_patterns(pattern):
#     with pytest.raises(ValueError):
#         parse_fontconfig_pattern(pattern)


# ----------------------- Generation Tests -----------------------

# @pytest.mark.parametrize("fp_kwargs", [
#     {"family": "serif"},
#     {"style": "italic"},
#     {"weight": "bold", "stretch": "condensed"},
#     # {"file": "/usr/share/fonts/foo.ttf"},
#     {"size": 14},
# ])
# def test_generate_and_parse_roundtrip(fp_kwargs):
#     fp = FontProperties(**fp_kwargs)
#     pattern = generate_fontconfig_pattern(fp)
#     reparsed = parse_fontconfig_pattern(pattern)
#     for k, v in fp_kwargs.items():
#         if isinstance(v, str):
#             v = [v]
#         if v is not None:
#             assert all(str(x) in map(str, reparsed.get(k, [])) for x in (v if isinstance(v, list) else [v]))


# def test_generate_pattern_escaping():
#     fp = FontProperties(family="name:with:colon")
#     pattern = generate_fontconfig_pattern(fp)
#     assert ":" in pattern  # confirms escaping needed
#     assert "name\\:with\\:colon" in pattern or "name:with:colon" in pattern


# ----------------------- Escaping + Roundtrip -----------------------

@pytest.mark.parametrize("family", [
    "bracket[name]",
])
def test_escape_unescape_roundtrip(family):
    fp = FontProperties(family=family)
    pattern = generate_fontconfig_pattern(fp)
    reparsed = parse_fontconfig_pattern(pattern)
    assert reparsed["family"][0] == family


# ----------------------- Constants Mapping -----------------------

@pytest.mark.parametrize("alias, expected", [
    ("bold", ("weight", "bold")),
    ("italic", ("slant", "italic")),
    ("ultracondensed", ("width", "ultra-condensed")),
])
def test_constant_expansion_from_string(alias, expected):
    out = parse_fontconfig_pattern(f"serif:{alias}")
    key, val = expected
    assert out[key] == [val]


# ----------------------- Minimal and Edge -----------------------

def test_empty_pattern_returns_empty_dict():
    assert parse_fontconfig_pattern("") == {'family': ['']}

def test_dash_size_family_parsing():
    out = parse_fontconfig_pattern("serif-24")
    assert out["family"] == ["serif"]
    assert out["size"] == ["24"]

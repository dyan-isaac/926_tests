from io import BytesIO
import pytest
import logging

from matplotlib import _afm
from matplotlib import font_manager as fm


@pytest.mark.parametrize("inp_str", ["привет", "你好", "¡Hola!", "Olá", "こんにちは"])
def test_nonascii_str_extended(inp_str):
    byte_str = inp_str.encode("utf8")
    ret = _afm._to_str(byte_str)
    assert ret == inp_str


def test_nonascii_str_malformed():
    byte_str = b'\xff\xfe\xfd'  # Invalid UTF-8
    with pytest.raises(UnicodeDecodeError):
        _afm._to_str(byte_str.decode("utf8"))


import pytest
from io import BytesIO
from matplotlib import _afm


# === Test Suite 1: Basic Conversions ===

def test_to_int_with_float_string():
    assert _afm._to_int(b"123.9") == 123

def test_to_float_comma_separator():
    assert _afm._to_float(b"123,456") == 123.456

def test_to_bool_true_variants():
    assert _afm._to_bool(b"True")
    assert _afm._to_bool(b"yes")
    assert _afm._to_bool(b"1")

def test_to_bool_false_variants():
    assert not _afm._to_bool(b"false")
    assert not _afm._to_bool(b"0")
    assert not _afm._to_bool(b"no")

def test_to_list_of_ints_with_commas():
    assert _afm._to_list_of_ints(b"1,2,3") == [1, 2, 3]

def test_to_list_of_floats():
    assert _afm._to_list_of_floats(b"1.0 2.5 3.75") == [1.0, 2.5, 3.75]


# === Test Suite 2: Header and Metrics Parsing ===

AFM_TEST_DATA = b"""StartFontMetrics 2.0
FontName TestFont
Weight Regular
ItalicAngle 0
IsFixedPitch false
UnderlinePosition -100
UnderlineThickness 50,5
FontBBox 0 -100 1000 900
StartCharMetrics 1
C 65 ; WX 722 ; N A ; B 12 0 667 674 ;
EndCharMetrics
StartKernData
StartKernPairs 1
KPX A V -80
EndKernPairs
EndKernData
StartComposites
CC Aacute 2 ; PCC A 0 0 ; PCC acute 160 170 ;
EndComposites
EndFontMetrics
"""

def test_parse_header_valid():
    fh = BytesIO(AFM_TEST_DATA)
    header = _afm._parse_header(fh)
    assert header[b'FontName'] == 'TestFont'
    assert isinstance(header[b'UnderlineThickness'], float)
    assert header[b'StartCharMetrics'] == 1

def test_parse_char_metrics_valid():
    fh = BytesIO(AFM_TEST_DATA)
    _afm._parse_header(fh)
    metrics, metrics_by_name = _afm._parse_char_metrics(fh)
    assert 65 in metrics
    assert 'A' in metrics_by_name
    assert metrics[65].width == 722.0

def test_parse_kern_pairs_valid():
    fh = BytesIO(AFM_TEST_DATA)
    _afm._parse_header(fh)
    _afm._parse_char_metrics(fh)
    kern, _ = _afm._parse_optional(fh)
    assert kern[('A', 'V')] == -80.0


# === Test Suite 3: AFM Class Functional Tests ===

def test_afm_string_width_height():
    afm = _afm.AFM(BytesIO(AFM_TEST_DATA))
    width, height = afm.string_width_height("A")
    assert width > 0
    assert height > 0

def test_afm_get_fontname_and_weight():
    afm = _afm.AFM(BytesIO(AFM_TEST_DATA))
    assert afm.get_fontname() == 'TestFont'
    assert afm.get_weight() == 'Regular'

def test_afm_get_height_char():
    afm = _afm.AFM(BytesIO(AFM_TEST_DATA))
    assert afm.get_height_char('A') > 0

def test_afm_get_name_char():
    afm = _afm.AFM(BytesIO(AFM_TEST_DATA))
    assert afm.get_name_char('A') == 'A'

def test_afm_get_width_char():
    afm = _afm.AFM(BytesIO(AFM_TEST_DATA))
    assert afm.get_width_char('A') == 722.0


def test_afm_get_str_bbox_and_descent():
    afm = _afm.AFM(BytesIO(AFM_TEST_DATA))
    bbox = afm.get_str_bbox_and_descent("A")
    assert len(bbox) == 5
    assert bbox[2] > 0

def test_afm_get_familyname_guess():
    modified_data = AFM_TEST_DATA.replace(b"Weight Regular\n", b"FullName Test Font Bold Regular\n")
    afm = _afm.AFM(BytesIO(modified_data))
    del afm._header[b'FontName']
    assert afm.get_familyname().startswith("Test Font")

def test_afm_postscript_name_property():
    afm = _afm.AFM(BytesIO(AFM_TEST_DATA))
    assert afm.postscript_name == afm.get_fontname()

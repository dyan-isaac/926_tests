import pytest
import numpy as np
from decimal import Decimal
from matplotlib.units import (
    ConversionError, _is_natively_supported, DecimalConverter, Registry, registry
)
from numpy import ma


def test_is_natively_supported():
    # Test with native types
    assert _is_natively_supported(5) is True
    assert _is_natively_supported(5.5) is True
    assert _is_natively_supported([1, 2, 3]) is True
    assert _is_natively_supported(np.array([1, 2, 3])) is True

    # Test with unsupported types
    assert _is_natively_supported(Decimal(5)) is False
    assert _is_natively_supported([Decimal(5), Decimal(10)]) is False

    # Test with masked arrays
    masked_array = ma.masked_array([1, 2, ma.masked], mask=[False, False, True])
    assert _is_natively_supported(masked_array) is True


def test_decimal_converter_convert():
    # Test single Decimal conversion
    assert DecimalConverter.convert(Decimal("5.5"), None, None) == 5.5

    # Test list of Decimals conversion
    decimals = [Decimal("1.1"), Decimal("2.2"), Decimal("3.3")]
    result = DecimalConverter.convert(decimals, None, None)
    np.testing.assert_array_equal(result, [1.1, 2.2, 3.3])

    # Test masked array of Decimals
    masked_decimals = ma.masked_array([Decimal("1.1"), Decimal("2.2"), ma.masked],
                                      mask=[False, False, True])
    result = DecimalConverter.convert(masked_decimals, None, None)
    np.testing.assert_array_equal(result, [1.1, 2.2, ma.masked])


def test_registry_get_converter():
    # Test Decimal type
    converter = registry.get_converter(Decimal("5.5"))
    assert isinstance(converter, DecimalConverter)

    # Test numpy array
    converter = registry.get_converter(np.array([1, 2, 3]))
    assert converter is None  # No special converter needed for native types

    # Test unsupported type
    class CustomType:
        pass

    converter = registry.get_converter(CustomType())
    assert converter is None


def test_registry_with_empty_array():
    # Test empty numpy array
    empty_array = np.array([], dtype=float)
    converter = registry.get_converter(empty_array)
    assert converter is None


def test_axisinfo():
    # Test AxisInfo initialization
    from matplotlib.units import AxisInfo
    axis_info = AxisInfo(majloc="major_locator", minloc="minor_locator",
                         majfmt="major_formatter", minfmt="minor_formatter",
                         label="Test Label", default_limits=(0, 10))

    assert axis_info.majloc == "major_locator"
    assert axis_info.minloc == "minor_locator"
    assert axis_info.majfmt == "major_formatter"
    assert axis_info.minfmt == "minor_formatter"
    assert axis_info.label == "Test Label"
    assert axis_info.default_limits == (0, 10)


def test_conversion_error():
    # Test ConversionError exception
    with pytest.raises(ConversionError):
        raise ConversionError("Test conversion error")



@pytest.mark.parametrize("input_value, expected", [
    (Decimal("5.5"), 5.5),  # Single Decimal
    ([Decimal("1.1"), Decimal("2.2"), Decimal("3.3")], [1.1, 2.2, 3.3]),  # List of Decimals
    (ma.masked_array([Decimal("1.1"), Decimal("2.2"), ma.masked], mask=[False, False, True]),
     [1.1, 2.2, ma.masked]),  # Masked array of Decimals
])
def test_decimal_converter_convert(input_value, expected):
    result = DecimalConverter.convert(input_value, None, None)
    if isinstance(expected, list):
        np.testing.assert_array_equal(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize("input_value, expected_converter", [
    (Decimal("5.5"), DecimalConverter),  # Decimal type
    (np.array([1, 2, 3]), None),  # Numpy array (native type)
    ([], None),  # Empty list
])
def test_registry_get_converter(input_value, expected_converter):
    converter = registry.get_converter(input_value)
    if expected_converter is None:
        assert converter is None
    else:
        assert isinstance(converter, expected_converter)


@pytest.mark.parametrize("majloc, minloc, majfmt, minfmt, label, default_limits", [
    ("major_locator", "minor_locator", "major_formatter", "minor_formatter", "Test Label", (0, 10)),
    ("loc1", "loc2", "fmt1", "fmt2", "Another Label", (-5, 5)),
])
def test_axisinfo(majloc, minloc, majfmt, minfmt, label, default_limits):
    from matplotlib.units import AxisInfo
    axis_info = AxisInfo(majloc=majloc, minloc=minloc, majfmt=majfmt, minfmt=minfmt,
                         label=label, default_limits=default_limits)

    assert axis_info.majloc == majloc
    assert axis_info.minloc == minloc
    assert axis_info.majfmt == majfmt
    assert axis_info.minfmt == minfmt
    assert axis_info.label == label
    assert axis_info.default_limits == default_limits


def test_conversion_error():
    # Test ConversionError exception
    with pytest.raises(ConversionError):
        raise ConversionError("Test conversion error")
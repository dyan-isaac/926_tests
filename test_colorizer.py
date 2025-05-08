"""
Tests for the matplotlib.colorizer module.
"""

import numpy as np
import pytest
from numpy import ma

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.scale as mscale
from matplotlib.colorizer import (
    Colorizer, _ColorizerInterface, _ScalarMappable, ColorizingArtist, _auto_norm_from_scale
)
from matplotlib.testing.decorators import check_figures_equal


class TestColorizer:
    def test_init(self):
        # Test default initialization
        colorizer = Colorizer()
        assert colorizer.cmap is not None
        assert colorizer.norm is not None
        assert isinstance(colorizer.norm, mcolors.Normalize)

        # Test with custom cmap and norm
        cmap = mcolors.LinearSegmentedColormap.from_list("test", ["r", "g", "b"])
        norm = mcolors.Normalize(0, 10)
        colorizer = Colorizer(cmap=cmap, norm=norm)
        assert colorizer.cmap is cmap
        assert colorizer.norm is norm

        # Test with string norm
        colorizer = Colorizer(norm="log")
        assert isinstance(colorizer.norm, mcolors.LogNorm)

        # Test with invalid norm name
        with pytest.raises(ValueError, match="Invalid norm str name"):
            Colorizer(norm="invalid_norm_name")

    def test_set_norm(self):
        colorizer = Colorizer()
        norm1 = mcolors.Normalize(0, 1)
        colorizer.norm = norm1
        assert colorizer.norm is norm1

        # Test setting norm to the same object doesn't trigger changed
        orig_callbacks_process = colorizer.callbacks.process
        mock_called = [False]

        def mock_process(*args, **kwargs):
            mock_called[0] = True
            return orig_callbacks_process(*args, **kwargs)

        colorizer.callbacks.process = mock_process
        colorizer.norm = norm1  # Setting to the same object
        assert not mock_called[0]

        # Test with string norm
        colorizer.norm = "log"
        assert isinstance(colorizer.norm, mcolors.LogNorm)

        # Test callback connection
        orig_norm = colorizer.norm
        test_called = [False]

        def test_callback(*args):
            test_called[0] = True

        colorizer.callbacks.connect('changed', test_callback)
        orig_norm.callbacks.process('changed')  # Trigger change
        assert test_called[0]

    def test_to_rgba(self):
        colorizer = Colorizer()

        # Test scalar input
        result = colorizer.to_rgba(0.5)

        assert isinstance(result, tuple) and len(result) == 4
        assert all(isinstance(val, np.float64) for val in result)

        # Test array input
        data = np.linspace(0, 1, 10)
        result = colorizer.to_rgba(data)
        assert result.shape == (10, 4)

        # Test masked array
        masked_data = ma.masked_array([0.1, 0.2, 0.3], mask=[0, 1, 0])
        result = colorizer.to_rgba(masked_data)
        assert result.shape == (3, 4)

        # Test without normalization
        masked_data = ma.masked_array([0.1, 0.2, 0.3], mask=[0, 1, 0])
        result_masked = colorizer.to_rgba(masked_data)
        assert result_masked.shape == (3, 4)

        # Test with alpha
        result_alpha = colorizer.to_rgba(data, alpha=0.5)
        assert np.all(result_alpha[:, 3] == 0.5)

        # Test bytes output
        result_bytes = colorizer.to_rgba(data, bytes=True)
        assert result_bytes.dtype == np.uint8
        assert np.max(result_bytes) <= 255

    def test_pass_image_data(self):
        # Test RGB array (3 channels)
        rgb = np.ones((5, 5, 3))
        result = Colorizer._pass_image_data(rgb, alpha=0.5)
        assert result.shape == (5, 5, 4)
        assert np.all(result[:, :, 3] == 0.5)

        # Test RGBA array (4 channels)
        rgba = np.ones((5, 5, 4))
        rgba[:, :, 3] = 0.7
        result = Colorizer._pass_image_data(rgba)
        assert result.shape == (5, 5, 4)
        assert np.all(result[:, :, 3] == 0.7)

        # Test invalid shape
        with pytest.raises(ValueError, match="Third dimension must be 3 or 4"):
            Colorizer._pass_image_data(np.ones((5, 5, 5)))

        # Test uint8 input
        uint8_rgb = np.full((5, 5, 3), 128, dtype=np.uint8)
        result = Colorizer._pass_image_data(uint8_rgb)
        assert result.dtype == np.float32
        assert np.all(result[:, :, :3] == 128 / 255)

        # Test uint8 input with bytes=True
        result_bytes = Colorizer._pass_image_data(uint8_rgb, bytes=True)
        assert result_bytes.dtype == np.uint8
        assert np.all(result_bytes[:, :, :3] == 128)

        # Test float input with invalid range
        invalid_float = np.ones((5, 5, 3)) * 2  # values > 1
        with pytest.raises(ValueError, match="must be in the 0..1 range"):
            Colorizer._pass_image_data(invalid_float)

        # Test NaN handling
        float_with_nan = np.ones((5, 5, 3))
        float_with_nan[2, 2, 0] = np.nan
        result = Colorizer._pass_image_data(float_with_nan)
        assert np.all(result[2, 2] == 0)  # All zeros where NaN is present

        # Test masked array
        masked = ma.masked_array(
            np.ones((3, 3, 3)),
            mask=np.zeros((3, 3, 3), dtype=bool)
        )
        masked.mask[1, 1, 0] = True
        result = Colorizer._pass_image_data(masked)
        assert result[1, 1, 3] == 0  # Alpha is 0 where mask is True

    def test_scale_norm(self):
        colorizer = Colorizer()
        norm = mcolors.Normalize()
        A = np.linspace(0, 10, 10)

        # Test with vmin/vmax
        colorizer._scale_norm(None, vmin=2, vmax=8, A=A)
        assert colorizer.vmin == 2
        assert colorizer.vmax == 8

        # Test with norm instance and vmin/vmax
        with pytest.raises(ValueError, match="Passing a Normalize instance simultaneously"):
            colorizer._scale_norm(norm=norm, vmin=1, vmax=9, A=A)

    def test_autoscale(self):
        colorizer = Colorizer()
        A = np.array([2, 4, 6, 8, 10])

        # Test autoscale
        colorizer.autoscale(A)
        assert colorizer.vmin == 2
        assert colorizer.vmax == 10

        # Test with None array
        with pytest.raises(TypeError, match="You must first set_array"):
            colorizer.autoscale(None)

    def test_autoscale_None(self):
        colorizer = Colorizer()
        A = np.array([2, 4, 6, 8, 10])

        # Set initial limits
        colorizer.set_clim(0, 20)

        # Test autoscale_None - shouldn't change limits
        colorizer.autoscale_None(A)
        assert colorizer.vmin == 0
        assert colorizer.vmax == 20

        # Now set one limit to None
        colorizer.norm.vmin = None
        colorizer.autoscale_None(A)
        assert colorizer.vmin == 2  # Updated
        assert colorizer.vmax == 20  # Unchanged

        # Test with None array
        with pytest.raises(TypeError, match="You must first set_array"):
            colorizer.autoscale_None(None)

    def test_set_cmap(self):
        colorizer = Colorizer()
        orig_cmap = colorizer.cmap

        # Test changing colormap
        colorizer._set_cmap("viridis")
        assert colorizer.cmap.name == "viridis"
        assert colorizer.cmap is not orig_cmap

        # Test with callback
        test_called = [False]

        def test_callback(*args):
            test_called[0] = True

        colorizer.callbacks.connect('changed', test_callback)
        colorizer._set_cmap("plasma")
        assert test_called[0]

    def test_cmap_property(self):
        colorizer = Colorizer(cmap="inferno")
        assert colorizer.cmap.name == "inferno"

        # Test setter
        colorizer.cmap = "viridis"
        assert colorizer.cmap.name == "viridis"

    def test_set_clim(self):
        colorizer = Colorizer()
        colorizer.set_clim(2, 8)
        assert colorizer.vmin == 2
        assert colorizer.vmax == 8

        # Test single tuple argument
        colorizer.set_clim((3, 7))
        assert colorizer.vmin == 3
        assert colorizer.vmax == 7

        # Test only setting vmin
        colorizer.set_clim(vmin=5)
        assert colorizer.vmin == 5
        assert colorizer.vmax == 7

        # Test only setting vmax
        colorizer.set_clim(vmax=10)
        assert colorizer.vmin == 5
        assert colorizer.vmax == 10

    def test_get_clim(self):
        colorizer = Colorizer()
        colorizer.set_clim(3, 9)
        vmin, vmax = colorizer.get_clim()
        assert vmin == 3
        assert vmax == 9

    def test_changed(self):
        colorizer = Colorizer()

        # Test callback gets called
        test_called = [False]

        def test_callback(*args):
            test_called[0] = True

        colorizer.callbacks.connect('changed', test_callback)
        colorizer.changed()
        assert test_called[0]

        # Test stale flag gets set
        colorizer.stale = False
        colorizer.changed()
        assert colorizer.stale

    def test_vmin_vmax_properties(self):
        colorizer = Colorizer()
        colorizer.set_clim(2, 8)
        assert colorizer.vmin == 2
        assert colorizer.vmax == 8

        # Test setters
        colorizer.vmin = 3
        assert colorizer.vmin == 3
        assert colorizer.vmax == 8

        colorizer.vmax = 9
        assert colorizer.vmin == 3
        assert colorizer.vmax == 9

    def test_clip_property(self):
        colorizer = Colorizer()
        assert not colorizer.clip  # Default is False

        # Test setter
        colorizer.clip = True
        assert colorizer.clip

        colorizer.clip = False
        assert not colorizer.clip


class TestColorizerInterface:
    class DummyImplementation(_ColorizerInterface):
        def __init__(self):
            self._colorizer = Colorizer()
            self._A = np.array([1, 2, 3, 4, 5])

    def test_scale_norm(self):
        dummy = self.DummyImplementation()
        norm = mcolors.Normalize()
        dummy._scale_norm(None, 2, 8)
        assert dummy._colorizer.vmin == 2
        assert dummy._colorizer.vmax == 8

    def test_to_rgba(self):
        dummy = self.DummyImplementation()
        result = dummy.to_rgba(0.5)
        result_array = np.array(result)
        assert result_array.shape == (4,)

    def test_get_set_clim(self):
        dummy = self.DummyImplementation()
        dummy.set_clim(3, 7)
        vmin, vmax = dummy.get_clim()
        assert vmin == 3
        assert vmax == 7

    def test_get_alpha(self):
        dummy = self.DummyImplementation()
        assert dummy.get_alpha() == 1

    def test_cmap_property(self):
        dummy = self.DummyImplementation()
        assert dummy.cmap is not None

        # Test setter
        dummy.cmap = "viridis"
        assert dummy.cmap.name == "viridis"

        # Test get_cmap and set_cmap
        assert dummy.get_cmap().name == "viridis"
        dummy.set_cmap("plasma")
        assert dummy.get_cmap().name == "plasma"

    def test_autoscale(self):
        dummy = self.DummyImplementation()
        dummy._colorizer.set_clim(0, 10)
        dummy.autoscale()
        assert dummy._colorizer.vmin == 1
        assert dummy._colorizer.vmax == 5

    def test_autoscale_None(self):
        dummy = self.DummyImplementation()
        dummy.set_clim(None, 10)
        dummy.autoscale_None()
        assert dummy._colorizer.vmin == 1
        assert dummy._colorizer.vmax == 10

    def test_colorbar_property(self):
        dummy = self.DummyImplementation()
        assert dummy.colorbar is None

        # Test setter
        dummy.colorbar = "mock_colorbar"
        assert dummy.colorbar == "mock_colorbar"

    def test_format_cursor_data_override(self):
        dummy = self.DummyImplementation()

        # Test with masked data
        masked_data = ma.masked_array([0.5], mask=[True])[0]
        assert dummy._format_cursor_data_override(masked_data) == "[]"

        # Test with normal data
        assert "[5.0]" in dummy._format_cursor_data_override(5.0)

        # Test with boundary norm
        dummy.norm = mcolors.BoundaryNorm([0, 2, 4, 6, 8, 10], 5)
        assert "[5.]" in dummy._format_cursor_data_override(5.0)

        # Test with singular norm
        dummy.norm = mcolors.Normalize(5, 5)
        assert "[5.0]" in dummy._format_cursor_data_override(5.0)

        # Test with non-finite value
        assert "[nan]" in dummy._format_cursor_data_override(np.nan)


class TestScalarMappable:
    def test_init(self):
        # Test default initialization
        sm = _ScalarMappable()
        assert isinstance(sm._colorizer, Colorizer)
        assert sm._A is None

        # Test with custom norm and cmap
        norm = mcolors.Normalize(0, 10)
        cmap = "viridis"
        sm = _ScalarMappable(norm=norm, cmap=cmap)
        assert sm._colorizer.norm is norm
        assert sm._colorizer.cmap.name == "viridis"

        # Test with colorizer
        colorizer = Colorizer(cmap="plasma")
        sm = _ScalarMappable(colorizer=colorizer)
        assert sm._colorizer is colorizer

        # Test with conflicting parameters
        with pytest.raises(ValueError, match="cannot be used simultaneously"):
            _ScalarMappable(norm=norm, colorizer=colorizer)

    def test_set_array(self):
        sm = _ScalarMappable()

        # Test with None
        sm.set_array(None)
        assert sm._A is None

        # Test with array
        data = np.array([1.0, 2.0, 3.0])
        sm.set_array(data)
        assert np.array_equal(sm._A, data)

        # Test with invalid dtype
        with pytest.raises(TypeError, match="cannot be converted to float"):
            sm.set_array(np.array(["a", "b", "c"]))

        # Test autoscaling
        sm = _ScalarMappable()
        sm.set_array(data)
        assert sm._colorizer.vmin == 1.0
        assert sm._colorizer.vmax == 3.0

    def test_get_array(self):
        sm = _ScalarMappable()
        assert sm.get_array() is None

        data = np.array([1.0, 2.0, 3.0])
        sm.set_array(data)
        assert np.array_equal(sm.get_array(), data)

    def test_changed(self):
        sm = _ScalarMappable()

        # Test callback gets called
        test_called = [False]

        def test_callback(mappable):
            test_called[0] = True
            assert mappable is sm

        sm.callbacks.connect('changed', test_callback)
        sm.changed()
        assert test_called[0]

        # Test stale flag gets set
        sm.stale = False
        sm.changed()
        assert sm.stale

    def test_check_exclusionary_keywords(self):
        # Test with None colorizer
        _ScalarMappable._check_exclusionary_keywords(None, norm=None, cmap=None)

        # Test with colorizer and no other kwargs
        _ScalarMappable._check_exclusionary_keywords(Colorizer(), norm=None, cmap=None)

        # Test with colorizer and conflicting kwargs
        with pytest.raises(ValueError, match="cannot be used simultaneously"):
            _ScalarMappable._check_exclusionary_keywords(
                Colorizer(), norm=mcolors.Normalize(), cmap=None
            )

    def test_get_colorizer(self):
        # Test with colorizer
        colorizer = Colorizer()
        result = _ScalarMappable._get_colorizer(None, None, colorizer)
        assert result is colorizer

        # Test without colorizer
        norm = mcolors.Normalize()
        cmap = "viridis"
        result = _ScalarMappable._get_colorizer(cmap, norm, None)
        assert isinstance(result, Colorizer)
        assert result.norm is norm
        assert result.cmap.name == "viridis"


class MockArtist:
    def __init__(self):
        self.stale = False


class TestColorizingArtist:
    def test_init(self):
        colorizer = Colorizer()
        ca = ColorizingArtist(colorizer)
        assert ca._colorizer is colorizer

        # Test with invalid colorizer type
        with pytest.raises(TypeError,
                           match="'colorizer' must be an instance of matplotlib.colorizer.Colorizer, not a str"):
            ColorizingArtist("not a colorizer")

    def test_colorizer_property(self):
        colorizer1 = Colorizer(cmap="viridis")
        colorizer2 = Colorizer(cmap="plasma")
        ca = ColorizingArtist(colorizer1)
        assert ca.colorizer is colorizer1

        # Test setter
        ca.colorizer = colorizer2
        assert ca.colorizer is colorizer2

        # Test with invalid type
        with pytest.raises(TypeError, match="must be an instance of matplotlib.colorizer.Colorizer"):
            ca.colorizer = "not a colorizer"

    def test_set_colorizer_check_keywords(self):
        colorizer1 = Colorizer()
        colorizer2 = Colorizer()
        ca = ColorizingArtist(colorizer1)

        # Test valid case
        ca._set_colorizer_check_keywords(colorizer2)
        assert ca.colorizer is colorizer2

        # Test with conflicting keywords
        with pytest.raises(ValueError, match="cannot be used simultaneously"):
            ca._set_colorizer_check_keywords(colorizer2, norm=mcolors.Normalize())


class TestAutoNormFromScale:
    def test_with_common_scales(self):
        # Test with linear scale
        norm_cls = _auto_norm_from_scale(mscale.LinearScale)
        assert issubclass(norm_cls, mcolors.Normalize)

        # Test with log scale
        norm_cls = _auto_norm_from_scale(mscale.LogScale)
        assert issubclass(norm_cls, mcolors.LogNorm)

    def test_nonpositive_handling(self):
        # Test with a scale that handles non-positive values
        norm_cls = _auto_norm_from_scale(mscale.LogScale)

        # Create norm instance based on what parameters it accepts
        try:
            # Try with nonpositive parameter if supported
            norm = norm_cls(nonpositive='mask')
            # Test that it masks non-positive values
            assert np.ma.is_masked(norm(0))
        except TypeError:
            # If nonpositive is not supported, test basic functionality
            norm = norm_cls()
            # LogNorm should still mask/clip non-positive values somehow
            result = norm(0)
            assert result == 0 or np.ma.is_masked(result) or np.isnan(result)
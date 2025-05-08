import numpy as np
import pytest

import matplotlib as mpl
from matplotlib import cm
from matplotlib import colors


class TestColormapRegistryExtended:
    @pytest.mark.parametrize("cmap_name,expected_type", [
        ('viridis', colors.ListedColormap),
        ('jet', colors.LinearSegmentedColormap),
        ('tab10', colors.ListedColormap),
        ('Paired', colors.ListedColormap),
        ('Blues', colors.LinearSegmentedColormap),
        ('hot', colors.LinearSegmentedColormap),
        ('gist_gray', colors.LinearSegmentedColormap),
        ('binary', colors.LinearSegmentedColormap),
        ('Spectral', colors.LinearSegmentedColormap),
        ('coolwarm', colors.LinearSegmentedColormap),
    ])
    def test_builtin_colormap_types(self, cmap_name, expected_type):
        """Test that different builtin colormaps have the expected types."""
        cmap = mpl.colormaps[cmap_name]
        assert isinstance(cmap, expected_type), f"Colormap '{cmap_name}' is {type(cmap).__name__}, not {expected_type.__name__}"

    def test_ensure_cmap(self):
        """Test the _ensure_cmap utility function."""
        # With string
        cmap1 = cm._ensure_cmap('viridis')
        assert isinstance(cmap1, colors.Colormap)
        assert cmap1.name == 'viridis'

        # With Colormap instance
        original = mpl.colormaps['plasma']
        cmap2 = cm._ensure_cmap(original)
        assert cmap2 is original  # Should return the same object

        # With None (uses default from rcParams)
        default_name = mpl.rcParams["image.cmap"]
        cmap3 = cm._ensure_cmap(None)
        assert cmap3.name == default_name

        # With invalid name
        with pytest.raises(ValueError):
            cm._ensure_cmap('nonexistent_colormap')

    def test_resampling(self):
        """Test colormap resampling with different lut values."""
        # Get the original with default lut size
        original = mpl.colormaps['viridis']

        # Custom lut values to test
        lut_values = [8, 64, 256, 1024]

        for lut in lut_values:
            # Fixed: Using square brackets to access the colormap and passing lut as a parameter
            resampled = mpl.colormaps['viridis'].resampled(lut)
            assert len(resampled.colors) == lut

    def test_register_same_cmap_different_names(self):
        """Test registering the same colormap with different names."""
        # Create a test colormap
        test_cmap = colors.LinearSegmentedColormap.from_list('base_name', ['red', 'blue'])

        # Register with multiple names
        names = ['test_name1', 'test_name2', 'test_name3']

        try:
            for name in names:
                mpl.colormaps.register(test_cmap, name=name)

            # Verify all names are registered
            for name in names:
                assert name in mpl.colormaps
                assert mpl.colormaps[name].name == name

            # Verify they all generate the same colors
            colors1 = mpl.colormaps['test_name1'](0.5)
            colors2 = mpl.colormaps['test_name2'](0.5)
            colors3 = mpl.colormaps['test_name3'](0.5)

            assert np.array_equal(colors1, colors2)
            assert np.array_equal(colors1, colors3)

        finally:
            # Clean up
            for name in names:
                try:
                    mpl.colormaps.unregister(name)
                except Exception:
                    pass

    def test_listed_colormap_registration(self):
        """Test registering and using ListedColormap."""
        colors_list = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # RGB colors
        cmap = colors.ListedColormap(colors_list, name='rgb_test')

        try:
            # Register the colormap
            mpl.colormaps.register(cmap)

            # Test retrieving and using the colormap
            retrieved = mpl.colormaps['rgb_test']
            assert isinstance(retrieved, colors.ListedColormap)
            assert len(retrieved.colors) == 3

            # Check colors are correct - comparing RGB values only (first 3 values)
            assert np.allclose(retrieved.colors[0][:3], [1, 0, 0])  # Red
            assert np.allclose(retrieved.colors[1][:3], [0, 1, 0])  # Green
            assert np.allclose(retrieved.colors[2][:3], [0, 0, 1])  # Blue

            # Test mapping values
            # For N=3 colors in ListedColormap:
            # Indices 0, 1, 2 correspond to values 0, 0.5, 1.0 in the [0,1] interval
            # So we need to test with values that map correctly to these indices

            # These index calculations handle any number of colors:
            n_colors = len(colors_list)
            idx0 = 0 / (n_colors - 1) if n_colors > 1 else 0  # First index
            idx1 = 1 / (n_colors - 1) if n_colors > 1 else 0  # Second index
            idx2 = 2 / (n_colors - 1) if n_colors > 1 else 0  # Third index

            # Get mapped colors (only RGB components)
            mapped_color_0 = retrieved(idx0)[:3]
            mapped_color_1 = retrieved(idx1)[:3]
            mapped_color_2 = retrieved(idx2)[:3]

            # Now assert against the expected values
            assert np.allclose(mapped_color_0, [1, 0, 0])  # First color (red)
            assert np.allclose(mapped_color_1, [0, 1, 0])  # Second color (green)
            assert np.allclose(mapped_color_2, [0, 0, 1])  # Third color (blue)

        finally:
            # Clean up
            try:
                mpl.colormaps.unregister('rgb_test')
            except KeyError:
                pass  # Colormap wasn't registered or already unregistered


    def test_colormap_application_to_data(self):
        """Test applying colormaps to data arrays."""
        # Create a simple data array
        data = np.linspace(0, 1, 10)

        # Test with a few different colormaps
        cmaps_to_test = ['viridis', 'plasma', 'inferno', 'magma']

        for cmap_name in cmaps_to_test:
            cmap = mpl.colormaps[cmap_name]

            # Apply the colormap to data
            mapped_colors = cmap(data)

            # Check the shape and type of result
            assert mapped_colors.shape == (10, 4)  # RGBA values for each data point
            assert mapped_colors.dtype == np.float64

            # Check bounds - all values should be between 0 and 1
            assert np.all(mapped_colors >= 0)
            assert np.all(mapped_colors <= 1)

    def test_get_cmap_with_none(self):
        """Test get_cmap with None uses the default from rcParams."""
        default_name = mpl.rcParams['image.cmap']
        default_cmap = mpl.colormaps[default_name]

        # Temporarily save the current setting
        old_rcparam = mpl.rcParams['image.cmap']

        try:
            # Test with default setting - using the correct API
            cmap1 = mpl.colormaps.get_cmap(None)  # Using get_cmap from colormaps
            assert cmap1.name == default_name

            # Change the default and test again
            mpl.rcParams['image.cmap'] = 'plasma'
            cmap2 = mpl.colormaps.get_cmap(None)  # Using get_cmap from colormaps
            assert cmap2.name == 'plasma'

        finally:
            # Restore the original setting
            mpl.rcParams['image.cmap'] = old_rcparam
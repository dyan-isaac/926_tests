import numpy as np
import pytest
from matplotlib import colors as mcolors
from matplotlib.colors import LightSource, to_rgb, to_rgba, to_rgba_array, to_hex


# ------------------ Partitioned Color Conversion Tests ------------------

@pytest.mark.parametrize("input_color, expected", [
    ("red", (1.0, 0.0, 0.0, 1.0)),
    ("#00ff00", (0.0, 1.0, 0.0, 1.0)),
    ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 1.0)),
    ((0.2, 0.4, 0.6, 0.8), (0.2, 0.4, 0.6, 0.8)),
    ("none", (0.0, 0.0, 0.0, 0.0))
])
def test_to_rgba_partition(input_color, expected):
    assert to_rgba(input_color) == expected


@pytest.mark.parametrize("color, alpha, expected", [
    ("blue", 0.5, (0.0, 0.0, 1.0, 0.5)),
    ((1, 0, 0), 0.3, (1, 0, 0, 0.3)),
    ((0.2, 0.2, 0.2, 0.1), 0.8, (0.2, 0.2, 0.2, 0.8)),
])
def test_to_rgba_with_explicit_alpha(color, alpha, expected):
    assert to_rgba(color, alpha) == expected


def test_to_rgba_array_mixed_inputs():
    color_list = ["red", (0.5, 0.5, 0.5), "#123456", "none"]
    result = to_rgba_array(color_list)
    assert result.shape == (4, 4)
    assert (result[-1] == (0, 0, 0, 0)).all()


def test_to_rgba_array_with_alpha_vector():
    colors = ["red", "blue"]
    alphas = [0.3, 0.6]
    result = to_rgba_array(colors, alphas)
    assert result[0, -1] == 0.3
    assert result[1, -1] == 0.6


def test_to_rgb_drops_alpha():
    assert to_rgb((1.0, 0.5, 0.2, 0.7)) == (1.0, 0.5, 0.2)


def test_to_hex_alpha_behavior():
    assert to_hex((1.0, 0.5, 0.0), keep_alpha=False) == "#ff8000"
    assert to_hex((1.0, 0.5, 0.0, 0.5), keep_alpha=True) == "#ff800080"


# ------------------ LightSource and Shading Edge Cases ------------------

def test_lightsource_direction_unit_vector():
    ls = LightSource(azdeg=90, altdeg=0)
    dir = ls.direction
    assert np.allclose(np.linalg.norm(dir), 1.0)


def test_lightsource_hillshade_shape():
    z = np.random.rand(10, 10)
    ls = LightSource()
    result = ls.hillshade(z)
    assert result.shape == z.shape
    assert (0 <= result).all() and (result <= 1).all()


def test_lightsource_shade_normals_bounds():
    normals = np.random.rand(10, 10, 3)
    normals /= np.linalg.norm(normals, axis=2)[..., None]
    ls = LightSource()
    shaded = ls.shade_normals(normals)
    assert (0 <= shaded).all() and (shaded <= 1).all()


# ------------------ Boundary/Vector Alpha Handling ------------------

# def test_rgba_array_vector_alpha_tile_single_color():
#     color = [(0.1, 0.2, 0.3)]
#     alphas = np.linspace(0, 1, 5)
#     result = to_rgba_array(color, alphas)
#     assert result.shape == (5, 4)
#     assert np.allclose(result[:, :3], [color[0]] * 5)
#     assert np.allclose(result[:, 3], alphas)


# def test_rgba_array_shape_mismatch_raises():
#     colors = [(1, 0, 0), (0, 1, 0)]
#     alphas = [0.5]
#     with pytest.raises(ValueError):
#         to_rgba_array(colors, alphas)


def test_rgba_array_out_of_bounds_clipped():
    bad = np.array([[2, 0, 0, 1], [0, -1, 0, 1]])
    with pytest.raises(ValueError):
        to_rgba_array(bad)


# ------------------ HSV-RGB Round Trip ------------------

def test_hsv_rgb_round_trip_stability():
    hsv = np.random.rand(10, 3)
    rgb = mcolors.hsv_to_rgb(hsv)
    assert np.allclose(hsv, mcolors.rgb_to_hsv(rgb), atol=1e-6)

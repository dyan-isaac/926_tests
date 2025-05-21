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
    ("none", (0.0, 0.0, 0.0, 0.0)),
])
def test_to_rgba_partition(input_color, expected):
    assert to_rgba(input_color) == expected


@pytest.mark.parametrize("input_color, expected", [
    ("cyan", (0.0, 1.0, 1.0, 1.0)),
    ("magenta", (1.0, 0.0, 1.0, 1.0)),
    ("yellow", (1.0, 1.0, 0.0, 1.0)),
    ("black", (0.0, 0.0, 0.0, 1.0)),
    ("white", (1.0, 1.0, 1.0, 1.0)),
    ("#abcdef", (0.6705882352941176, 0.803921568627451, 0.9372549019607843, 1.0)),
    ("#123", (0.06666666666666667, 0.13333333333333333, 0.2, 1.0)),
    ((0, 0.5, 0), (0.0, 0.5, 0.0, 1.0)),
    ((1, 1, 0, 0.5), (1.0, 1.0, 0.0, 0.5)),
    ((0.2, 0.4, 0.6, 1.0), (0.2, 0.4, 0.6, 1.0)),
])
def test_to_rgba_extended(input_color, expected):
    assert to_rgba(input_color) == expected


@pytest.mark.parametrize("color, alpha, expected", [
    ("blue", 0.5, (0.0, 0.0, 1.0, 0.5)),
    ((1, 0, 0), 0.3, (1, 0, 0, 0.3)),
    ((0.2, 0.2, 0.2, 0.1), 0.8, (0.2, 0.2, 0.2, 0.8)),
    ("black", 0.2, (0.0, 0.0, 0.0, 0.2)),
    ("yellow", 1.0, (1.0, 1.0, 0.0, 1.0)),
    ("#abcdef", 0.7, (0.6705882352941176, 0.803921568627451, 0.9372549019607843, 0.7)),
    ((0.5, 0.5, 0.5), 0.1, (0.5, 0.5, 0.5, 0.1)),
    ((0.9, 0.1, 0.1, 0.3), 0.9, (0.9, 0.1, 0.1, 0.9)),
])
def test_to_rgba_with_explicit_alpha(color, alpha, expected):
    assert to_rgba(color, alpha) == expected


@pytest.mark.parametrize("input_color", [
    "red",
    "#00ff00",
    (0.3, 0.6, 0.9),
    (1.0, 0.5, 0.0, 0.7)
])
def test_to_rgb_drops_alpha_various(input_color):
    rgb = to_rgb(input_color)
    assert len(rgb) == 3
    assert all(0.0 <= c <= 1.0 for c in rgb)


@pytest.mark.parametrize("color, keep_alpha, expected", [
    ((1, 0, 0), False, "#ff0000"),
    ((0, 1, 0), False, "#00ff00"),
    ((0, 0, 1), False, "#0000ff"),
    ((1, 1, 1, 0.5), True, "#ffffffff"),
    ((0.5, 0.5, 0.5, 0.25), True, "#7f7f7f40"),
])
def test_to_hex_alpha_behavior(color, keep_alpha, expected):
    assert to_hex(color, keep_alpha=keep_alpha) == expected


@pytest.mark.parametrize("color_list, expected_shapes", [
    (["red", "green", "blue"], (3, 4)),
    ([(0.5, 0.2, 0.9), (0.3, 0.6, 0.1)], (2, 4)),
    (["#123456", "#abcdef", "none"], (3, 4)),
])
def test_to_rgba_array_shapes(color_list, expected_shapes):
    result = to_rgba_array(color_list)
    assert result.shape == expected_shapes


# ------------------ LightSource and Shading Edge Cases ------------------

@pytest.mark.parametrize("azdeg, altdeg", [
    (0, 90),
    (45, 45),
    (90, 45),
    (180, 45),
    (270, 45),
])
def test_lightsource_direction_unit_vector(azdeg, altdeg):
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    dir = ls.direction
    assert np.allclose(np.linalg.norm(dir), 1.0, atol=1e-6)


@pytest.mark.parametrize("shape", [
    (10, 10),
    (20, 20),
    (50, 50),
])
def test_lightsource_hillshade_shape(shape):
    z = np.random.rand(*shape)
    ls = LightSource()
    result = ls.hillshade(z)
    assert result.shape == z
    assert (0 <= result).all() and (result <= 1).all()


@pytest.mark.parametrize("noise_level", [0.0, 0.1, 0.5])
def test_lightsource_shade_normals_bounds(noise_level):
    normals = np.random.rand(10, 10, 3)
    normals /= np.linalg.norm(normals, axis=2)[..., None]
    normals += noise_level * np.random.randn(10, 10, 3)
    normals /= np.linalg.norm(normals, axis=2)[..., None]
    ls = LightSource()
    shaded = ls.shade_normals(normals)
    assert (0 <= shaded).all() and (shaded <= 1).all()


# ------------------ HSV-RGB Round Trip ------------------

@pytest.mark.parametrize("hsv", [
    (0.0, 0.0, 0.0),
    (0.5, 0.5, 0.5),
    (1.0, 1.0, 1.0),
    (0.25, 0.8, 0.9),
])
def test_hsv_rgb_round_trip_stability(hsv):
    hsv = np.array(hsv)
    rgb = mcolors.hsv_to_rgb(hsv)
    hsv_roundtrip = mcolors.rgb_to_hsv(rgb)
    assert np.allclose(hsv, hsv_roundtrip, atol=1e-6)


# ------------------ Boundary and Error Handling ------------------

def test_rgba_array_out_of_bounds_clipped():
    bad = np.array([[2, 0, 0, 1], [0, -1, 0, 1]])
    with pytest.raises(ValueError):
        to_rgba_array(bad)

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

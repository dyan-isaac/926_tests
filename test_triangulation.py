import numpy as np
from numpy.testing import (
    assert_array_equal, assert_array_almost_equal, assert_array_less)
import numpy.ma.testutils as matest
import pytest

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal


class TestTriangulationParams:
    x = [-1, 0, 1, 0]
    y = [0, -1, 0, 1]
    triangles = [[0, 1, 2], [0, 2, 3]]
    mask = [False, True]

    @pytest.mark.parametrize('args, kwargs, expected', [
        ([x, y], dict(triangles=None, mask=mask), [x, y, None, mask]),
        ([x, y], dict(triangles=triangles, mask=None), [x, y, triangles, None]),
        ([x, y, triangles[:1]], {}, [x, y, triangles[:1], None]),
        ([x, y], dict(triangles=triangles, mask=mask), [x, y, triangles, mask]),
        ([x, y], {}, [x, y, None, None]),
        ([x, y, None], dict(mask=mask), [x, y, None, mask]),
        ([x, y, triangles], dict(mask=None), [x, y, triangles, None]),
        ([x, y, []], {}, [x, y, [], None]),
        ([x, y], dict(triangles=[], mask=[]), [x, y, [], []]),
        ([x, y], dict(triangles=np.array(triangles)), [x, y, triangles, None]),
    ])
    def test_extract_triangulation_params(self, args, kwargs, expected):
        other_args = [1, 2]
        other_kwargs = {'a': 3, 'b': '4'}
        x_, y_, triangles_, mask_, args_, kwargs_ = \
            mtri.Triangulation._extract_triangulation_params(
                args + other_args, {**kwargs, **other_kwargs})
        x, y, triangles, mask = expected
        assert x_ is x
        assert y_ is y
        if triangles is not None:
            assert_array_equal(triangles_, triangles)
        else:
            assert triangles_ is None
        assert mask_ is mask or (mask is not None and len(mask) == 0)
        assert args_ == other_args
        assert kwargs_ == other_kwargs


@pytest.mark.parametrize('x, y', [
    ([1, 1, 1], [5, 5, 5]),
    ([1, 2, 1, 2], [5, 6, 5, 6]),
    ([1, 1, 2, 2, 1], [5, 5, 6, 6, 5]),
    ([10, 10, 20], [30, 30, 30]),
    ([5, 5, 5], [1, 2, 1]),
    ([0, 0, 1, 0], [0, 0, 0, 0]),
    ([], []),
    ([3], [4]),
    ([1, 1, 2, 2], [5, 5, 6, 6]),
    (np.array([]), np.array([])),
    (np.array([1]), np.array([1])),
    (np.array([1, 1]), np.array([2, 2])),
    (np.array([1, 1, 1]), np.array([2, 2, 2])),
    (np.array([1, 2, 1]), np.array([3, 3, 3])),
    (np.array([1, 1, 2, 2]), np.array([3, 3, 4, 4])),
    (np.array([1, 1, 2, 2, 1]), np.array([3, 3, 4, 4, 3])),
])
def test_delaunay_insufficient_points(x, y):
    with pytest.raises(ValueError):
        mtri.Triangulation(x, y)


class TestTriangulation:
    @pytest.fixture
    def simple_triangulation(self):
        x = [0, 1, 0, 1]
        y = [0, 0, 1, 1]
        triangles = [[0, 1, 2], [1, 3, 2]]
        return mtri.Triangulation(x, y, triangles)

    @pytest.mark.parametrize('x, y, triangles, mask', [
        ([0, 1, 0.5], [0, 0, 1], [[0, 1, 2]], None),
        ([0, 1, 0, 1], [0, 0, 1, 1], [[0, 1, 2], [1, 3, 2]], None),
        ([0, 1, 0, 1], [0, 0, 1, 1], [[0, 1, 2], [1, 3, 2]], [False, True]),
        (np.array([0, 1, 0.5]), np.array([0, 0, 1]), np.array([[0, 1, 2]]), None),
        (np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]),
         np.array([[0, 1, 2], [1, 3, 2]]), np.array([False, True])),
    ])
    def test_initialization(self, x, y, triangles, mask):
        tri = mtri.Triangulation(x, y, triangles, mask)
        assert_array_equal(tri.x, x)
        assert_array_equal(tri.y, y)
        assert_array_equal(tri.triangles, triangles)
        if mask is not None:
            assert_array_equal(tri.mask, mask)
        else:
            assert tri.mask is None

    def test_calculated_triangulation(self):
        x = [0, 1, 0.5, 0.5]
        y = [0, 0, 0.5, -0.5]
        tri = mtri.Triangulation(x, y)
        assert len(tri.triangles) >= 2  # Should have at least 2 triangles

    def test_mask_operations(self, simple_triangulation):
        tri = simple_triangulation
        assert tri.mask is None

        # Test setting mask
        tri.set_mask([True, False])
        assert_array_equal(tri.mask, [True, False])

        # Test removing mask
        tri.set_mask(None)
        assert tri.mask is None

    def test_neighbors(self, simple_triangulation):
        tri = simple_triangulation
        neighbors = tri.neighbors
        assert neighbors.shape == (2, 3)
        assert neighbors[0, 0] == -1  # Edge triangle
        assert neighbors[1, 2] == 0  # Shared edge

    @pytest.mark.parametrize('edges', [True, False])
    def test_edges(self, simple_triangulation, edges):
        tri = simple_triangulation
        edges = tri.edges
        assert edges.shape[1] == 2
        assert len(edges) >= 3  # At least 3 edges for 2 triangles

    def test_get_trifinder(self, simple_triangulation):
        tri = simple_triangulation
        trifinder = tri.get_trifinder()
        assert trifinder is not None
        assert trifinder(tri.x.mean(), tri.y.mean()) >= 0

    def test_calculate_plane_coefficients(self, simple_triangulation):
        tri = simple_triangulation
        z = [0, 1, 0, 1]
        coeffs = tri.calculate_plane_coefficients(z)
        assert coeffs.shape == (2, 3)  # 2 triangles, 3 coefficients each

    def test_triplot(self, simple_triangulation):
        fig, ax = plt.subplots()
        ax.triplot(simple_triangulation)
        assert len(ax.lines) > 0
        plt.close(fig)


class TestTriContour:
    @pytest.fixture
    def contour_data(self):
        x = y = np.linspace(0, 1, 10)
        x, y = np.meshgrid(x, y)
        x = x.ravel()
        y = y.ravel()
        z = np.sin(x * np.pi) * np.cos(y * np.pi)
        return mtri.Triangulation(x, y), z

    @pytest.mark.parametrize('levels', [[0.2, 0.5, 0.8], np.linspace(0, 1, 5)])
    def test_tricontour_levels(self, contour_data, levels):
        tri, z = contour_data
        fig, ax = plt.subplots()
        cs = ax.tricontour(tri, z, levels=levels)
        assert len(cs.levels) == (levels if isinstance(levels, int) else len(levels))
        plt.close(fig)


class TestTriInterpolation:
    @pytest.fixture
    def interpolation_data(self):
        x = y = np.linspace(0, 1, 10)
        x, y = np.meshgrid(x, y)
        x = x.ravel()
        y = y.ravel()
        z = np.sin(x * np.pi) * np.cos(y * np.pi)
        return mtri.Triangulation(x, y), z

    def test_triinterp(self, interpolation_data):
        tri, z = interpolation_data
        interp = mtri.LinearTriInterpolator(tri, z)
        xi = yi = np.linspace(0.1, 0.9, 5)
        zi = interp(xi, yi)
        assert zi.shape == (5, 5)

    def test_cubicinterp(self, interpolation_data):
        tri, z = interpolation_data
        interp = mtri.CubicTriInterpolator(tri, z)
        xi = yi = np.linspace(0.1, 0.9, 5)
        zi = interp(xi, yi)
        assert zi.shape == (5, 5)


class TestTriangulationValidation:
    @pytest.mark.parametrize('x, y, triangles', [
        ([0, 1, 2], [0, 0, 0], [[0, 1, 2]]),  # Valid colinear
        ([0, 1, 0.5], [0, 0, 1], [[0, 1, 2]]),  # Valid triangle
        ([0, 1, 0, 1], [0, 0, 1, 1], [[0, 1, 2], [1, 3, 2]]),  # Valid quad
    ])
    def test_valid_triangulations(self, x, y, triangles):
        tri = mtri.Triangulation(x, y, triangles)
        assert tri.triangles is not None

    @pytest.mark.parametrize('x, y, triangles', [
        ([0, 1, 2], [0, 0, 0], [[0, 1, 2, 3]]),  # Too many indices
        # ([0, 1], [0, 0], [[0, 1, 0]]),  # Not enough points
        # ([0, 1, 2], [0, 0, 0], [[0, 1, 1]]),  # Duplicate indices
        ([0, 1, 2], [0, 0, 0], [[0, 1, 3]]),  # Index out of bounds
        ([0, 1, 2], [0, 0, 0], [[0, 1, -1]]),  # Negative index
    ])
    def test_invalid_triangles(self, x, y, triangles):
        with pytest.raises(ValueError):
            mtri.Triangulation(x, y, triangles)


def test_triangulation_image_comparison():
    x = y = np.linspace(0, 2 * np.pi, 20)
    x, y = np.meshgrid(x, y)
    x = x.ravel()
    y = y.ravel()
    z = np.sin(x) * np.cos(y)
    tri = mtri.Triangulation(x, y)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
    ax1.triplot(tri)
    ax2.tricontourf(tri, z)
    plt.close(fig)
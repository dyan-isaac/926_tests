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
        ([x, y], dict(triangles=None, mask=mask), [x, y, None, mask]),  # Explicit None triangles
        ([x, y], dict(triangles=triangles, mask=None), [x, y, triangles, None]),  # Explicit None mask
        # Tuple inputs
        ([x, y, triangles[:1]], {}, [x, y, triangles[:1], None]),  # Partial triangles list
        ([x, y], dict(triangles=triangles, mask=mask), [x, y, triangles, mask]),  # Both specified
        ([x, y], {}, [x, y, None, None]),  # Neither triangles nor mask specified
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
        assert_array_equal(triangles_, triangles)
        assert mask_ is mask
        assert args_ == other_args
        assert kwargs_ == other_kwargs


@pytest.mark.parametrize('x, y', [
    # Triangulation should raise a ValueError if passed less than 3 points.
    ([1, 1, 1], [5, 5, 5]),  # All points the same
    ([1, 2, 1, 2], [5, 6, 5, 6]),  # Only 2 unique points alternating
    ([1, 1, 2, 2, 1], [5, 5, 6, 6, 5]),  # 2 unique points with repeats
    ([10, 10, 20], [30, 30, 30]),  # 2 unique x values, 1 unique y value
    ([5, 5, 5], [1, 2, 1]),  # 1 unique x value, 2 unique y values
    ([0, 0, 1, 0], [0, 0, 0, 0]),  # Multiple zeros with only 2 unique points
    ([], []),  # Empty arrays
    ([3], [4]),  # Single point
    ([1, 1, 2, 2], [5, 5, 6, 6]),  # Two unique points in sequence
])
def test_delaunay_insufficient_points(x, y):
    with pytest.raises(ValueError):
        mtri.Triangulation(x, y)


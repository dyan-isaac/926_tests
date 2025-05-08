import pytest
import numpy as np
from matplotlib.transforms import (
    Bbox, Affine2D, IdentityTransform, TransformedBbox, BlendedGenericTransform,
    BlendedAffine2D, composite_transform_factory, nonsingular, interval_contains,
    interval_contains_open, offset_copy
)
from matplotlib.path import Path


def test_bbox_creation():
    bbox = Bbox([[0, 0], [1, 1]])
    assert bbox.x0 == 0
    assert bbox.y0 == 0
    assert bbox.x1 == 1
    assert bbox.y1 == 1
    assert bbox.width == 1
    assert bbox.height == 1


def test_bbox_transformation():
    bbox = Bbox([[0, 0], [1, 1]])
    transform = Affine2D().scale(2)
    transformed_bbox = bbox.transformed(transform)
    assert transformed_bbox.x0 == 0
    assert transformed_bbox.y0 == 0
    assert transformed_bbox.x1 == 2
    assert transformed_bbox.y1 == 2


def test_affine2d_rotation():
    transform = Affine2D().rotate_deg(90)
    point = np.array([1, 0])
    transformed_point = transform.transform(point)
    np.testing.assert_array_almost_equal(transformed_point, [0, 1])


def test_identity_transform():
    transform = IdentityTransform()
    point = np.array([1, 2])
    transformed_point = transform.transform(point)
    np.testing.assert_array_equal(transformed_point, point)


def test_transformed_bbox():
    bbox = Bbox([[0, 0], [1, 1]])
    transform = Affine2D().scale(2)
    transformed_bbox = TransformedBbox(bbox, transform)
    points = transformed_bbox.get_points()
    np.testing.assert_array_almost_equal(points, [[0, 0], [2, 2]])


def test_blended_generic_transform():
    x_transform = Affine2D().scale(2)
    y_transform = Affine2D().scale(3)
    blended_transform = BlendedGenericTransform(x_transform, y_transform)
    points = np.array([[1, 1]])
    transformed_points = blended_transform.transform(points)
    np.testing.assert_array_almost_equal(transformed_points, [[2, 3]])


def test_blended_affine2d():
    x_transform = Affine2D().scale(2)
    y_transform = Affine2D().scale(3)
    blended_transform = BlendedAffine2D(x_transform, y_transform)
    points = np.array([[1, 1]])
    transformed_points = blended_transform.transform(points)
    np.testing.assert_array_almost_equal(transformed_points, [[2, 3]])


def test_composite_transform_factory():
    transform_a = Affine2D().scale(2)
    transform_b = Affine2D().translate(1, 1)
    composite_transform = composite_transform_factory(transform_a, transform_b)
    points = np.array([[1, 1]])
    transformed_points = composite_transform.transform(points)
    np.testing.assert_array_almost_equal(transformed_points, [[3, 3]])


@pytest.mark.parametrize("vmin, vmax, expected", [
    (0, 1, (0, 1)),  # Normal range
    (1, 0, (0, 1)),  # Swapped range
    (0, 0, (-0.001, 0.001)),  # Zero range
    (np.nan, 1, (-0.001, 0.001)),  # NaN input
])
def test_nonsingular(vmin, vmax, expected):
    result = nonsingular(vmin, vmax)
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize("interval, val, expected", [
    ((0, 1), 0.5, True),  # Inside interval
    ((0, 1), 1, True),  # On the boundary
    ((0, 1), -1, False),  # Outside interval
])
def test_interval_contains(interval, val, expected):
    assert interval_contains(interval, val) == expected


@pytest.mark.parametrize("interval, val, expected", [
    ((0, 1), 0.5, True),  # Inside interval
    ((0, 1), 1, False),  # On the boundary
    ((0, 1), -1, False),  # Outside interval
])
def test_interval_contains_open(interval, val, expected):
    assert interval_contains_open(interval, val) == expected


def test_offset_copy():
    transform = Affine2D()
    offset_transform = offset_copy(transform, x=1, y=1, units="dots")
    point = np.array([0, 0])
    transformed_point = offset_transform.transform(point)
    np.testing.assert_array_almost_equal(transformed_point, [1, 1])
import pytest
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec, SubplotSpec, SubplotParams


@pytest.mark.parametrize("nrows, ncols, expected", [
    (2, 3, (2, 3)),  # Normal grid
    (1, 1, (1, 1)),  # Single cell grid
])
def test_gridspec_creation(nrows, ncols, expected):
    gs = GridSpec(nrows, ncols)
    assert gs.get_geometry() == expected


@pytest.mark.parametrize("nrows, ncols", [
    (0, 3),  # Invalid rows
    (2, 0),  # Invalid columns
])
def test_gridspec_invalid_dimensions(nrows, ncols):
    with pytest.raises(ValueError, match="must be a positive integer"):
        GridSpec(nrows, ncols)


@pytest.mark.parametrize("width_ratios, height_ratios, expected_width, expected_height", [
    ([1, 2, 3], [4, 5], [1, 2, 3], [4, 5]),  # Custom ratios
    (None, None, None, None),  # Default ratios
])
def test_gridspec_width_height_ratios(width_ratios, height_ratios, expected_width, expected_height):
    gs = GridSpec(2, 3, width_ratios=width_ratios, height_ratios=height_ratios)
    assert gs.get_width_ratios() == expected_width
    assert gs.get_height_ratios() == expected_height


@pytest.mark.parametrize("key, expected_rowspan, expected_colspan", [
    ((0, 1), range(0, 1), range(1, 2)),  # Single cell
    ((slice(0, 2), slice(1, 3)), range(0, 2), range(1, 3)),  # Multiple cells
])
def test_gridspec_getitem(key, expected_rowspan, expected_colspan):
    gs = GridSpec(2, 3)
    ss = gs[key]
    assert isinstance(ss, SubplotSpec)
    assert ss.rowspan == expected_rowspan
    assert ss.colspan == expected_colspan


@pytest.mark.parametrize("squeeze, expected_type", [
    (True, plt.Axes),  # Single subplot
    (False, list),  # Always return 2D array
])
def test_gridspec_subplots_squeeze(squeeze, expected_type):
    fig = plt.figure()
    gs = GridSpec(1, 1, figure=fig)
    ax = gs.subplots(squeeze=squeeze)
    assert isinstance(ax, expected_type)


@pytest.mark.parametrize("left, right, top, bottom, wspace, hspace", [
    (0.1, 0.9, 0.9, 0.1, 0.2, 0.3),  # Custom parameters
    (None, None, None, None, None, None),  # Default parameters
])
def test_gridspec_update(left, right, top, bottom, wspace, hspace):
    gs = GridSpec(2, 2, left=0.1, right=0.9, top=0.9, bottom=0.1)
    gs.update(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)
    assert gs.left == (left if left is not None else 0.1)
    assert gs.right == (right if right is not None else 0.9)
    assert gs.top == (top if top is not None else 0.9)
    assert gs.bottom == (bottom if bottom is not None else 0.1)


@pytest.mark.parametrize("nrows, ncols, expected", [
    (2, 2, (2, 2)),  # Normal grid
    (1, 3, (1, 3)),  # Single row
])
def test_gridspec_from_subplot_spec(nrows, ncols, expected):
    fig = plt.figure()
    gs = GridSpec(2, 2, figure=fig)
    sub_gs = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs[0, 0])
    assert sub_gs.get_geometry() == expected


@pytest.mark.parametrize("left, right, bottom, top, wspace, hspace", [
    (0.1, 0.9, 0.1, 0.9, 0.2, 0.3),  # Custom parameters
    (None, None, None, None, None, None),  # Default parameters
])
def test_subplot_params_update(left, right, bottom, top, wspace, hspace):
    sp = SubplotParams(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.3)
    sp.update(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    assert sp.left == (left if left is not None else 0.1)
    assert sp.right == (right if right is not None else 0.9)
    assert sp.bottom == (bottom if bottom is not None else 0.1)
    assert sp.top == (top if top is not None else 0.9)
    assert sp.wspace == (wspace if wspace is not None else 0.2)
    assert sp.hspace == (hspace if hspace is not None else 0.3)


@pytest.mark.parametrize("left, right, bottom, top", [
    (1.0, 0.9, 0.1, 0.9),  # Invalid left >= right
    (0.1, 0.9, 1.0, 0.9),  # Invalid bottom >= top
])
def test_subplot_params_invalid_update(left, right, bottom, top):
    sp = SubplotParams(left=0.1, right=0.9, bottom=0.1, top=0.9)
    with pytest.raises(ValueError):
        sp.update(left=left, right=right, bottom=bottom, top=top)
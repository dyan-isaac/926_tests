import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave, AxesImage
from matplotlib import image as mimage
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO

from matplotlib.image import _ImageBase

# ---------------------- Normalization Tests ----------------------

@pytest.mark.parametrize("arr,dtype", [
    (np.array([[1, 2], [3, 4]], dtype=np.uint8), np.uint8),
    (np.array([[0.0, 1.0]], dtype=np.float32), np.float32),
    (np.array([[[1, 1, 1]]], dtype=np.uint16), np.uint16),
])
def test_normalize_valid_inputs(arr, dtype):
    out, is_scaled, alpha = _ImageBase._normalize_image_array(arr, None, norm=False)
    assert out.dtype == np.float32
    assert out.shape[:2] == arr.shape[:2]


def test_normalize_rgb_invalid_depth():
    bad = np.ones((10, 10, 5))
    with pytest.raises(TypeError):
        ImageBase._normalize_image_array(bad, None, norm=False)


def test_normalize_with_alpha():
    data = np.ones((10, 10))
    alpha = np.linspace(0, 1, 10).reshape((10, 1))
    out, _, a = _ImageBase._normalize_image_array(data, alpha, norm=False)
    assert a.shape == (10, 10)


def test_normalize_warns_on_out_of_bounds():
    arr = np.array([[2.0, -1.0]])
    with pytest.warns(UserWarning):
        _ImageBase._normalize_image_array(arr, None, norm=True)


# ---------------------- Interpolation Tests ----------------------

@pytest.mark.parametrize("interp", list(mimage._interpd_.keys()))
def test_set_valid_interpolation(interp):
    fig, ax = plt.subplots()
    im = ax.imshow(np.random.rand(5, 5))
    im.set_interpolation(interp)
    assert im.get_interpolation() == interp


def test_set_invalid_interpolation():
    fig, ax = plt.subplots()
    im = ax.imshow(np.random.rand(5, 5))
    with pytest.raises(ValueError):
        im.set_interpolation("invalid")


# ---------------------- Resampling Thresholds ----------------------

def test_resample_max_columns():
    arr = np.ones((10, 2**23 + 1))
    fig, ax = plt.subplots()
    im = ax.imshow(arr)
    assert im.get_array().shape[1] > 2**23


def test_resample_max_rows():
    arr = np.ones((2**24 + 1, 10))
    fig, ax = plt.subplots()
    im = ax.imshow(arr)
    assert im.get_array().shape[0] > 2**24


# ---------------------- get_cursor_data() ----------------------

def test_get_cursor_data_clip():
    fig, ax = plt.subplots()
    data = np.arange(100).reshape((10, 10))
    im = ax.imshow(data)
    assert im.get_cursor_data((1000, 1000)) is None


def test_get_cursor_data_inside():
    fig, ax = plt.subplots()
    data = np.arange(100).reshape((10, 10))
    im = ax.imshow(data)
    assert im.get_cursor_data((5, 5)) is not None


# ---------------------- Boolean & Float16 Support ----------------------

@pytest.mark.parametrize("dtype", [np.bool_, np.float16, np.float128])
def test_imshow_unusual_dtypes(dtype):
    fig, ax = plt.subplots()
    data = np.ones((10, 10), dtype=dtype)
    ax.imshow(data)
    assert ax.images


def test_imshow_bool_cast():
    fig, ax = plt.subplots()
    data = np.array([[True, False], [False, True]])
    im = ax.imshow(data)
    assert isinstance(im, AxesImage)


# ---------------------- imsave and imread ----------------------

def test_imsave_and_read_roundtrip():
    data = np.random.rand(10, 10, 3)
    buf = BytesIO()
    imsave(buf, data, format='png')
    buf.seek(0)
    out = imread(buf)
    assert out.shape[:2] == data.shape[:2]


def test_imsave_fails_on_invalid_format():
    data = np.random.rand(5, 5)
    buf = BytesIO()
    with pytest.raises(ValueError):
        imsave(buf, data, format='badformat')


def test_thumbnail_output_type():
    data = np.random.rand(10, 10)
    fig = mimage.thumbnail(data, thumbnail_size=(5, 5))
    assert hasattr(fig, 'savefig')  # Should be a Figure


def test_thumbnail_invalid_input():
    with pytest.raises(TypeError):
        mimage.thumbnail("not-an-array")

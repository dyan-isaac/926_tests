import pytest
import numpy as np
from matplotlib.backend_bases import (
    GraphicsContextBase, TimerBase, RendererBase, FigureCanvasBase
)
from matplotlib.figure import Figure
from matplotlib.transforms import Affine2D

class TestGraphicsContextBaseUncovered:
    def test_alpha(self):
        gc = GraphicsContextBase()
        gc.set_alpha(0.5)
        assert gc.get_alpha() == 0.5
        assert gc.get_forced_alpha()

    def test_snap(self):
        gc = GraphicsContextBase()
        for snap in [True, False, None]:
            gc.set_snap(snap)
            assert gc.get_snap() == snap

    def test_invalid_dashes(self):
        gc = GraphicsContextBase()
        with pytest.raises(ValueError):
            gc.set_dashes(0, [-1])
        with pytest.raises(ValueError):
            gc.set_dashes(0, [0, 0])

    def test_sketch(self):
        gc = GraphicsContextBase()
        gc.set_sketch_params(scale=1.5, length=10, randomness=3)
        assert isinstance(gc.get_sketch_params(), tuple)

    def test_hatch(self):
        gc = GraphicsContextBase()
        gc.set_hatch("///")
        assert gc.get_hatch_path() is not None

    # def test_clip_warn(self, caplog):
    #     gc = GraphicsContextBase()
    #     class Clip:
    #         def get_transformed_path_and_affine(self):
    #             class Dummy:
    #                 vertices = np.array([[1, 2], [np.inf, 3]])
    #             return Dummy(), Affine2D()
    #     gc.set_clip_path(Clip())
    #     with caplog.at_level("WARNING"):
    #         path, trans = gc.get_clip_path()
    #         assert path is None

class TestTimerBaseUncovered:
    def test_interval_coercion(self):
        t = TimerBase(interval=0.3)
        assert isinstance(t.interval, int) and t.interval >= 1

    def test_callback_stack(self):
        t = TimerBase()
        cb = lambda: None
        t.add_callback(cb)
        assert t.callbacks
        t.remove_callback(cb)
        assert not t.callbacks

class TestRendererBaseUncovered:
    def test_pixels_passthrough(self):
        r = RendererBase()
        assert r.points_to_pixels(72) == 72

    def test_magnification_default(self):
        r = RendererBase()
        assert r.get_image_magnification() == 1.0

    def test_flipy(self):
        r = RendererBase()
        assert r.flipy() is True

    # def test_text_as_path(self):
    #     class DummyGC:
    #         def get_rgb(self): return (1, 0, 0, 1)
    #         def set_linewidth(self, lw): pass
    #     class DummyFont:
    #         def get_size_in_points(self): return 12
    #     r = RendererBase()
    #     r._draw_text_as_path(DummyGC(), 1, 2, "hi", DummyFont(), 0, ismath="TeX")

    # def test_text_dimensions(self):
    #     class DummyFont:
    #         def get_size_in_points(self): return 12
    #     r = RendererBase()
    #     w, h, d = r.get_text_width_height_descent("x", DummyFont(), ismath=False)
    #     assert all(isinstance(val, float) for val in [w, h, d])

    def test_draw_context(self):
        r = RendererBase()
        with r._draw_disabled():
            assert callable(getattr(r, 'draw_path', None))

class TestFigureCanvasBaseUncovered:
    def test_device_ratio(self):
        fig = Figure()
        canvas = FigureCanvasBase(fig)
        assert canvas.device_pixel_ratio == 1
        assert canvas._set_device_pixel_ratio(2)
        assert canvas.device_pixel_ratio == 2

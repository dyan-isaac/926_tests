import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import weakref

import numpy as np
import pytest

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from matplotlib.testing.decorators import check_figures_equal


class NullMovieWriter(animation.AbstractMovieWriter):
    """
    A minimal MovieWriter.  It doesn't actually write anything.
    It just saves the arguments that were given to the setup() and
    grab_frame() methods as attributes, and counts how many times
    grab_frame() is called.

    This class doesn't have an __init__ method with the appropriate
    signature, and it doesn't define an isAvailable() method, so
    it cannot be added to the 'writers' registry.
    """

    def setup(self, fig, outfile, dpi, *args):
        self.fig = fig
        self.outfile = outfile
        self.dpi = dpi
        self.args = args
        self._count = 0

    def grab_frame(self, **savefig_kwargs):
        from matplotlib.animation import _validate_grabframe_kwargs
        _validate_grabframe_kwargs(savefig_kwargs)
        self.savefig_kwargs = savefig_kwargs
        self._count += 1

    def finish(self):
        pass


# Suite 1: NullMovieWriter Tests
def test_null_movie_writer_zero_frames():
    fig, _ = plt.subplots()
    writer = NullMovieWriter()
    anim = animation.FuncAnimation(fig, lambda i: [], frames=0)
    anim.save("unused.null", writer=writer)
    assert writer._count == 0

# Suite 2: FuncAnimation Parameter Variants
def test_funcanimation_with_partial_func():
    from functools import partial

    fig, ax = plt.subplots()
    line, = ax.plot([])

    def animate(frame, factor):
        line.set_ydata(np.sin(np.linspace(0, 2 * np.pi, 100) + frame * factor))
        return line,

    anim = animation.FuncAnimation(
        fig, partial(animate, factor=0.2), frames=3)
    assert isinstance(anim, animation.FuncAnimation)

# def test_funcanimation_with_non_iterable_frames():
#     fig, ax = plt.subplots()
#     anim = animation.FuncAnimation(fig, lambda i: [], frames=3)
#     assert list(anim.new_frame_seq()) == list(range(3))


# Suite 3: Writer Behavior Edge Cases
def test_writer_grab_frame_with_kwargs_fails():
    fig, _ = plt.subplots()
    writer = NullMovieWriter()
    writer.setup(fig, "file", dpi=100)
    with pytest.raises(TypeError):
        writer.grab_frame(dpi=100)

def test_writer_finish_does_not_crash():
    writer = NullMovieWriter()
    fig, _ = plt.subplots()
    writer.setup(fig, "unused.gif", dpi=100)
    writer.finish()  # No error expected


# Suite 4: HTMLWriter Specific Tests
def test_html_writer_embed_limit_zero():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    anim = animation.FuncAnimation(fig, lambda i: [], frames=3)
    writer = animation.HTMLWriter(embed_frames=True)
    writer.setup(fig, 'unused.html', dpi=100)
    writer._bytes_limit = 1  # Intentionally low
    anim._draw_next_frame(0, blit=False)
    writer.grab_frame()
    assert writer._hit_limit is True


# Suite 5: Frame Format Handling
@pytest.mark.parametrize("fmt", ["jpeg", "tiff", "png"])
def test_supported_formats_in_file_writer(fmt):
    fig, _ = plt.subplots()
    writer = animation.FFMpegFileWriter()
    writer.frame_format = fmt
    assert writer.frame_format == fmt


# Suite 6: Save Count Behavior
def test_save_count_with_generator_and_cache_false():
    fig, _ = plt.subplots()
    anim = animation.FuncAnimation(fig, lambda i: [], frames=(i for i in range(10)), cache_frame_data=False)
    assert anim._cache_frame_data is False


# Suite 7: Frame Data Mutation
def test_mutating_frame_data_has_no_side_effect():
    class FrameData(list):
        def __setitem__(self, key, value):
            raise RuntimeError("FrameData should not be mutated")

    fig, _ = plt.subplots()
    frames = [FrameData([i]) for i in range(3)]
    anim = animation.FuncAnimation(fig, lambda i: [], frames=frames)
    next(anim.new_frame_seq())


# Suite 8: JSHTML Mode Tests
def test_to_jshtml_returns_script_block():
    fig, _ = plt.subplots()
    anim = animation.FuncAnimation(fig, lambda i: [], frames=2)
    html = anim.to_jshtml(embed_frames=True)
    assert '<script ' in html


# Suite 10: File Naming and Output Path Logic
def test_temp_prefix_override_in_file_writer(tmp_path):
    fig, _ = plt.subplots()
    writer = animation.FFMpegFileWriter()
    custom_prefix = tmp_path / "custom"
    writer.setup(fig, "video.mp4", dpi=100, frame_prefix=custom_prefix)
    assert str(writer.temp_prefix).startswith(str(custom_prefix))
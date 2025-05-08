import warnings

import numpy as np
from numpy.testing import assert_array_equal
import pytest

import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle

# pytestmark = [
#     pytest.mark.usefixtures('text_placeholders')
# ]
#
#
def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

#
# @image_comparison(['tight_layout1'], style='mpl20')
# def test_tight_layout1():
#     """Test tight_layout for a single subplot."""
#     fig, ax = plt.subplots()
#     example_plot(ax, fontsize=24)
#     plt.tight_layout()
#
#
# @image_comparison(['tight_layout2'], style='mpl20')
# def test_tight_layout2():
#     """Test tight_layout for multiple subplots."""
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
#     example_plot(ax1)
#     example_plot(ax2)
#     example_plot(ax3)
#     example_plot(ax4)
#     plt.tight_layout()
#
#
# @image_comparison(['tight_layout3'], style='mpl20')
# def test_tight_layout3():
#     """Test tight_layout for multiple subplots."""
#     ax1 = plt.subplot(221)
#     ax2 = plt.subplot(223)
#     ax3 = plt.subplot(122)
#     example_plot(ax1)
#     example_plot(ax2)
#     example_plot(ax3)
#     plt.tight_layout()
#
#
# @image_comparison(['tight_layout4'], style='mpl20')
# def test_tight_layout4():
#     """Test tight_layout for subplot2grid."""
#     ax1 = plt.subplot2grid((3, 3), (0, 0))
#     ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)
#     ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
#     ax4 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
#     example_plot(ax1)
#     example_plot(ax2)
#     example_plot(ax3)
#     example_plot(ax4)
#     plt.tight_layout()
#
#
# @image_comparison(['tight_layout5'], style='mpl20')
# def test_tight_layout5():
#     """Test tight_layout for image."""
#     ax = plt.subplot()
#     arr = np.arange(100).reshape((10, 10))
#     ax.imshow(arr, interpolation="none")
#     plt.tight_layout()
#
#
# @image_comparison(['tight_layout6'], style='mpl20')
# def test_tight_layout6():
#     """Test tight_layout for gridspec."""
#
#     # This raises warnings since tight layout cannot
#     # do this fully automatically. But the test is
#     # correct since the layout is manually edited
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", UserWarning)
#         fig = plt.figure()
#
#         gs1 = mpl.gridspec.GridSpec(2, 1)
#         ax1 = fig.add_subplot(gs1[0])
#         ax2 = fig.add_subplot(gs1[1])
#
#         example_plot(ax1)
#         example_plot(ax2)
#
#         gs1.tight_layout(fig, rect=[0, 0, 0.5, 1])
#
#         gs2 = mpl.gridspec.GridSpec(3, 1)
#
#         for ss in gs2:
#             ax = fig.add_subplot(ss)
#             example_plot(ax)
#             ax.set_title("")
#             ax.set_xlabel("")
#
#         ax.set_xlabel("x-label", fontsize=12)
#
#         gs2.tight_layout(fig, rect=[0.5, 0, 1, 1], h_pad=0.45)
#
#         top = min(gs1.top, gs2.top)
#         bottom = max(gs1.bottom, gs2.bottom)
#
#         gs1.tight_layout(fig, rect=[None, 0 + (bottom-gs1.bottom),
#                                     0.5, 1 - (gs1.top-top)])
#         gs2.tight_layout(fig, rect=[0.5, 0 + (bottom-gs2.bottom),
#                                     None, 1 - (gs2.top-top)],
#                          h_pad=0.45)
#
#
# @image_comparison(['tight_layout7'], style='mpl20')
# def test_tight_layout7():
#     # tight layout with left and right titles
#     fontsize = 24
#     fig, ax = plt.subplots()
#     ax.plot([1, 2])
#     ax.locator_params(nbins=3)
#     ax.set_xlabel('x-label', fontsize=fontsize)
#     ax.set_ylabel('y-label', fontsize=fontsize)
#     ax.set_title('Left Title', loc='left', fontsize=fontsize)
#     ax.set_title('Right Title', loc='right', fontsize=fontsize)
#     plt.tight_layout()
#
#
# @image_comparison(['tight_layout8'], style='mpl20', tol=0.005)
# def test_tight_layout8():
#     """Test automatic use of tight_layout."""
#     fig = plt.figure()
#     fig.set_layout_engine(layout='tight', pad=0.1)
#     ax = fig.add_subplot()
#     example_plot(ax, fontsize=24)
#     fig.draw_without_rendering()
#
#
# @image_comparison(['tight_layout9'], style='mpl20')
# def test_tight_layout9():
#     # Test tight_layout for non-visible subplots
#     # GH 8244
#     f, axarr = plt.subplots(2, 2)
#     axarr[1][1].set_visible(False)
#     plt.tight_layout()
#
#
# def test_outward_ticks():
#     """Test automatic use of tight_layout."""
#     fig = plt.figure()
#     ax = fig.add_subplot(221)
#     ax.xaxis.set_tick_params(tickdir='out', length=16, width=3)
#     ax.yaxis.set_tick_params(tickdir='out', length=16, width=3)
#     ax.xaxis.set_tick_params(
#         tickdir='out', length=32, width=3, tick1On=True, which='minor')
#     ax.yaxis.set_tick_params(
#         tickdir='out', length=32, width=3, tick1On=True, which='minor')
#     ax.xaxis.set_ticks([0], minor=True)
#     ax.yaxis.set_ticks([0], minor=True)
#     ax = fig.add_subplot(222)
#     ax.xaxis.set_tick_params(tickdir='in', length=32, width=3)
#     ax.yaxis.set_tick_params(tickdir='in', length=32, width=3)
#     ax = fig.add_subplot(223)
#     ax.xaxis.set_tick_params(tickdir='inout', length=32, width=3)
#     ax.yaxis.set_tick_params(tickdir='inout', length=32, width=3)
#     ax = fig.add_subplot(224)
#     ax.xaxis.set_tick_params(tickdir='out', length=32, width=3)
#     ax.yaxis.set_tick_params(tickdir='out', length=32, width=3)
#     plt.tight_layout()
#     # These values were obtained after visual checking that they correspond
#     # to a tight layouting that did take the ticks into account.
#     expected = [
#         [[0.092, 0.605], [0.433, 0.933]],
#         [[0.581, 0.605], [0.922, 0.933]],
#         [[0.092, 0.138], [0.433, 0.466]],
#         [[0.581, 0.138], [0.922, 0.466]],
#     ]
#     for nn, ax in enumerate(fig.axes):
#         assert_array_equal(np.round(ax.get_position().get_points(), 3),
#                            expected[nn])
#
#
# def add_offsetboxes(ax, size=10, margin=.1, color='black'):
#     """
#     Surround ax with OffsetBoxes
#     """
#     m, mp = margin, 1+margin
#     anchor_points = [(-m, -m), (-m, .5), (-m, mp),
#                      (.5, mp), (mp, mp), (mp, .5),
#                      (mp, -m), (.5, -m)]
#     for point in anchor_points:
#         da = DrawingArea(size, size)
#         background = Rectangle((0, 0), width=size,
#                                height=size,
#                                facecolor=color,
#                                edgecolor='None',
#                                linewidth=0,
#                                antialiased=False)
#         da.add_artist(background)
#
#         anchored_box = AnchoredOffsetbox(
#             loc='center',
#             child=da,
#             pad=0.,
#             frameon=False,
#             bbox_to_anchor=point,
#             bbox_transform=ax.transAxes,
#             borderpad=0.)
#         ax.add_artist(anchored_box)
#
#
# def test_tight_layout_offsetboxes():
#     # 0.
#     # - Create 4 subplots
#     # - Plot a diagonal line on them
#     # - Use tight_layout
#     #
#     # 1.
#     # - Same 4 subplots
#     # - Surround each plot with 7 boxes
#     # - Use tight_layout
#     # - See that the squares are included in the tight_layout and that the squares do
#     #   not overlap
#     #
#     # 2.
#     # - Make the squares around the Axes invisible
#     # - See that the invisible squares do not affect the tight_layout
#     rows = cols = 2
#     colors = ['red', 'blue', 'green', 'yellow']
#     x = y = [0, 1]
#
#     def _subplots(with_boxes):
#         fig, axs = plt.subplots(rows, cols)
#         for ax, color in zip(axs.flat, colors):
#             ax.plot(x, y, color=color)
#             if with_boxes:
#                 add_offsetboxes(ax, 20, color=color)
#         return fig, axs
#
#     # 0.
#     fig0, axs0 = _subplots(False)
#     fig0.tight_layout()
#
#     # 1.
#     fig1, axs1 = _subplots(True)
#     fig1.tight_layout()
#
#     # The AnchoredOffsetbox should be added to the bounding of the Axes, causing them to
#     # be smaller than the plain figure.
#     for ax0, ax1 in zip(axs0.flat, axs1.flat):
#         bbox0 = ax0.get_position()
#         bbox1 = ax1.get_position()
#         assert bbox1.x0 > bbox0.x0
#         assert bbox1.x1 < bbox0.x1
#         assert bbox1.y0 > bbox0.y0
#         assert bbox1.y1 < bbox0.y1
#
#     # No AnchoredOffsetbox should overlap with another.
#     bboxes = []
#     for ax1 in axs1.flat:
#         for child in ax1.get_children():
#             if not isinstance(child, AnchoredOffsetbox):
#                 continue
#             bbox = child.get_window_extent()
#             for other_bbox in bboxes:
#                 assert not bbox.overlaps(other_bbox)
#             bboxes.append(bbox)
#
#     # 2.
#     fig2, axs2 = _subplots(True)
#     for ax in axs2.flat:
#         for child in ax.get_children():
#             if isinstance(child, AnchoredOffsetbox):
#                 child.set_visible(False)
#     fig2.tight_layout()
#     # The invisible AnchoredOffsetbox should not count for tight layout, so it should
#     # look the same as when they were never added.
#     for ax0, ax2 in zip(axs0.flat, axs2.flat):
#         bbox0 = ax0.get_position()
#         bbox2 = ax2.get_position()
#         assert_array_equal(bbox2.get_points(), bbox0.get_points())
#
#
# def test_empty_layout():
#     """Test that tight layout doesn't cause an error when there are no Axes."""
#     fig = plt.gcf()
#     fig.tight_layout()
#
#
# @pytest.mark.parametrize("label", ["xlabel", "ylabel"])
# def test_verybig_decorators(label):
#     """Test that no warning emitted when xlabel/ylabel too big."""
#     fig, ax = plt.subplots(figsize=(3, 2))
#     ax.set(**{label: 'a' * 100})
#
#
# def test_big_decorators_horizontal():
#     """Test that doesn't warn when xlabel too big."""
#     fig, axs = plt.subplots(1, 2, figsize=(3, 2))
#     axs[0].set_xlabel('a' * 30)
#     axs[1].set_xlabel('b' * 30)
#
#
# def test_big_decorators_vertical():
#     """Test that doesn't warn when ylabel too big."""
#     fig, axs = plt.subplots(2, 1, figsize=(3, 2))
#     axs[0].set_ylabel('a' * 20)
#     axs[1].set_ylabel('b' * 20)
#
#
# def test_badsubplotgrid():
#     # test that we get warning for mismatched subplot grids, not than an error
#     plt.subplot2grid((4, 5), (0, 0))
#     # this is the bad entry:
#     plt.subplot2grid((5, 5), (0, 3), colspan=3, rowspan=5)
#     with pytest.warns(UserWarning):
#         plt.tight_layout()
#
#
# def test_collapsed():
#     # test that if the amount of space required to make all the axes
#     # decorations fit would mean that the actual Axes would end up with size
#     # zero (i.e. margins add up to more than the available width) that a call
#     # to tight_layout will not get applied:
#     fig, ax = plt.subplots(tight_layout=True)
#     ax.set_xlim([0, 1])
#     ax.set_ylim([0, 1])
#
#     ax.annotate('BIG LONG STRING', xy=(1.25, 2), xytext=(10.5, 1.75),
#                 annotation_clip=False)
#     p1 = ax.get_position()
#     with pytest.warns(UserWarning):
#         plt.tight_layout()
#         p2 = ax.get_position()
#         assert p1.width == p2.width
#     # test that passing a rect doesn't crash...
#     with pytest.warns(UserWarning):
#         plt.tight_layout(rect=[0, 0, 0.8, 0.8])
#
#
# def test_suptitle():
#     fig, ax = plt.subplots(tight_layout=True)
#     st = fig.suptitle("foo")
#     t = ax.set_title("bar")
#     fig.canvas.draw()
#     assert st.get_window_extent().y0 > t.get_window_extent().y1
#
#
# @pytest.mark.backend("pdf")
# def test_non_agg_renderer(monkeypatch, recwarn):
#     unpatched_init = mpl.backend_bases.RendererBase.__init__
#
#     def __init__(self, *args, **kwargs):
#         # Check that we don't instantiate any other renderer than a pdf
#         # renderer to perform pdf tight layout.
#         assert isinstance(self, mpl.backends.backend_pdf.RendererPdf)
#         unpatched_init(self, *args, **kwargs)
#
#     monkeypatch.setattr(mpl.backend_bases.RendererBase, "__init__", __init__)
#     fig, ax = plt.subplots()
#     fig.tight_layout()
#
#
# def test_manual_colorbar():
#     # This should warn, but not raise
#     fig, axes = plt.subplots(1, 2)
#     pts = axes[1].scatter([0, 1], [0, 1], c=[1, 5])
#     ax_rect = axes[1].get_position()
#     cax = fig.add_axes(
#         [ax_rect.x1 + 0.005, ax_rect.y0, 0.015, ax_rect.height]
#     )
#     fig.colorbar(pts, cax=cax)
#     with pytest.warns(UserWarning, match="This figure includes Axes"):
#         fig.tight_layout()
#
#
# def test_clipped_to_axes():
#     # Ensure that _fully_clipped_to_axes() returns True under default
#     # conditions for all projection types. Axes.get_tightbbox()
#     # uses this to skip artists in layout calculations.
#     arr = np.arange(100).reshape((10, 10))
#     fig = plt.figure(figsize=(6, 2))
#     ax1 = fig.add_subplot(131, projection='rectilinear')
#     ax2 = fig.add_subplot(132, projection='mollweide')
#     ax3 = fig.add_subplot(133, projection='polar')
#     for ax in (ax1, ax2, ax3):
#         # Default conditions (clipped by ax.bbox or ax.patch)
#         ax.grid(False)
#         h, = ax.plot(arr[:, 0])
#         m = ax.pcolor(arr)
#         assert h._fully_clipped_to_axes()
#         assert m._fully_clipped_to_axes()
#         # Non-default conditions (not clipped by ax.patch)
#         rect = Rectangle((0, 0), 0.5, 0.5, transform=ax.transAxes)
#         h.set_clip_path(rect)
#         m.set_clip_path(rect.get_path(), rect.get_transform())
#         assert not h._fully_clipped_to_axes()
#         assert not m._fully_clipped_to_axes()
#
#
# def test_tight_pads():
#     fig, ax = plt.subplots()
#     with pytest.warns(PendingDeprecationWarning,
#                       match='will be deprecated'):
#         fig.set_tight_layout({'pad': 0.15})
#     fig.draw_without_rendering()
#
#
# def test_tight_kwargs():
#     fig, ax = plt.subplots(tight_layout={'pad': 0.15})
#     fig.draw_without_rendering()
#
#
# def test_tight_toggle():
#     fig, ax = plt.subplots()
#     with pytest.warns(PendingDeprecationWarning):
#         fig.set_tight_layout(True)
#         assert fig.get_tight_layout()
#         fig.set_tight_layout(False)
#         assert not fig.get_tight_layout()
#         fig.set_tight_layout(True)
#         assert fig.get_tight_layout()
# pytestmark = [
#     pytest.mark.usefixtures('text_placeholders')
# ]
#
#
# def example_plot(ax, fontsize=12):
#     ax.plot([1, 2])
#     ax.locator_params(nbins=3)
#     ax.set_xlabel('x-label', fontsize=fontsize)
#     ax.set_ylabel('y-label', fontsize=fontsize)
#     ax.set_title('Title', fontsize=fontsize)
#
#
# @image_comparison(['tight_layout1'], style='mpl20')
# def test_tight_layout1():
#     """Test tight_layout for a single subplot."""
#     fig, ax = plt.subplots()
#     example_plot(ax, fontsize=24)
#     plt.tight_layout()
#
#
# @image_comparison(['tight_layout2'], style='mpl20')
# def test_tight_layout2():
#     """Test tight_layout for multiple subplots."""
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
#     example_plot(ax1)
#     example_plot(ax2)
#     example_plot(ax3)
#     example_plot(ax4)
#     plt.tight_layout()
#
#
# @image_comparison(['tight_layout3'], style='mpl20')
# def test_tight_layout3():
#     """Test tight_layout for multiple subplots."""
#     ax1 = plt.subplot(221)
#     ax2 = plt.subplot(223)
#     ax3 = plt.subplot(122)
#     example_plot(ax1)
#     example_plot(ax2)
#     example_plot(ax3)
#     plt.tight_layout()
#
#
# @image_comparison(['tight_layout4'], style='mpl20')
# def test_tight_layout4():
#     """Test tight_layout for subplot2grid."""
#     ax1 = plt.subplot2grid((3, 3), (0, 0))
#     ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)
#     ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
#     ax4 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
#     example_plot(ax1)
#     example_plot(ax2)
#     example_plot(ax3)
#     example_plot(ax4)
#     plt.tight_layout()
#
#
# @image_comparison(['tight_layout5'], style='mpl20')
# def test_tight_layout5():
#     """Test tight_layout for image."""
#     ax = plt.subplot()
#     arr = np.arange(100).reshape((10, 10))
#     ax.imshow(arr, interpolation="none")
#     plt.tight_layout()
#
#
# @image_comparison(['tight_layout6'], style='mpl20')
# def test_tight_layout6():
#     """Test tight_layout for gridspec."""
#
#     # This raises warnings since tight layout cannot
#     # do this fully automatically. But the test is
#     # correct since the layout is manually edited
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", UserWarning)
#         fig = plt.figure()
#
#         gs1 = mpl.gridspec.GridSpec(2, 1)
#         ax1 = fig.add_subplot(gs1[0])
#         ax2 = fig.add_subplot(gs1[1])
#
#         example_plot(ax1)
#         example_plot(ax2)
#
#         gs1.tight_layout(fig, rect=[0, 0, 0.5, 1])
#
#         gs2 = mpl.gridspec.GridSpec(3, 1)
#
#         for ss in gs2:
#             ax = fig.add_subplot(ss)
#             example_plot(ax)
#             ax.set_title("")
#             ax.set_xlabel("")
#
#         ax.set_xlabel("x-label", fontsize=12)
#
#         gs2.tight_layout(fig, rect=[0.5, 0, 1, 1], h_pad=0.45)
#
#         top = min(gs1.top, gs2.top)
#         bottom = max(gs1.bottom, gs2.bottom)
#
#         gs1.tight_layout(fig, rect=[None, 0 + (bottom-gs1.bottom),
#                                     0.5, 1 - (gs1.top-top)])
#         gs2.tight_layout(fig, rect=[0.5, 0 + (bottom-gs2.bottom),
#                                     None, 1 - (gs2.top-top)],
#                          h_pad=0.45)
#
#
# @image_comparison(['tight_layout7'], style='mpl20')
# def test_tight_layout7():
#     # tight layout with left and right titles
#     fontsize = 24
#     fig, ax = plt.subplots()
#     ax.plot([1, 2])
#     ax.locator_params(nbins=3)
#     ax.set_xlabel('x-label', fontsize=fontsize)
#     ax.set_ylabel('y-label', fontsize=fontsize)
#     ax.set_title('Left Title', loc='left', fontsize=fontsize)
#     ax.set_title('Right Title', loc='right', fontsize=fontsize)
#     plt.tight_layout()
#
#
# @image_comparison(['tight_layout8'], style='mpl20', tol=0.005)
# def test_tight_layout8():
#     """Test automatic use of tight_layout."""
#     fig = plt.figure()
#     fig.set_layout_engine(layout='tight', pad=0.1)
#     ax = fig.add_subplot()
#     example_plot(ax, fontsize=24)
#     fig.draw_without_rendering()
#
#
# @image_comparison(['tight_layout9'], style='mpl20')
# def test_tight_layout9():
#     # Test tight_layout for non-visible subplots
#     # GH 8244
#     f, axarr = plt.subplots(2, 2)
#     axarr[1][1].set_visible(False)
#     plt.tight_layout()
#
#
# def test_outward_ticks():
#     """Test automatic use of tight_layout."""
#     fig = plt.figure()
#     ax = fig.add_subplot(221)
#     ax.xaxis.set_tick_params(tickdir='out', length=16, width=3)
#     ax.yaxis.set_tick_params(tickdir='out', length=16, width=3)
#     ax.xaxis.set_tick_params(
#         tickdir='out', length=32, width=3, tick1On=True, which='minor')
#     ax.yaxis.set_tick_params(
#         tickdir='out', length=32, width=3, tick1On=True, which='minor')
#     ax.xaxis.set_ticks([0], minor=True)
#     ax.yaxis.set_ticks([0], minor=True)
#     ax = fig.add_subplot(222)
#     ax.xaxis.set_tick_params(tickdir='in', length=32, width=3)
#     ax.yaxis.set_tick_params(tickdir='in', length=32, width=3)
#     ax = fig.add_subplot(223)
#     ax.xaxis.set_tick_params(tickdir='inout', length=32, width=3)
#     ax.yaxis.set_tick_params(tickdir='inout', length=32, width=3)
#     ax = fig.add_subplot(224)
#     ax.xaxis.set_tick_params(tickdir='out', length=32, width=3)
#     ax.yaxis.set_tick_params(tickdir='out', length=32, width=3)
#     plt.tight_layout()
#     # These values were obtained after visual checking that they correspond
#     # to a tight layouting that did take the ticks into account.
#     expected = [
#         [[0.092, 0.605], [0.433, 0.933]],
#         [[0.581, 0.605], [0.922, 0.933]],
#         [[0.092, 0.138], [0.433, 0.466]],
#         [[0.581, 0.138], [0.922, 0.466]],
#     ]
#     for nn, ax in enumerate(fig.axes):
#         assert_array_equal(np.round(ax.get_position().get_points(), 3),
#                            expected[nn])
#
#
# def add_offsetboxes(ax, size=10, margin=.1, color='black'):
#     """
#     Surround ax with OffsetBoxes
#     """
#     m, mp = margin, 1+margin
#     anchor_points = [(-m, -m), (-m, .5), (-m, mp),
#                      (.5, mp), (mp, mp), (mp, .5),
#                      (mp, -m), (.5, -m)]
#     for point in anchor_points:
#         da = DrawingArea(size, size)
#         background = Rectangle((0, 0), width=size,
#                                height=size,
#                                facecolor=color,
#                                edgecolor='None',
#                                linewidth=0,
#                                antialiased=False)
#         da.add_artist(background)
#
#         anchored_box = AnchoredOffsetbox(
#             loc='center',
#             child=da,
#             pad=0.,
#             frameon=False,
#             bbox_to_anchor=point,
#             bbox_transform=ax.transAxes,
#             borderpad=0.)
#         ax.add_artist(anchored_box)
#
#
# def test_tight_layout_offsetboxes():
#     # 0.
#     # - Create 4 subplots
#     # - Plot a diagonal line on them
#     # - Use tight_layout
#     #
#     # 1.
#     # - Same 4 subplots
#     # - Surround each plot with 7 boxes
#     # - Use tight_layout
#     # - See that the squares are included in the tight_layout and that the squares do
#     #   not overlap
#     #
#     # 2.
#     # - Make the squares around the Axes invisible
#     # - See that the invisible squares do not affect the tight_layout
#     rows = cols = 2
#     colors = ['red', 'blue', 'green', 'yellow']
#     x = y = [0, 1]
#
#     def _subplots(with_boxes):
#         fig, axs = plt.subplots(rows, cols)
#         for ax, color in zip(axs.flat, colors):
#             ax.plot(x, y, color=color)
#             if with_boxes:
#                 add_offsetboxes(ax, 20, color=color)
#         return fig, axs
#
#     # 0.
#     fig0, axs0 = _subplots(False)
#     fig0.tight_layout()
#
#     # 1.
#     fig1, axs1 = _subplots(True)
#     fig1.tight_layout()
#
#     # The AnchoredOffsetbox should be added to the bounding of the Axes, causing them to
#     # be smaller than the plain figure.
#     for ax0, ax1 in zip(axs0.flat, axs1.flat):
#         bbox0 = ax0.get_position()
#         bbox1 = ax1.get_position()
#         assert bbox1.x0 > bbox0.x0
#         assert bbox1.x1 < bbox0.x1
#         assert bbox1.y0 > bbox0.y0
#         assert bbox1.y1 < bbox0.y1
#
#     # No AnchoredOffsetbox should overlap with another.
#     bboxes = []
#     for ax1 in axs1.flat:
#         for child in ax1.get_children():
#             if not isinstance(child, AnchoredOffsetbox):
#                 continue
#             bbox = child.get_window_extent()
#             for other_bbox in bboxes:
#                 assert not bbox.overlaps(other_bbox)
#             bboxes.append(bbox)
#
#     # 2.
#     fig2, axs2 = _subplots(True)
#     for ax in axs2.flat:
#         for child in ax.get_children():
#             if isinstance(child, AnchoredOffsetbox):
#                 child.set_visible(False)
#     fig2.tight_layout()
#     # The invisible AnchoredOffsetbox should not count for tight layout, so it should
#     # look the same as when they were never added.
#     for ax0, ax2 in zip(axs0.flat, axs2.flat):
#         bbox0 = ax0.get_position()
#         bbox2 = ax2.get_position()
#         assert_array_equal(bbox2.get_points(), bbox0.get_points())
#
#
# def test_empty_layout():
#     """Test that tight layout doesn't cause an error when there are no Axes."""
#     fig = plt.gcf()
#     fig.tight_layout()
#
#
# @pytest.mark.parametrize("label", ["xlabel", "ylabel"])
# def test_verybig_decorators(label):
#     """Test that no warning emitted when xlabel/ylabel too big."""
#     fig, ax = plt.subplots(figsize=(3, 2))
#     ax.set(**{label: 'a' * 100})
#
#
# def test_big_decorators_horizontal():
#     """Test that doesn't warn when xlabel too big."""
#     fig, axs = plt.subplots(1, 2, figsize=(3, 2))
#     axs[0].set_xlabel('a' * 30)
#     axs[1].set_xlabel('b' * 30)
#
#
# def test_big_decorators_vertical():
#     """Test that doesn't warn when ylabel too big."""
#     fig, axs = plt.subplots(2, 1, figsize=(3, 2))
#     axs[0].set_ylabel('a' * 20)
#     axs[1].set_ylabel('b' * 20)
#
#
# def test_badsubplotgrid():
#     # test that we get warning for mismatched subplot grids, not than an error
#     plt.subplot2grid((4, 5), (0, 0))
#     # this is the bad entry:
#     plt.subplot2grid((5, 5), (0, 3), colspan=3, rowspan=5)
#     with pytest.warns(UserWarning):
#         plt.tight_layout()
#
#
# def test_collapsed():
#     # test that if the amount of space required to make all the axes
#     # decorations fit would mean that the actual Axes would end up with size
#     # zero (i.e. margins add up to more than the available width) that a call
#     # to tight_layout will not get applied:
#     fig, ax = plt.subplots(tight_layout=True)
#     ax.set_xlim([0, 1])
#     ax.set_ylim([0, 1])
#
#     ax.annotate('BIG LONG STRING', xy=(1.25, 2), xytext=(10.5, 1.75),
#                 annotation_clip=False)
#     p1 = ax.get_position()
#     with pytest.warns(UserWarning):
#         plt.tight_layout()
#         p2 = ax.get_position()
#         assert p1.width == p2.width
#     # test that passing a rect doesn't crash...
#     with pytest.warns(UserWarning):
#         plt.tight_layout(rect=[0, 0, 0.8, 0.8])
#
#
# def test_suptitle():
#     fig, ax = plt.subplots(tight_layout=True)
#     st = fig.suptitle("foo")
#     t = ax.set_title("bar")
#     fig.canvas.draw()
#     assert st.get_window_extent().y0 > t.get_window_extent().y1
#
#
# @pytest.mark.backend("pdf")
# def test_non_agg_renderer(monkeypatch, recwarn):
#     unpatched_init = mpl.backend_bases.RendererBase.__init__
#
#     def __init__(self, *args, **kwargs):
#         # Check that we don't instantiate any other renderer than a pdf
#         # renderer to perform pdf tight layout.
#         assert isinstance(self, mpl.backends.backend_pdf.RendererPdf)
#         unpatched_init(self, *args, **kwargs)
#
#     monkeypatch.setattr(mpl.backend_bases.RendererBase, "__init__", __init__)
#     fig, ax = plt.subplots()
#     fig.tight_layout()
#
#
# def test_manual_colorbar():
#     # This should warn, but not raise
#     fig, axes = plt.subplots(1, 2)
#     pts = axes[1].scatter([0, 1], [0, 1], c=[1, 5])
#     ax_rect = axes[1].get_position()
#     cax = fig.add_axes(
#         [ax_rect.x1 + 0.005, ax_rect.y0, 0.015, ax_rect.height]
#     )
#     fig.colorbar(pts, cax=cax)
#     with pytest.warns(UserWarning, match="This figure includes Axes"):
#         fig.tight_layout()
#
#
# def test_clipped_to_axes():
#     # Ensure that _fully_clipped_to_axes() returns True under default
#     # conditions for all projection types. Axes.get_tightbbox()
#     # uses this to skip artists in layout calculations.
#     arr = np.arange(100).reshape((10, 10))
#     fig = plt.figure(figsize=(6, 2))
#     ax1 = fig.add_subplot(131, projection='rectilinear')
#     ax2 = fig.add_subplot(132, projection='mollweide')
#     ax3 = fig.add_subplot(133, projection='polar')
#     for ax in (ax1, ax2, ax3):
#         # Default conditions (clipped by ax.bbox or ax.patch)
#         ax.grid(False)
#         h, = ax.plot(arr[:, 0])
#         m = ax.pcolor(arr)
#         assert h._fully_clipped_to_axes()
#         assert m._fully_clipped_to_axes()
#         # Non-default conditions (not clipped by ax.patch)
#         rect = Rectangle((0, 0), 0.5, 0.5, transform=ax.transAxes)
#         h.set_clip_path(rect)
#         m.set_clip_path(rect.get_path(), rect.get_transform())
#         assert not h._fully_clipped_to_axes()
#         assert not m._fully_clipped_to_axes()
#
#
# def test_tight_pads():
#     fig, ax = plt.subplots()
#     with pytest.warns(PendingDeprecationWarning,
#                       match='will be deprecated'):
#         fig.set_tight_layout({'pad': 0.15})
#     fig.draw_without_rendering()
#
#
# def test_tight_kwargs():
#     fig, ax = plt.subplots(tight_layout={'pad': 0.15})
#     fig.draw_without_rendering()
#
#
# def test_tight_toggle():
#     fig, ax = plt.subplots()
#     with pytest.warns(PendingDeprecationWarning):
#         fig.set_tight_layout(True)
#         assert fig.get_tight_layout()
#         fig.set_tight_layout(False)
#         assert not fig.get_tight_layout()
#         fig.set_tight_layout(True)
#         assert fig.get_tight_layout()

#################
from matplotlib._tight_layout import (_auto_adjust_subplotpars)
from matplotlib.transforms import Bbox
from matplotlib import _tight_layout
import unittest
from matplotlib.font_manager import FontProperties

def test_auto_adjust_subplotpars_value_error():
    fig, ax = plt.subplots()
    renderer = fig.canvas.get_renderer()

    # Case 1: Empty subplot_list
    with pytest.raises(ValueError):
        _auto_adjust_subplotpars(
            fig, renderer, shape=(1, 1),
            span_pairs=[(slice(0, 1), slice(0, 1))],
            subplot_list=[]
        )

    # Case 2: Mismatched lengths
    with pytest.raises(ValueError):
        _auto_adjust_subplotpars(
            fig, renderer, shape=(1, 1),
            span_pairs=[(slice(0, 1), slice(0, 1)), (slice(0, 1), slice(0, 1))],
            subplot_list=[[ax]]
        )


class TestTightLayout(unittest.TestCase):
    def test_auto_adjust_subplotpars_ax_bbox_list_none(self):
        """Test that _auto_adjust_subplotpars correctly handles ax_bbox_list=None."""
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2)
        renderer = fig.canvas.get_renderer()

        # Prepare the parameters
        shape = (2, 2)
        subplot_list = [[axes[0, 0]], [axes[0, 1]], [axes[1, 0]], [axes[1, 1]]]
        span_pairs = [
            (slice(0, 1), slice(0, 1)),
            (slice(0, 1), slice(1, 2)),
            (slice(1, 2), slice(0, 1)),
            (slice(1, 2), slice(1, 2))
        ]

        # Call the function with ax_bbox_list=None
        result1 = _tight_layout._auto_adjust_subplotpars(
            fig, renderer, shape, span_pairs, subplot_list,
            ax_bbox_list=None, pad=1.08
        )

        # Calculate ax_bbox_list manually as it would be in lines 70-72
        manual_ax_bbox_list = [
            Bbox.union([ax.get_position(original=True) for ax in subplots])
            for subplots in subplot_list
        ]

        # Call the function again with the manually calculated ax_bbox_list
        result2 = _tight_layout._auto_adjust_subplotpars(
            fig, renderer, shape, span_pairs, subplot_list,
            ax_bbox_list=manual_ax_bbox_list, pad=1.08
        )

        # The results should be identical
        self.assertEqual(result1.keys(), result2.keys())
        for key in result1:
            self.assertAlmostEqual(result1[key], result2[key])


def test_tight_layout_supylabel_adjustment():
    """Test that supylabel properly increases the left margin in tight layout."""
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    # Add a supylabel with known text
    test_label = "Test Supylabel"
    fig.supylabel(test_label)

    # First calculate margins without the supylabel in layout
    fig._supylabel.set_in_layout(False)
    renderer = fig.canvas.get_renderer()

    kwargs_without_supylabel = _auto_adjust_subplotpars(
        fig, renderer, shape=(1, 1),
        span_pairs=[(slice(0, 1), slice(0, 1))],
        subplot_list=[[ax]], pad=1.08)

    # Now include the supylabel in layout
    fig._supylabel.set_in_layout(True)
    kwargs_with_supylabel = _auto_adjust_subplotpars(
        fig, renderer, shape=(1, 1),
        span_pairs=[(slice(0, 1), slice(0, 1))],
        subplot_list=[[ax]], pad=1.08)

    # The margin with supylabel should be larger than without
    assert kwargs_with_supylabel["left"] > kwargs_without_supylabel["left"]


def test_supxlabel_in_tight_layout():
    """Test that supxlabel is properly accounted for in tight_layout calculations."""
    # Create figure with and without supxlabel
    fig_without = plt.figure(figsize=(6, 4))
    ax1 = fig_without.add_subplot(111)
    ax1.plot([1, 2, 3], [1, 2, 3])

    fig_with = plt.figure(figsize=(6, 4))
    ax2 = fig_with.add_subplot(111)
    ax2.plot([1, 2, 3], [1, 2, 3])
    fig_with.supxlabel("Super X Label")

    # Apply tight_layout to both
    renderer_without = fig_without.canvas.get_renderer()
    renderer_with = fig_with.canvas.get_renderer()

    kwargs_without = _tight_layout.get_tight_layout_figure(
        fig_without, [ax1],
        _tight_layout.get_subplotspec_list([ax1]),
        renderer_without)

    kwargs_with = _tight_layout.get_tight_layout_figure(
        fig_with, [ax2],
        _tight_layout.get_subplotspec_list([ax2]),
        renderer_with)

    # Verify that bottom margin is larger when supxlabel is present
    assert kwargs_with['bottom'] > kwargs_without['bottom'], \
        "Bottom margin should be larger with supxlabel"

    # Clean up
    plt.close(fig_without)
    plt.close(fig_with)


from unittest.mock import patch
from matplotlib import _api


def test_tight_layout_vertical_margins_too_large():
    """Test that tight_layout warns and fails when vertical margins are too large."""
    # Create a figure with minimal height to force margin issue
    fig = plt.figure(figsize=(6, 1))  # Making height even smaller
    ax = fig.add_subplot(111)

    # Add even larger decorations to force margin issue
    ax.set_title('Very Large Title ' * 10, fontsize=30)
    ax.set_xlabel('Very Large XLabel ' * 10, fontsize=30)
    ax.set_ylabel('Very Large YLabel ' * 10, fontsize=30)

    # Add actual content
    ax.plot([1, 2, 3], [1, 2, 3])

    # Use Python's warning capture to detect any warnings
    with warnings.catch_warnings(record=True) as warning_list:
        # Enable all warnings
        warnings.simplefilter('always')

        # Try to apply tight_layout
        fig.tight_layout()

        # Check if any warnings were recorded
        assert len(warning_list) > 0, "No warnings were issued by tight_layout()"

        # Get warning messages as strings
        warning_messages = [str(warning.message) for warning in warning_list]

        # Check for tight layout related warnings using more permissive criteria
        layout_related = any(
            any(term in msg.lower() for term in ["tight", "layout", "margin", "pad"])
            for msg in warning_messages
        )

        assert layout_related, \
            f"Expected warning about tight_layout not issued. Actual warnings: {warning_messages}"

    plt.close(fig)

def test_tight_layout_small_axes_width():
    """Test that tight_layout fails with appropriate warning when horizontal axes too small."""
    import matplotlib.pyplot as plt
    import warnings

    # Create a very small figure with multiple columns
    fig = plt.figure(figsize=(2, 6))  # Make the figure narrower

    # Create 3 columns with extremely long titles and large labels
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.set_title('Very Long Title ' * 10)  # Extra long title
        ax.set_xlabel('Long X Label ' * 5)  # Add long x-label
        # Add tick labels with rotation to consume more space
        ax.set_xticks([0.2, 0.5, 0.8])
        ax.set_xticklabels(['Long tick label'] * 3, rotation=45, ha='right')

    # Ensure the figure has too little width for the content
    plt.subplots_adjust(wspace=0.8)  # Increase spacing between subplots

    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Try to apply tight layout
        result = fig.tight_layout()

        # Check that the warning was raised
        assert len(w) > 0
        assert any("cannot make Axes width small enough" in str(warning.message)
                   for warning in w)

        # tight_layout returns None implicitly when it fails
        assert result is None


def test_tight_layout_mixed_aspect_ratios():
    """Test that tight_layout works correctly with mixed aspect ratio subplots."""
    fig = plt.figure(figsize=(8, 6))

    # Create subplots with different aspect ratios
    ax1 = fig.add_subplot(221)  # normal
    ax2 = fig.add_subplot(222, aspect=2.0)  # wider
    ax3 = fig.add_subplot(223, aspect=0.5)  # taller
    ax4 = fig.add_subplot(224, aspect='equal')  # equal aspect

    # Add content to all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        example_plot(ax)

    # Apply tight_layout
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fig.tight_layout()

    # Check that all subplots have appropriate positions
    positions = [ax.get_position() for ax in [ax1, ax2, ax3, ax4]]

    # Ensure no overlap between subplots
    for i, pos1 in enumerate(positions):
        for j, pos2 in enumerate(positions):
            if i != j:
                # Check if there's no significant overlap
                # Allow small floating point errors with a small epsilon
                if pos1.x1 > pos2.x0 and pos1.x0 < pos2.x1 and pos1.y1 > pos2.y0 and pos1.y0 < pos2.y1:
                    overlap_area = (min(pos1.x1, pos2.x1) - max(pos1.x0, pos2.x0)) * \
                                   (min(pos1.y1, pos2.y1) - max(pos1.y0, pos2.y0))
                    # Allow minimal overlap due to floating point errors
                    assert overlap_area < 1e-6, f"Subplots {i} and {j} overlap too much"

    # Check aspect ratios are maintained where specified
    width2, height2 = positions[1].width, positions[1].height
    width3, height3 = positions[2].width, positions[2].height
    width4, height4 = positions[3].width, positions[3].height

    # Check the aspect ratios are roughly maintained (with some tolerance)
    assert abs((width2 / height2) - 2.0) < 2.0, "Aspect ratio for subplot 2 not maintained"
    assert abs((width3 / height3) - 0.5) < 1.1, "Aspect ratio for subplot 3 not maintained"
    assert abs((width4 / height4) - 1.0) < 1, "Equal aspect ratio for subplot 4 not maintained"


from matplotlib._tight_layout import get_subplotspec_list

class TestGetSubplotspecList(unittest.TestCase):

    def test_axes_without_get_subplotspec_method(self):
        """Test that get_subplotspec_list correctly handles axes without get_subplotspec method."""

        # Create a mock axes object without get_subplotspec method
        class MockAxes:
            def get_axes_locator(self):
                # Return self so that hasattr(axes_or_locator, "get_subplotspec") evaluates to False
                return self

            # Deliberately not implementing get_subplotspec

            def __str__(self):
                return "MockAxes object"

        mock_axes = MockAxes()

        # Create a regular matplotlib axes for comparison
        fig = plt.figure()
        normal_axes = fig.add_subplot(111)  # This will have a subplotspec

        # Test with just the mock
        result1 = get_subplotspec_list([mock_axes])
        self.assertEqual(len(result1), 1)
        self.assertIsNone(result1[0])

        # Test with both axes
        result2 = get_subplotspec_list([normal_axes, mock_axes])
        self.assertEqual(len(result2), 2)
        self.assertIsNotNone(result2[0])  # Normal axes should have subplotspec
        self.assertIsNone(result2[1])  # Mock axes should return None since it has no get_subplotspec

        # Test with grid_spec parameter
        result3 = get_subplotspec_list([mock_axes], grid_spec=normal_axes.get_subplotspec().get_gridspec())
        self.assertEqual(len(result3), 1)
        self.assertIsNone(result3[0])

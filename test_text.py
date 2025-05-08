from datetime import datetime
import io
import warnings

import numpy as np
from numpy.testing import assert_almost_equal
from packaging.version import parse as parse_version
import pyparsing
import pytest

import matplotlib as mpl
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.transforms as mtransforms
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
from matplotlib.text import Text, Annotation, OffsetFrom

pyparsing_version = parse_version(pyparsing.__version__)


def test_text_color_properties():
    """Test setting and getting color properties for Text objects."""
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, "Hello", color='red')

    # Test color setting and retrieval
    assert t.get_color() == 'red'

    # Test changing color
    t.set_color('blue')
    assert t.get_color() == 'blue'

    # Test that c is an alias for color
    t.set(c='green')
    assert t.get_color() == 'green'


def test_text_fontsize_properties():
    """Test setting and getting fontsize properties for Text objects."""
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, "Hello", fontsize=12)

    # Test fontsize setting and retrieval
    assert t.get_fontsize() == 12

    # Test changing fontsize
    t.set_fontsize(20)
    assert t.get_fontsize() == 20

    # Test that size is an alias for fontsize
    t.set(size=15)
    assert t.get_fontsize() == 15


def test_text_fontweight_properties():
    """Test setting and getting fontweight properties for Text objects."""
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, "Hello", fontweight='bold')

    # Test fontweight setting and retrieval
    assert t.get_weight() == 'bold'

    # Test changing fontweight
    t.set_weight('light')
    assert t.get_weight() == 'light'

    # Test numeric weight
    t.set_weight(700)
    assert t.get_weight() == 700


def test_text_fontstyle_properties():
    """Test setting and getting fontstyle properties for Text objects."""
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, "Hello", fontstyle='italic')

    # Test fontstyle setting and retrieval
    assert t.get_style() == 'italic'

    # Test changing fontstyle
    t.set_style('normal')
    assert t.get_style() == 'normal'


def test_text_fontstretch_properties():
    """Test setting and getting fontstretch properties for Text objects."""
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, "Hello", stretch='condensed')

    # Test fontstretch setting and retrieval
    assert t.get_stretch() == 'condensed'

    # Test changing fontstretch
    t.set_stretch('expanded')
    assert t.get_stretch() == 'expanded'

    # Test numeric stretch
    t.set_stretch(700)
    assert t.get_stretch() == 700


def test_text_alignment_properties():
    """Test setting and getting alignment properties for Text objects."""
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, "Hello", ha='center', va='center')

    # Test alignment setting and retrieval
    assert t.get_horizontalalignment() == 'center'
    assert t.get_verticalalignment() == 'center'

    # Test changing alignment
    t.set_horizontalalignment('right')
    t.set_verticalalignment('top')
    assert t.get_horizontalalignment() == 'right'
    assert t.get_verticalalignment() == 'top'

    # Test shortcuts
    t.set_ha('left')
    t.set_va('bottom')
    assert t.get_ha() == 'left'
    assert t.get_va() == 'bottom'


def test_text_rotation_properties():
    """Test setting and getting rotation properties for Text objects."""
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, "Hello", rotation=45)

    # Test rotation setting and retrieval
    assert t.get_rotation() == 45

    # Test changing rotation
    t.set_rotation(90)
    assert t.get_rotation() == 90

    # Test string rotation values
    t.set_rotation('vertical')
    assert t.get_rotation() == 90

    t.set_rotation('horizontal')
    assert t.get_rotation() == 0


def test_text_position_properties():
    """Test setting and getting position properties for Text objects."""
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.6, "Hello")

    # Test position setting and retrieval
    assert t.get_position() == (0.5, 0.6)

    # Test changing position
    t.set_position((0.7, 0.8))
    assert t.get_position() == (0.7, 0.8)

    # Test individual x and y positions
    t.set_x(0.3)
    t.set_y(0.4)
    assert t.get_position() == (0.3, 0.4)


def test_text_clip_properties():
    """Test setting and getting clip properties for Text objects."""
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, "Hello", clip_on=True)

    # Test clip_on setting and retrieval
    assert t.get_clip_on() == True

    # Test changing clip_on
    t.set_clip_on(False)
    assert t.get_clip_on() == False


def test_text_alpha_properties():
    """Test setting and getting alpha properties for Text objects."""
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, "Hello", alpha=0.5)

    # Test alpha setting and retrieval
    assert t.get_alpha() == 0.5

    # Test changing alpha
    t.set_alpha(0.7)
    assert t.get_alpha() == 0.7


def test_text_zorder_properties():
    """Test setting and getting zorder properties for Text objects."""
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, "Hello", zorder=10)

    # Test zorder setting and retrieval
    assert t.get_zorder() == 10

    # Test changing zorder
    t.set_zorder(5)
    assert t.get_zorder() == 5


def test_text_visible_properties():
    """Test setting and getting visible properties for Text objects."""
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, "Hello", visible=False)

    # Test visible setting and retrieval
    assert t.get_visible() == False

    # Test changing visible
    t.set_visible(True)
    assert t.get_visible() == True


def test_text_wrap_length():
    """Test that wrap correctly handles different text lengths."""
    fig, ax = plt.subplots(figsize=(2, 2))  # Small figure to force wrapping

    # Short text should not wrap
    t1 = ax.text(0.5, 0.8, "Short", wrap=True)
    fig.canvas.draw()
    assert t1._get_wrapped_text() == "Short"

    # Long text should wrap
    long_text = "This is a very long text that should definitely wrap on a small figure"
    t2 = ax.text(0.5, 0.5, long_text, wrap=True)
    fig.canvas.draw()
    wrapped = t2._get_wrapped_text()
    assert wrapped != long_text
    assert "\n" in wrapped


def test_annotation_update_position():
    """Test that annotation position updates correctly work."""
    fig, ax = plt.subplots()
    ann = ax.annotate("Test", xy=(0.2, 0.2), xycoords='data',
                      xytext=(0.5, 0.5), textcoords='axes fraction')

    # Initial position
    fig.canvas.draw()
    initial_bbox = ann.get_window_extent(fig.canvas.renderer)

    # Update position
    ann.xy = (0.8, 0.8)
    ann.xyann = (0.1, 0.1)
    fig.canvas.draw()
    new_bbox = ann.get_window_extent(fig.canvas.renderer)

    # Position should change
    assert not np.allclose(initial_bbox.get_points(), new_bbox.get_points(), rtol=1e-6)


def test_multiline_text_dimensions():
    """Test that multiline text has appropriate dimensions."""
    fig, ax = plt.subplots()
    s1 = ax.text(0.5, 0.5, "Single line")
    s2 = ax.text(0.5, 0.7, "Line 1\nLine 2\nLine 3")

    fig.canvas.draw()

    # Multiline text should be taller
    bbox1 = s1.get_window_extent(fig.canvas.renderer)
    bbox2 = s2.get_window_extent(fig.canvas.renderer)

    assert bbox2.height > bbox1.height

    # Height should be roughly proportional to number of lines
    # (with some allowance for different line spacing)
    assert bbox2.height > 2.5 * bbox1.height


def test_annotation_different_coordinate_systems():
    """Test annotation with different coordinate systems for text and position."""
    fig, ax = plt.subplots()

    # Create annotations with different coordinate systems
    ann1 = ax.annotate("Test 1", xy=(0.5, 0.5), xycoords='data',
                       xytext=(10, 20), textcoords='offset points')

    ann2 = ax.annotate("Test 2", xy=(0.5, 0.5), xycoords='axes fraction',
                       xytext=(0.1, 0.1), textcoords='figure fraction')

    # Draw to initialize positions
    fig.canvas.draw()

    # Verify that different coordinate systems result in different positions
    bbox1 = ann1.get_window_extent(fig.canvas.renderer)
    bbox2 = ann2.get_window_extent(fig.canvas.renderer)

    assert not np.allclose(bbox1.get_points(), bbox2.get_points(), rtol=1e-6)


def test_text_update_from():
    """Test that Text.update_from correctly copies properties."""
    fig, ax = plt.subplots()
    t1 = ax.text(0.1, 0.1, "Original", color='red', fontsize=14, rotation=45,
                 ha='left', va='bottom', fontweight='bold')
    t2 = ax.text(0.5, 0.5, "Copy")

    # Update t2 from t1
    t2.update_from(t1)

    # Check that properties were copied
    assert t2.get_color() == 'red'
    assert t2.get_fontsize() == 14
    assert t2.get_rotation() == 45
    assert t2.get_ha() == 'left'
    assert t2.get_va() == 'bottom'
    assert t2.get_weight() == 'bold'

    # Position should not be copied
    assert t2.get_position() == (0.5, 0.5)
    assert t2.get_text() == "Copy"


def test_text_get_rotation_mode():
    """Test getting and setting rotation mode of Text objects."""
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, "Test", rotation=45)

    # Default rotation mode
    assert t.get_rotation_mode() == 'default'

    # Set rotation mode
    t.set_rotation_mode("anchor")
    assert t.get_rotation_mode() == "anchor"

    t.set_rotation_mode("default")
    assert t.get_rotation_mode() == "default"

    # Invalid rotation mode should raise ValueError
    with pytest.raises(ValueError):
        t.set_rotation_mode("invalid_mode")


def test_text_get_ref_coord():
    """Test getting and setting reference coordinates of Text objects."""
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, "Test")

    # Default reference point
    assert t._x == 0.5
    assert t._y == 0.5

    # Change reference point
    t._x = 0.7
    t._y = 0.3
    assert t._x == 0.7
    assert t._y == 0.3


def test_text_get_transform():
    """Test getting and setting transforms of Text objects."""
    fig, ax = plt.subplots()

    # Default transform is data transform
    t1 = ax.text(0.5, 0.5, "Test 1")
    assert t1.get_transform() == ax.transData

    # Specify axes transform
    t2 = ax.text(0.5, 0.5, "Test 2", transform=ax.transAxes)
    assert t2.get_transform() == ax.transAxes

    # Specify figure transform
    t3 = fig.text(0.5, 0.5, "Test 3")
    assert t3.get_transform() == fig.transFigure

    # Change transform
    custom_transform = mtransforms.Affine2D().scale(2, 2) + ax.transData
    t1.set_transform(custom_transform)
    assert t1.get_transform() == custom_transform


def test_text_contains_different_renderers():
    """Test that contains method works with different renderers."""
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, "Test", fontsize=20)  # Large fontsize for reliable hit testing

    # Draw the figure to initialize text layout
    fig.canvas.draw()

    # Get renderer from canvas
    renderer = fig.canvas.get_renderer()

    # Get the window extent of the text
    bbox = t.get_window_extent(renderer)

    # Create a point inside the text bbox in display coordinates
    x_inside = (bbox.x0 + bbox.x1) / 2
    y_inside = (bbox.y0 + bbox.y1) / 2

    # Create a mouse event at this location
    event_inside = MouseEvent('button_press_event', fig.canvas, x_inside, y_inside, 1, None)

    # Create a point outside the text bbox
    x_outside = bbox.x1 + 10
    y_outside = bbox.y1 + 10
    event_outside = MouseEvent('button_press_event', fig.canvas, x_outside, y_outside, 1, None)

    # Test contains method
    contains_inside, details_inside = t.contains(event_inside)
    contains_outside, details_outside = t.contains(event_outside)

    assert contains_inside == True
    assert contains_outside == False


def test_annotation_arrow_styles():
    """Test different arrow styles for annotations."""
    fig, ax = plt.subplots()
    arrow_styles = ['->', '-[', '-|>', '<->', '<|-|>', 'fancy', 'simple', 'wedge']

    annotations = []
    for i, style in enumerate(arrow_styles):
        ann = ax.annotate(f"Style: {style}",
                          xy=(0.2, 0.1 + i * 0.1),
                          xytext=(0.5, 0.1 + i * 0.1),
                          arrowprops=dict(arrowstyle=style))
        annotations.append(ann)

    # Draw to initialize
    fig.canvas.draw()

    # Verify that all annotations have arrow patches
    for ann in annotations:
        assert ann.arrow_patch is not None


def test_annotation_textcoords_offset():
    """Test that offset coordinates work correctly in annotations."""
    fig, ax = plt.subplots()
    ann1 = ax.annotate("test", xy=(0.5, 0.5), xycoords='data',
                       xytext=(20, 20), textcoords='offset points')

    ann2 = ax.annotate("test", xy=(0.5, 0.5), xycoords='data',
                       xytext=(20, 20), textcoords='offset points',
                       annotation_clip=True)

    # Draw to initialize
    fig.canvas.draw()

    # Check that the annotation_clip property is correctly set
    assert ann1.get_annotation_clip() == None
    assert ann2.get_annotation_clip() == True

    # Verify that the text positions are offset from data point
    renderer = fig.canvas.get_renderer()
    bbox1 = ann1.get_window_extent(renderer)
    data_point = ax.transData.transform((0.5, 0.5))

    # The text position should be offset from the data point
    # We're checking that the text isn't directly on top of the data point
    text_center_x = (bbox1.x0 + bbox1.x1) / 2
    text_center_y = (bbox1.y0 + bbox1.y1) / 2

    assert abs(text_center_x - data_point[0]) > 10
    assert abs(text_center_y - data_point[1]) > 10


def test_annotation_get_set_visible():
    """Test visibility of annotation components."""
    fig, ax = plt.subplots()
    ann = ax.annotate("test", xy=(0.2, 0.2), xytext=(0.5, 0.5),
                      arrowprops=dict(arrowstyle="->"))

    # Both text and arrow should be visible by default
    assert ann.get_visible() == True
    assert ann.arrow_patch.get_visible() == True

    # Hide annotation
    ann.set_visible(False)
    assert ann.get_visible() == False

    # Show annotation
    ann.set_visible(True)
    assert ann.get_visible() == True
    assert ann.arrow_patch.get_visible() == True

    # Hide just the arrow
    ann.arrow_patch.set_visible(False)
    assert ann.get_visible() == True
    assert ann.arrow_patch.get_visible() == False


def test_text_get_set_usetex():
    """Test getting and setting usetex property."""
    fig, ax = plt.subplots()
    # Default value should match rcParams
    t1 = ax.text(0.1, 0.1, "Test 1")
    assert t1.get_usetex() == plt.rcParams['text.usetex']

    # Explicitly set usetex=True
    t2 = ax.text(0.2, 0.2, "Test 2", usetex=True)
    assert t2.get_usetex() == True

    # Explicitly set usetex=False
    t3 = ax.text(0.3, 0.3, "Test 3", usetex=False)
    assert t3.get_usetex() == False

    # Change usetex
    t1.set_usetex(True)
    assert t1.get_usetex() == True

    t2.set_usetex(False)
    assert t2.get_usetex() == False


def test_text_with_math():
    """Test text with math content."""
    fig, ax = plt.subplots()
    # Simple text with math
    t1 = ax.text(0.5, 0.5, r"$\alpha + \beta = \gamma$")

    # Text with math and regular content
    t2 = ax.text(0.5, 0.7, r"Regular text and $\alpha + \beta = \gamma$")

    # Draw to initialize
    fig.canvas.draw()

    # Both should have bbox
    assert t1.get_window_extent() is not None
    assert t2.get_window_extent() is not None

    # Text with math should still support color changes
    t1.set_color('red')
    assert t1.get_color() == 'red'


def test_text_with_custom_transform():
    """Test text with custom transform."""
    fig, ax = plt.subplots()

    # Create a custom transform that scales and rotates
    scale = mtransforms.Affine2D().scale(2, 0.5)
    rotate = mtransforms.Affine2D().rotate_deg(30)
    custom_transform = scale + rotate + ax.transData

    # Create text with custom transform
    t = ax.text(0.5, 0.5, "Custom Transform", transform=custom_transform)

    # Verify the transform
    assert t.get_transform() == custom_transform

    # Create a new custom transform
    new_transform = mtransforms.Affine2D().rotate_deg(45) + ax.transData

    # Update the transform
    t.set_transform(new_transform)
    assert t.get_transform() == new_transform


def test_text_override_rcparams():
    """Test that text properties override rcparams defaults."""
    original_size = plt.rcParams['font.size']
    original_family = plt.rcParams['font.family']

    try:
        # Change rcParams
        plt.rcParams['font.size'] = 12
        plt.rcParams['font.family'] = 'serif'

        fig, ax = plt.subplots()

        # Text with default properties
        t1 = ax.text(0.1, 0.1, "Default")
        assert t1.get_fontsize() == 12
        assert t1.get_fontfamily() == ['serif']

        # Text with overridden properties
        t2 = ax.text(0.2, 0.2, "Override", fontsize=20, family='sans-serif')
        assert t2.get_fontsize() == 20
        assert t2.get_fontfamily() == ['sans-serif']

        # Change rcParams after text creation
        plt.rcParams['font.size'] = 14

        # Default text should not change
        assert t1.get_fontsize() == 12

        # New text should use new defaults
        t3 = ax.text(0.3, 0.3, "New Default")
        assert t3.get_fontsize() == 14

    finally:
        # Restore original values
        plt.rcParams['font.size'] = original_size
        plt.rcParams['font.family'] = original_family


def test_text_multiline_alignment():
    """Test alignment of multiline text."""
    fig, ax = plt.subplots()

    # Create multiline text with different alignments
    text_left = ax.text(0.2, 0.5, "Line 1\nLonger Line 2\nL3", ha='left')
    text_center = ax.text(0.5, 0.5, "Line 1\nLonger Line 2\nL3", ha='center')
    text_right = ax.text(0.8, 0.5, "Line 1\nLonger Line 2\nL3", ha='right')

    # Draw to initialize
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Get bounding boxes
    bbox_left = text_left.get_window_extent(renderer)
    bbox_center = text_center.get_window_extent(renderer)
    bbox_right = text_right.get_window_extent(renderer)

    # All three bboxes should have the same width
    assert np.isclose(bbox_left.width, bbox_center.width, rtol=0.1)
    assert np.isclose(bbox_center.width, bbox_right.width, rtol=0.1)

    # But their positions should be different
    assert bbox_left.x0 < bbox_center.x0
    assert bbox_center.x0 < bbox_right.x0


def test_text_pickable():
    """Test picking of text objects."""
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, "Pickable Text")

    # Text should be pickable by default
    assert t.get_picker() == None

    # Make text not pickable
    t.set_picker(False)
    assert t.get_picker() == False

    # Set custom picker function
    def custom_picker(text, event):
        return True, {}

    t.set_picker(custom_picker)
    assert t.get_picker() == custom_picker
    assert t.pickable()


def test_text_set_text():
    """Test changing text content with set_text."""
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, "Original")

    # Check initial text
    assert t.get_text() == "Original"

    # Change text
    t.set_text("Updated")
    assert t.get_text() == "Updated"

    # Change to empty string
    t.set_text("")
    assert t.get_text() == ""

    # Change to number
    t.set_text(123)
    assert t.get_text() == "123"

    # Change to None (should convert to empty string)
    t.set_text(None)
    assert t.get_text() == ""


def test_text_stale_after_changes():
    """Test that text is marked stale after property changes."""
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, "Test")

    # Draw to initialize
    fig.canvas.draw()

    # After drawing, text should not be stale
    assert not t.stale

    # Changing properties should mark text as stale
    t.set_text("New Text")
    assert t.stale

    # Drawing again should clear stale flag
    fig.canvas.draw()
    assert not t.stale

    # Other property changes should also mark as stale
    t.set_color('red')
    assert t.stale

    t.set_fontsize(20)
    assert t.stale

    # Position changes too
    t.set_position((0.7, 0.7))
    assert t.stale



def test_text_get_set_wrap():
    """Test getting and setting wrap property."""
    fig, ax = plt.subplots()

    # Default wrap should be False
    t = ax.text(0.5, 0.5, "Test")
    assert t.get_wrap() == False

    # Set wrap to True
    t.set_wrap(True)
    assert t.get_wrap() == True

    # Set wrap to False
    t.set_wrap(False)
    assert t.get_wrap() == False


def test_text_parse_math():
    """Test parse_math property."""
    fig, ax = plt.subplots()

    # Default parse_math should be True
    t1 = ax.text(0.1, 0.1, "$x^2$")
    assert t1.get_parse_math() == True

    # Explicitly set parse_math=False
    t2 = ax.text(0.2, 0.2, "$x^2$", parse_math=False)
    assert t2.get_parse_math() == False

    # Change parse_math
    t1.set_parse_math(False)
    assert t1.get_parse_math() == False

    t2.set_parse_math(True)
    assert t2.get_parse_math() == True


def test_text_fontfamily_shorthand():
    """Test font family shorthand properties."""
    fig, ax = plt.subplots()

    # Test serif font
    t1 = ax.text(0.1, 0.1, "Serif", family='serif')
    assert t1.get_fontfamily() == ['serif']

    # Test sans-serif font
    t2 = ax.text(0.2, 0.2, "Sans-serif", family='sans-serif')
    assert t2.get_fontfamily() == ['sans-serif']

    # Test monospace font
    t3 = ax.text(0.3, 0.3, "Monospace", family='monospace')
    assert t3.get_fontfamily() == ['monospace']

    # Test custom font name
    t4 = ax.text(0.4, 0.4, "Custom", family='Arial')
    assert 'Arial' in t4.get_fontfamily()



def test_text_url_property():
    """Test URL property for text."""
    fig, ax = plt.subplots()

    # Default URL should be None
    t = ax.text(0.5, 0.5, "Hyperlink")
    assert t.get_url() is None

    # Set URL
    t.set_url("https://matplotlib.org")
    assert t.get_url() == "https://matplotlib.org"

    # Create text with URL
    t2 = ax.text(0.6, 0.6, "Another link", url="https://python.org")
    assert t2.get_url() == "https://python.org"


def test_text_path_effects():
    """Test path effects for text."""
    from matplotlib.patheffects import withStroke
    fig, ax = plt.subplots()

    # Default path effects should be empty
    t = ax.text(0.5, 0.5, "Text")
    assert len(t.get_path_effects()) == 0

    # Set path effects
    stroke = withStroke(linewidth=3, foreground='red')
    t.set_path_effects([stroke])

    # Check that path effects were set
    effects = t.get_path_effects()
    assert len(effects) == 1
    assert effects[0] == stroke


def test_text_get_set_gid():
    """Test getting and setting gid (group id) for text."""
    fig, ax = plt.subplots()

    # Default gid should be None
    t = ax.text(0.5, 0.5, "Text")
    assert t.get_gid() is None

    # Set gid
    t.set_gid("text_group")
    assert t.get_gid() == "text_group"

    # Create text with gid
    t2 = ax.text(0.6, 0.6, "Text 2", gid="another_group")
    assert t2.get_gid() == "another_group"



def test_text_contains_with_bbox():
    """Test contains method for text with bbox."""
    fig, ax = plt.subplots()
    t = ax.text(0.5, 0.5, "Text with bbox",
                bbox=dict(facecolor='red', edgecolor='blue', pad=10))

    # Draw to initialize
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Get window extent
    bbox = t.get_window_extent(renderer)

    # Create events inside and outside the bbox
    inside_x = (bbox.x0 + bbox.x1) / 2
    inside_y = (bbox.y0 + bbox.y1) / 2
    event_inside = MouseEvent('button_press_event', fig.canvas,
                              inside_x, inside_y, 1, None)

    outside_x = bbox.x1 + 50
    outside_y = bbox.y1 + 50
    event_outside = MouseEvent('button_press_event', fig.canvas,
                               outside_x, outside_y, 1, None)

    # Test contains method
    contains_inside, details_inside = t.contains(event_inside)
    contains_outside, details_outside = t.contains(event_outside)

    assert contains_inside == True
    assert contains_outside == False



def test_text_vert_alignment_with_rotation():
    """Test vertical alignment with rotation."""
    fig, ax = plt.subplots()
    alignments = ['top', 'center', 'bottom', 'baseline']
    texts = []

    # Create text with different vertical alignments and rotation
    for i, va in enumerate(alignments):
        t = ax.text(0.2 * i + 0.1, 0.5, "Test", va=va, rotation=45)
        texts.append(t)

    # Verify that all vertical alignments were set correctly
    for t, va in zip(texts, alignments):
        assert t.get_verticalalignment() == va



def test_text_fontproperties_object():
    """Test using FontProperties object with text."""
    from matplotlib.font_manager import FontProperties

    fig, ax = plt.subplots()

    # Create FontProperties
    font_prop = FontProperties(family='serif', weight='bold', style='italic', size=14)

    # Create text with FontProperties
    t = ax.text(0.5, 0.5, "Test", fontproperties=font_prop)

    # Check that properties were set
    assert t.get_fontfamily() == ['serif']
    assert t.get_weight() == 'bold'
    assert t.get_style() == 'italic'
    assert t.get_fontsize() == 14

    # Change font properties
    new_font = FontProperties(family='sans-serif', weight='normal', size=10)
    t.set_fontproperties(new_font)

    # Check that properties were updated
    assert t.get_fontfamily() == ['sans-serif']
    assert t.get_weight() == 'normal'
    assert t.get_fontsize() == 10


def test_text_custom_rasterized():
    """Test setting custom rasterized for text."""
    fig, ax = plt.subplots()

    # Default rasterized should be None or False
    t = ax.text(0.5, 0.5, "Test")
    assert t.get_rasterized() in (None, False)

    # Set rasterized to True
    t.set_rasterized(True)
    assert t.get_rasterized() == True

    # Set rasterized to False
    t.set_rasterized(False)
    assert t.get_rasterized() == False

    # Create text with rasterized=True
    t2 = ax.text(0.6, 0.6, "Rasterized", rasterized=True)
    assert t2.get_rasterized() == True


def test_text_deepcopy():
    """Test deepcopy of text objects."""
    import copy

    fig, ax = plt.subplots()
    t1 = ax.text(0.5, 0.5, "Original", color='red', fontsize=14)

    # Deep copy the text
    t2 = copy.deepcopy(t1)

    # Properties should be the same
    assert t2.get_text() == "Original"
    assert t2.get_color() == 'red'
    assert t2.get_fontsize() == 14
    assert t2.get_position() == (0.5, 0.5)

    # Changing t1 should not affect t2
    t1.set_text("Changed")
    t1.set_color('blue')
    assert t2.get_text() == "Original"
    assert t2.get_color() == 'red'


def test_text_get_set_zorder():
    """Test getting and setting zorder for text."""
    fig, ax = plt.subplots()

    # Default zorder
    t = ax.text(0.5, 0.5, "Test")
    assert t.get_zorder() == 3  # Default text zorder in matplotlib

    # Set zorder
    t.set_zorder(10)
    assert t.get_zorder() == 10

    # Create text with custom zorder
    t2 = ax.text(0.6, 0.6, "Higher", zorder=20)
    assert t2.get_zorder() == 20

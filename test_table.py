import datetime
from unittest.mock import Mock
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.table import CustomCell, Table
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.transforms import Bbox
import matplotlib.units as munits


## Basic Table Creation Tests

def test_simple_table_creation():
    fig, ax = plt.subplots()
    table = ax.table(cellText=[['A', 'B'], ['C', 'D']], loc='center')
    assert len(table.get_celld()) == 4
    plt.close(fig)


def test_table_positioning():
    positions = ['best', 'upper right', 'lower left', 'center']
    fig, axs = plt.subplots(len(positions), 1)
    for ax, loc in zip(axs, positions):
        ax.axis('off')
        ax.table(cellText=[['A']], loc=loc)
    plt.close(fig)


## Cell Content and Formatting Tests

def test_table_with_numeric_data():
    fig, ax = plt.subplots()
    numeric_data = [[1, 2.5], [3.14159, 4]]
    table = ax.table(cellText=numeric_data, loc='center')
    assert table[(1, 0)].get_text().get_text() == '3.14159'
    plt.close(fig)


def test_table_with_mixed_data_types():
    fig, ax = plt.subplots()
    mixed_data = [['Text', 123], [True, None]]
    table = ax.table(cellText=mixed_data, loc='center')
    assert table[(0, 1)].get_text().get_text() == '123'
    assert table[(1, 0)].get_text().get_text() == 'True'
    plt.close(fig)


def test_table_cell_alignment():
    alignments = ['left', 'center', 'right']
    fig, axs = plt.subplots(len(alignments), 1)
    data = [['Aligned Text', 123]]

    for ax, align in zip(axs, alignments):
        ax.axis('off')
        table = ax.table(cellText=data, loc='center')
        for key in table.get_celld():
            table[key].get_text().set_horizontalalignment(align)
    plt.close(fig)


## Table Styling Tests

def test_table_cell_colors():
    fig, ax = plt.subplots()
    ax.axis('off')
    cell_colors = [[(.9, .9, .9), (.8, .2, .2)],
                   [(.2, .8, .2), (.2, .2, .8)]]
    ax.table(cellText=[['A', 'B'], ['C', 'D']],
             cellColours=cell_colors,
             loc='center')
    plt.close(fig)


def test_table_row_column_colors():
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.table(cellText=[['A', 'B'], ['C', 'D']],
             rowColours=['lightblue', 'lightgreen'],
             colColours=['pink', 'yellow'],
             loc='center')
    plt.close(fig)


def test_table_line_properties():
    fig, ax = plt.subplots()
    ax.axis('off')
    tb = ax.table(cellText=[['A', 'B'], ['C', 'D']], loc='center')
    for _, cell in tb.get_celld().items():
        cell.set_linewidth(2)
        cell.set_edgecolor('darkblue')
    plt.close(fig)


## Advanced Table Features

def test_table_with_colspans():
    fig, ax = plt.subplots()
    ax.axis('off')
    tb = Table(ax, bbox=[0.1, 0.1, 0.8, 0.8])
    tb.add_cell(0, 0, 1, 2, text='Span Columns 0-1')
    tb.add_cell(1, 0, 1, 1, text='Col 0')
    tb.add_cell(1, 1, 1, 1, text='Col 1')
    ax.add_table(tb)
    plt.close(fig)


def test_table_with_rowspans():
    fig, ax = plt.subplots()
    ax.axis('off')
    tb = Table(ax, bbox=[0.1, 0.1, 0.8, 0.8])
    tb.add_cell(0, 0, 2, 1, text='Span Rows 0-1')
    tb.add_cell(0, 1, 1, 1, text='Row 0')
    tb.add_cell(1, 1, 1, 1, text='Row 1')
    ax.add_table(tb)
    plt.close(fig)


def test_table_highlight_cells():
    fig, ax = plt.subplots()
    ax.axis('off')
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    cell_text = [[str(val) for val in row] for row in data]
    tb = ax.table(cellText=cell_text, loc='center')
    for (row, col), cell in tb.get_celld().items():
        if row > 0 and col < 3:
            value = data[row - 1][col]
            cell.set_facecolor('lightcoral' if value > 5 else 'lightblue')
    plt.close(fig)


## Text Formatting Tests

def test_table_mathtext():
    fig, ax = plt.subplots()
    ax.axis('off')
    math_text = [['$\\alpha$', '$\\beta$'],
                 ['$\\gamma^2$', '$\\frac{x}{y}$']]
    tb = ax.table(cellText=math_text, loc='center')
    tb.set_fontsize(14)
    plt.close(fig)


def test_table_multiline_text():
    fig, ax = plt.subplots()
    ax.axis('off')
    cell_text = [['Line 1\nLine 2', 'Single Line'],
                 ['Single Line', 'Line 1\nLine 2\nLine 3']]
    tb = ax.table(cellText=cell_text, loc='center')
    plt.close(fig)


def test_table_with_multiple_fonts():
    fig, ax = plt.subplots()
    ax.axis('off')
    data = [['Default', 'Bold'], ['Italic', 'Sans']]
    tb = ax.table(cellText=data, loc='center')
    tb[0, 0].get_text().set_fontfamily('serif')
    tb[0, 1].get_text().set_fontweight('bold')
    tb[1, 0].get_text().set_style('italic')
    plt.close(fig)


## Table Layout Tests

def test_table_scale():
    fig, axs = plt.subplots(2, 1)
    data = [['A', 'B'], ['C', 'D']]

    axs[0].axis('off')
    axs[0].table(cellText=data, loc='center')

    axs[1].axis('off')
    tb = axs[1].table(cellText=data, loc='center')
    for cell in tb.get_celld().values():
        cell.set_height(cell.get_height() * 2)
        cell.set_width(cell.get_width() * 2)
    plt.close(fig)


def test_table_with_custom_padding():
    fig, axs = plt.subplots(3, 1)
    for i, pad in enumerate([0.01, 0.1, 0.5]):
        axs[i].axis('off')
        tb = axs[i].table(cellText=[['Padding', f'{pad}']], loc='center')
        for cell in tb.get_celld().values():
            cell.PAD = pad
    plt.close(fig)


## Table Interaction Tests

def test_table_add_remove_cells():
    fig, ax = plt.subplots()
    ax.axis('off')
    tb = Table(ax, bbox=[0.1, 0.1, 0.8, 0.8])
    tb.add_cell(0, 0, 1, 1, text='A')
    tb.add_cell(0, 1, 1, 1, text='B')
    tb._cells.pop((0, 0))
    tb.add_cell(0, 0, 1, 1, text='X')
    ax.add_table(tb)
    plt.close(fig)


def test_table_cell_properties():
    fig, ax = plt.subplots()
    ax.axis('off')
    tb = ax.table(cellText=[['A', 'B'], ['C', 'D']], loc='center')
    cell = tb[(0, 0)]
    cell.set_text_props(color='red', fontsize=12)
    cell.set_edgecolor('blue')
    cell.set_facecolor('yellow')
    assert cell.get_text().get_color() == 'red'
    plt.close(fig)


## Special Case Tests


def test_table_with_large_data():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    data = [[f'R{r}C{c}' for c in range(20)] for r in range(20)]
    ax.table(cellText=data, loc='center')
    plt.close(fig)


def test_table_with_unicode():
    fig, ax = plt.subplots()
    ax.axis('off')
    data = [['日本語', 'Русский'], ['中文', 'العربية']]
    ax.table(cellText=data, loc='center')
    plt.close(fig)

## Table Comparison Tests

@check_figures_equal()
def test_table_bbox(fig_test, fig_ref):
    data = [[2, 3], [4, 5]]
    col_labels = ('Foo', 'Bar')
    row_labels = ('Ada', 'Bob')

    ax_test = fig_test.subplots()
    ax_test.table(cellText=data, rowLabels=row_labels, colLabels=col_labels,
                  loc='center', bbox=[0.1, 0.2, 0.8, 0.6])

    ax_ref = fig_ref.subplots()
    ax_ref.table(cellText=data, rowLabels=row_labels, colLabels=col_labels,
                 loc='center', bbox=Bbox.from_extents(0.1, 0.2, 0.9, 0.8))


def test_customcell():
    types = ('horizontal', 'vertical', 'open', 'closed', 'T', 'R', 'B', 'L')
    codes = (
        (Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO, Path.MOVETO),
        (Path.MOVETO, Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO),
        (Path.MOVETO, Path.MOVETO, Path.MOVETO, Path.MOVETO, Path.MOVETO),
        (Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY),
        (Path.MOVETO, Path.MOVETO, Path.MOVETO, Path.LINETO, Path.MOVETO),
        (Path.MOVETO, Path.MOVETO, Path.LINETO, Path.MOVETO, Path.MOVETO),
        (Path.MOVETO, Path.LINETO, Path.MOVETO, Path.MOVETO, Path.MOVETO),
        (Path.MOVETO, Path.MOVETO, Path.MOVETO, Path.MOVETO, Path.LINETO),
    )

    for t, c in zip(types, codes):
        cell = CustomCell((0, 0), visible_edges=t, width=1, height=1)
        code = tuple(s for _, s in cell.get_path().iter_segments())
        assert c == code


## Table with External Data Tests

def test_table_with_numpy_array():
    data = np.array([[1, 2], [3, 4]])
    fig, ax = plt.subplots()
    table = ax.table(cellText=data, loc='center')
    assert table[(1, 0)].get_text().get_text() == '3'
    plt.close(fig)


## Table Unit Tests

def test_table_unit():
    class FakeUnit:
        def __init__(self, thing): pass

        def __repr__(self): return "Hello"

    fake_convertor = munits.ConversionInterface()
    fake_convertor.convert = Mock(side_effect=lambda v, u, a: 0)
    fake_convertor.default_units = Mock(side_effect=lambda v, a: None)
    fake_convertor.axisinfo = Mock(side_effect=lambda u, a: munits.AxisInfo())

    munits.registry[FakeUnit] = fake_convertor

    fig = plt.figure()
    fig.subplots().table([[FakeUnit("test")]])
    fake_convertor.convert.assert_not_called()
    munits.registry.pop(FakeUnit)
    plt.close(fig)


## Table Font Tests

def test_table_fontsize():
    tableData = [['a', 1], ['b', 2]]
    fig, ax = plt.subplots()
    test_fontsize = 20
    t = ax.table(cellText=tableData, loc='top', fontsize=test_fontsize)
    assert t[(0, 0)].get_fontsize() == test_fontsize
    plt.close(fig)


def test_table_font_properties():
    fig, ax = plt.subplots()
    tb = ax.table(cellText=[['A', 'B']], loc='center')
    cell = tb[(0, 0)]
    cell.get_text().set_fontfamily('serif')
    cell.get_text().set_fontstyle('italic')
    cell.get_text().set_fontweight('bold')
    assert cell.get_text().get_fontfamily() == ['serif']
    plt.close(fig)


## Table Edge Case Tests

def test_table_with_empty_cells():
    fig, ax = plt.subplots()
    ax.axis('off')
    data = [['A', ''], [None, 'D']]
    ax.table(cellText=data, loc='center')
    plt.close(fig)


def test_table_with_none_values():
    fig, ax = plt.subplots()
    ax.axis('off')
    data = [[None, None], [None, None]]
    ax.table(cellText=data, loc='center')
    plt.close(fig)


## Table Visual Properties Tests

def test_table_alpha_transparency():
    fig, ax = plt.subplots()
    ax.axis('off')
    data = [['A', 'B'], ['C', 'D']]
    tb = ax.table(cellText=data, loc='center')
    for i, cell in enumerate(tb.get_celld().values()):
        cell.set_alpha(0.5 + 0.1 * i)
        cell.set_facecolor('skyblue')
    plt.close(fig)


def test_table_rotation():
    fig, ax = plt.subplots()
    ax.axis('off')
    data = [['A', 'B'], ['C', 'D']]
    tb = ax.table(cellText=data, loc='center')
    rotations = [0, 45, 90, -45]
    for i, cell in enumerate(tb.get_celld().values()):
        cell.get_text().set_rotation(rotations[i % len(rotations)])
    plt.close(fig)


## Table Label Styling Tests

def test_table_row_column_labels_styling():
    fig, ax = plt.subplots()
    ax.axis('off')
    data = [['A', 'B'], ['C', 'D']]
    tb = ax.table(cellText=data,
                  rowLabels=['Row 1', 'Row 2'],
                  colLabels=['Col 1', 'Col 2'],
                  loc='center')
    for i in range(2):
        tb[i + 1, -1].set_facecolor('lightgray')  # Row labels
        tb[0, i].set_facecolor('lightgray')  # Column labels
    plt.close(fig)


def test_table_with_cell_borders():
    fig, ax = plt.subplots()
    ax.axis('off')
    tb = ax.table(cellText=[['A', 'B'], ['C', 'D']], loc='center')
    styles = [('solid', 2), ('dashed', 1), ('dotted', 1), ('dashdot', 2)]
    for i, cell in enumerate(tb.get_celld().values()):
        style, width = styles[i % len(styles)]
        cell.set_linestyle(style)
        cell.set_linewidth(width)
    plt.close(fig)


## Table Auto-sizing Tests

def test_table_auto_column_width():
    fig, ax = plt.subplots()
    ax.axis('off')
    tb = ax.table(
        cellText=[['Short', 'Long text that needs more space']],
        loc="center")
    tb.auto_set_column_width([0, 1])
    plt.close(fig)


def test_table_auto_font_size():
    fig, ax = plt.subplots()
    ax.axis('off')
    tb = ax.table(
        cellText=[['Text', 'More text'], ['Even more text', 'Lots of text here']],
        loc="center")
    tb.auto_set_font_size(True)
    plt.close(fig)


## Table with Special Characters

def test_table_with_special_chars():
    fig, ax = plt.subplots()
    ax.axis('off')
    special_chars = [['#', '$'], ['%', '&']]
    ax.table(cellText=special_chars, loc='center')
    plt.close(fig)


def test_table_with_newlines():
    fig, ax = plt.subplots()
    ax.axis('off')
    text_with_newlines = [['Line1\nLine2', 'Single'], ['Text', 'Multi\nLine\nText']]
    ax.table(cellText=text_with_newlines, loc='center')
    plt.close(fig)


## Table with Different Units

def test_table_with_inches_units():
    fig, ax = plt.subplots()
    ax.axis('off')
    tb = Table(ax, bbox=Bbox.from_bounds(1, 1, 2, 2))  # Inches
    tb.add_cell(0, 0, 1, 1, text='A')
    ax.add_table(tb)
    plt.close(fig)


## Table with Different Coordinate Systems

def test_table_with_axes_coordinates():
    fig, ax = plt.subplots()
    ax.axis('off')
    tb = Table(ax, bbox=[0.1, 0.1, 0.8, 0.8])  # Axes coordinates
    tb.add_cell(0, 0, 1, 1, text='A')
    ax.add_table(tb)
    plt.close(fig)


## Table with Mixed Coordinate Systems

def test_table_with_mixed_coordinates():
    fig, ax = plt.subplots()
    ax.axis('off')
    tb = Table(ax, bbox=Bbox.from_bounds(0.1, 0.1, 0.8, 0.8))  # Mixed
    tb.add_cell(0, 0, 1, 1, text='A')
    ax.add_table(tb)
    plt.close(fig)


## Table with Custom Transformations

def test_table_with_transformed_coordinates():
    fig, ax = plt.subplots()
    ax.axis('off')
    trans = ax.transAxes + ax.transData.inverted()
    tb = Table(ax, bbox=Bbox.from_bounds(0.1, 0.1, 0.8, 0.8), transform=trans)
    tb.add_cell(0, 0, 1, 1, text='A')
    ax.add_table(tb)
    plt.close(fig)
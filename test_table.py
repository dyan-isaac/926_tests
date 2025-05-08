import datetime
from unittest.mock import Mock

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.table import CustomCell, Table
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.transforms import Bbox
import matplotlib.units as munits


# def test_non_square():
#     # Check that creating a non-square table works
#     cellcolors = ['b', 'r']
#     plt.table(cellColours=cellcolors)
#
#
# @image_comparison(['table_zorder.png'], remove_text=True)
# def test_zorder():
#     data = [[66386, 174296],
#             [58230, 381139]]
#
#     colLabels = ('Freeze', 'Wind')
#     rowLabels = ['%d year' % x for x in (100, 50)]
#
#     cellText = []
#     yoff = np.zeros(len(colLabels))
#     for row in reversed(data):
#         yoff += row
#         cellText.append(['%1.1f' % (x/1000.0) for x in yoff])
#
#     t = np.linspace(0, 2*np.pi, 100)
#     plt.plot(t, np.cos(t), lw=4, zorder=2)
#
#     plt.table(cellText=cellText,
#               rowLabels=rowLabels,
#               colLabels=colLabels,
#               loc='center',
#               zorder=-2,
#               )
#
#     plt.table(cellText=cellText,
#               rowLabels=rowLabels,
#               colLabels=colLabels,
#               loc='upper center',
#               zorder=4,
#               )
#     plt.yticks([])
#
#
# @image_comparison(['table_labels.png'])
# def test_label_colours():
#     dim = 3
#
#     c = np.linspace(0, 1, dim)
#     colours = plt.colormaps["RdYlGn"](c)
#     cellText = [['1'] * dim] * dim
#
#     fig = plt.figure()
#
#     ax1 = fig.add_subplot(4, 1, 1)
#     ax1.axis('off')
#     ax1.table(cellText=cellText,
#               rowColours=colours,
#               loc='best')
#
#     ax2 = fig.add_subplot(4, 1, 2)
#     ax2.axis('off')
#     ax2.table(cellText=cellText,
#               rowColours=colours,
#               rowLabels=['Header'] * dim,
#               loc='best')
#
#     ax3 = fig.add_subplot(4, 1, 3)
#     ax3.axis('off')
#     ax3.table(cellText=cellText,
#               colColours=colours,
#               loc='best')
#
#     ax4 = fig.add_subplot(4, 1, 4)
#     ax4.axis('off')
#     ax4.table(cellText=cellText,
#               colColours=colours,
#               colLabels=['Header'] * dim,
#               loc='best')
#
#
# @image_comparison(['table_cell_manipulation.png'], style='mpl20')
# def test_diff_cell_table(text_placeholders):
#     cells = ('horizontal', 'vertical', 'open', 'closed', 'T', 'R', 'B', 'L')
#     cellText = [['1'] * len(cells)] * 2
#     colWidths = [0.1] * len(cells)
#
#     _, axs = plt.subplots(nrows=len(cells), figsize=(4, len(cells)+1), layout='tight')
#     for ax, cell in zip(axs, cells):
#         ax.table(
#                 colWidths=colWidths,
#                 cellText=cellText,
#                 loc='center',
#                 edges=cell,
#                 )
#         ax.axis('off')
#
#
# def test_customcell():
#     types = ('horizontal', 'vertical', 'open', 'closed', 'T', 'R', 'B', 'L')
#     codes = (
#         (Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO, Path.MOVETO),
#         (Path.MOVETO, Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO),
#         (Path.MOVETO, Path.MOVETO, Path.MOVETO, Path.MOVETO, Path.MOVETO),
#         (Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY),
#         (Path.MOVETO, Path.MOVETO, Path.MOVETO, Path.LINETO, Path.MOVETO),
#         (Path.MOVETO, Path.MOVETO, Path.LINETO, Path.MOVETO, Path.MOVETO),
#         (Path.MOVETO, Path.LINETO, Path.MOVETO, Path.MOVETO, Path.MOVETO),
#         (Path.MOVETO, Path.MOVETO, Path.MOVETO, Path.MOVETO, Path.LINETO),
#         )
#
#     for t, c in zip(types, codes):
#         cell = CustomCell((0, 0), visible_edges=t, width=1, height=1)
#         code = tuple(s for _, s in cell.get_path().iter_segments())
#         assert c == code
#
#
# @image_comparison(['table_auto_column.png'])
# def test_auto_column():
#     fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
#
#     # iterable list input
#     ax1.axis('off')
#     tb1 = ax1.table(
#         cellText=[['Fit Text', 2],
#                   ['very long long text, Longer text than default', 1]],
#         rowLabels=["A", "B"],
#         colLabels=["Col1", "Col2"],
#         loc="center")
#     tb1.auto_set_font_size(False)
#     tb1.set_fontsize(12)
#     tb1.auto_set_column_width([-1, 0, 1])
#
#     # iterable tuple input
#     ax2.axis('off')
#     tb2 = ax2.table(
#         cellText=[['Fit Text', 2],
#                   ['very long long text, Longer text than default', 1]],
#         rowLabels=["A", "B"],
#         colLabels=["Col1", "Col2"],
#         loc="center")
#     tb2.auto_set_font_size(False)
#     tb2.set_fontsize(12)
#     tb2.auto_set_column_width((-1, 0, 1))
#
#     # 3 single inputs
#     ax3.axis('off')
#     tb3 = ax3.table(
#         cellText=[['Fit Text', 2],
#                   ['very long long text, Longer text than default', 1]],
#         rowLabels=["A", "B"],
#         colLabels=["Col1", "Col2"],
#         loc="center")
#     tb3.auto_set_font_size(False)
#     tb3.set_fontsize(12)
#     tb3.auto_set_column_width(-1)
#     tb3.auto_set_column_width(0)
#     tb3.auto_set_column_width(1)
#
#     # 4 this used to test non-integer iterable input, which did nothing, but only
#     # remains to avoid re-generating the test image.
#     ax4.axis('off')
#     tb4 = ax4.table(
#         cellText=[['Fit Text', 2],
#                   ['very long long text, Longer text than default', 1]],
#         rowLabels=["A", "B"],
#         colLabels=["Col1", "Col2"],
#         loc="center")
#     tb4.auto_set_font_size(False)
#     tb4.set_fontsize(12)
#
#
# def test_table_cells():
#     fig, ax = plt.subplots()
#     table = Table(ax)
#
#     cell = table.add_cell(1, 2, 1, 1)
#     assert isinstance(cell, CustomCell)
#     assert cell is table[1, 2]
#
#     cell2 = CustomCell((0, 0), 1, 2, visible_edges=None)
#     table[2, 1] = cell2
#     assert table[2, 1] is cell2
#
#     # make sure getitem support has not broken
#     # properties and setp
#     table.properties()
#     plt.setp(table)
#
#
# @check_figures_equal()
# def test_table_bbox(fig_test, fig_ref):
#     data = [[2, 3],
#             [4, 5]]
#
#     col_labels = ('Foo', 'Bar')
#     row_labels = ('Ada', 'Bob')
#
#     cell_text = [[f"{x}" for x in row] for row in data]
#
#     ax_list = fig_test.subplots()
#     ax_list.table(cellText=cell_text,
#                   rowLabels=row_labels,
#                   colLabels=col_labels,
#                   loc='center',
#                   bbox=[0.1, 0.2, 0.8, 0.6]
#                   )
#
#     ax_bbox = fig_ref.subplots()
#     ax_bbox.table(cellText=cell_text,
#                   rowLabels=row_labels,
#                   colLabels=col_labels,
#                   loc='center',
#                   bbox=Bbox.from_extents(0.1, 0.2, 0.9, 0.8)
#                   )
#
#
# @check_figures_equal()
# def test_table_unit(fig_test, fig_ref):
#     # test that table doesn't participate in unit machinery, instead uses repr/str
#
#     class FakeUnit:
#         def __init__(self, thing):
#             pass
#         def __repr__(self):
#             return "Hello"
#
#     fake_convertor = munits.ConversionInterface()
#     # v, u, a = value, unit, axis
#     fake_convertor.convert = Mock(side_effect=lambda v, u, a: 0)
#     # not used, here for completeness
#     fake_convertor.default_units = Mock(side_effect=lambda v, a: None)
#     fake_convertor.axisinfo = Mock(side_effect=lambda u, a: munits.AxisInfo())
#
#     munits.registry[FakeUnit] = fake_convertor
#
#     data = [[FakeUnit("yellow"), FakeUnit(42)],
#             [FakeUnit(datetime.datetime(1968, 8, 1)), FakeUnit(True)]]
#
#     fig_test.subplots().table(data)
#     fig_ref.subplots().table([["Hello", "Hello"], ["Hello", "Hello"]])
#     fig_test.canvas.draw()
#     fake_convertor.convert.assert_not_called()
#
#     munits.registry.pop(FakeUnit)
#     assert not munits.registry.get_converter(FakeUnit)
#
#
# def test_table_dataframe(pd):
#     # Test if Pandas Data Frame can be passed in cellText
#
#     data = {
#         'Letter': ['A', 'B', 'C'],
#         'Number': [100, 200, 300]
#     }
#
#     df = pd.DataFrame(data)
#     fig, ax = plt.subplots()
#     table = ax.table(df, loc='center')
#
#     for r, (index, row) in enumerate(df.iterrows()):
#         for c, col in enumerate(df.columns if r == 0 else row.values):
#             assert table[r if r == 0 else r+1, c].get_text().get_text() == str(col)
#
#
# def test_table_fontsize():
#     # Test that the passed fontsize propagates to cells
#     tableData = [['a', 1], ['b', 2]]
#     fig, ax = plt.subplots()
#     test_fontsize = 20
#     t = ax.table(cellText=tableData, loc='top', fontsize=test_fontsize)
#     cell_fontsize = t[(0, 0)].get_fontsize()
#     assert cell_fontsize == test_fontsize, f"Actual:{test_fontsize},got:{cell_fontsize}"
#     cell_fontsize = t[(1, 1)].get_fontsize()
#     assert cell_fontsize == test_fontsize, f"Actual:{test_fontsize},got:{cell_fontsize}"

#########
def test_table_cell_alignment():
    # Test different cell alignment options
    alignments = ['left', 'center', 'right']
    fig, axs = plt.subplots(len(alignments), 1, figsize=(5, 6))
    data = [['Aligned Text', 123]]

    for ax, align in zip(axs, alignments):
        ax.axis('off')
        table = ax.table(cellText=data, loc='center')
        for key in table.get_celld():
            table[key].get_text().set_horizontalalignment(align)



def test_table_scale():
    # Test table scaling by adjusting cell sizes
    fig, axs = plt.subplots(2, 1)
    data = [['A', 'B'], ['C', 'D']]

    axs[0].axis('off')
    table1 = axs[0].table(cellText=data, loc='center')

    axs[1].axis('off')
    table2 = axs[1].table(cellText=data, loc='center')
    # Scale by adjusting height and width of each cell
    for key, cell in table2.get_celld().items():
        cell.set_height(cell.get_height() * 2.0)
        cell.set_width(cell.get_width() * 2.0)


def test_table_cell_colors():
    # Test individual cell coloring
    fig, ax = plt.subplots()
    ax.axis('off')

    cell_colors = [[(.9, .9, .9), (.8, .2, .2)],
                   [(.2, .8, .2), (.2, .2, .8)]]

    ax.table(cellText=[['A', 'B'], ['C', 'D']],
             cellColours=cell_colors,
             loc='center')


def test_table_empty_cells():
    # Test table with empty cells
    fig, ax = plt.subplots()
    ax.axis('off')

    data = [['A', ''], [None, 'D']]
    ax.table(cellText=data, loc='center')


def test_table_line_properties():
    # Test line width and color customization
    fig, ax = plt.subplots()
    ax.axis('off')

    tb = ax.table(cellText=[['A', 'B'], ['C', 'D']], loc='center')

    # Customize line properties of all cells
    for _, cell in tb.get_celld().items():
        cell.set_linewidth(2)
        cell.set_edgecolor('darkblue')

def test_table_mathtext():
    # Test table with LaTeX/MathText
    fig, ax = plt.subplots()
    ax.axis('off')

    math_text = [['$\\alpha$', '$\\beta$'],
                 ['$\\gamma^2$', '$\\frac{x}{y}$']]

    tb = ax.table(cellText=math_text, loc='center')
    tb.auto_set_font_size(False)
    tb.set_fontsize(14)


def test_table_multiline_text():
    # Test multiline text in cells
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')

    cell_text = [['Line 1\nLine 2', 'Single Line'],
                 ['Single Line', 'Line 1\nLine 2\nLine 3']]

    tb = ax.table(cellText=cell_text, loc='center')
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)

    # Adjust cell heights for multiline text
    for (row, col), cell in tb.get_celld().items():
        text = cell.get_text().get_text()
        if '\n' in text:
            lines = text.count('\n') + 1
            cell.set_height(cell.get_height() * lines * 0.7)


def test_table_with_colspans():
    # Test table with column spans
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')

    # Create empty table with specific dimensions
    tb = Table(ax, bbox=[0.1, 0.1, 0.8, 0.8])

    # Add cells with specific spans
    tb.add_cell(0, 0, 1, 2, text='Span Columns 0-1')  # row 0, col 0, height 1, width 2
    tb.add_cell(1, 0, 1, 1, text='Col 0')
    tb.add_cell(1, 1, 1, 1, text='Col 1')
    tb.add_cell(2, 0, 1, 3, text='Span Columns 0-2')  # row 2, col 0, height 1, width 3
    tb.add_cell(1, 2, 1, 1, text='Col 2')

    # Add table to the axis
    ax.add_table(tb)


def test_table_highlight_cells():
    # Test highlighting specific cells
    fig, ax = plt.subplots()
    ax.axis('off')

    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    cell_text = [[str(val) for val in row] for row in data]

    tb = ax.table(cellText=cell_text, loc='center')

    # Highlight cells based on values
    for (row, col), cell in tb.get_celld().items():
        if row > 0 and col < 3:  # Skip header if exists
            value = data[row - 1][col]
            if value > 5:
                cell.set_facecolor('lightcoral')
            elif value < 3:
                cell.set_facecolor('lightblue')


def test_table_highlight_cells():
    # Test highlighting specific cells
    fig, ax = plt.subplots()
    ax.axis('off')

    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    cell_text = [[str(val) for val in row] for row in data]

    tb = ax.table(cellText=cell_text, loc='center')

    # Highlight cells based on values
    for (row, col), cell in tb.get_celld().items():
        if row > 0 and col < 3:  # Skip header if exists
            value = data[row - 1][col]
            if value > 5:
                cell.set_facecolor('lightcoral')
            elif value < 3:
                cell.set_facecolor('lightblue')


def test_table_rotation():
    # Test rotating text within table cells
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')

    data = [['A', 'B'], ['C', 'D']]
    tb = ax.table(cellText=data, loc='center')

    # Apply different rotations to cell text
    rotations = [0, 45, 90, -45]
    idx = 0
    for _, cell in tb.get_celld().items():
        cell.get_text().set_rotation(rotations[idx % len(rotations)])
        idx += 1


def test_table_alpha_transparency():
    # Test cell alpha transparency
    fig, ax = plt.subplots()
    ax.axis('off')

    data = [['A', 'B'], ['C', 'D']]
    tb = ax.table(cellText=data, loc='center')

    # Set different alpha values for cells
    alphas = [0.3, 0.5, 0.7, 1.0]
    idx = 0
    for _, cell in tb.get_celld().items():
        cell.set_alpha(alphas[idx % len(alphas)])
        cell.set_facecolor('skyblue')
        idx += 1


def test_table_with_custom_padding():
    # Test adjusting cell padding
    fig, ax = plt.subplots(3, 1, figsize=(6, 8))

    for i, pad in enumerate([0.01, 0.1, 0.5]):
        ax[i].axis('off')
        tb = ax[i].table(cellText=[['Padding', f'{pad}']],
                         loc='center')

        for _, cell in tb.get_celld().items():
            cell.PAD = pad  # Adjust cell padding


def test_table_add_remove_cells():
    # Test adding and removing cells
    fig, ax = plt.subplots()
    ax.axis('off')

    tb = Table(ax, bbox=[0.1, 0.1, 0.8, 0.8])

    # Add cells
    tb.add_cell(0, 0, 1, 1, text='A')
    tb.add_cell(0, 1, 1, 1, text='B')
    tb.add_cell(1, 0, 1, 1, text='C')
    tb.add_cell(1, 1, 1, 1, text='D')

    # Remove a cell and replace it
    tb._cells.pop((1, 0))  # Remove cell C
    tb.add_cell(1, 0, 1, 1, text='X')  # Replace with X

    ax.add_table(tb)


def test_table_with_multiple_fonts():
    # Test using different fonts for different cells
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')

    data = [['Default', 'Bold'], ['Italic', 'Sans']]
    tb = ax.table(cellText=data, loc='center')

    tb[0, 0].get_text().set_fontfamily('serif')
    tb[0, 1].get_text().set_fontweight('bold')
    tb[1, 0].get_text().set_style('italic')
    tb[1, 1].get_text().set_fontfamily('sans-serif')


def test_table_row_column_labels_styling():
    # Test styling of row and column labels
    fig, ax = plt.subplots()
    ax.axis('off')

    data = [['A', 'B'], ['C', 'D']]
    row_labels = ['Row 1', 'Row 2']
    col_labels = ['Col 1', 'Col 2']

    tb = ax.table(cellText=data,
                  rowLabels=row_labels,
                  colLabels=col_labels,
                  loc='center')

    # Style row labels
    for i in range(len(row_labels)):
        cell = tb[i + 1, -1]  # Row labels are in column -1
        cell.set_facecolor('lightgray')
        cell.get_text().set_fontweight('bold')

    # Style column labels
    for j in range(len(col_labels)):
        cell = tb[0, j]  # Column labels are in row 0
        cell.set_facecolor('lightgray')
        cell.get_text().set_fontweight('bold')


def test_table_with_cell_borders():
    # Test customizing individual cell borders
    fig, ax = plt.subplots()
    ax.axis('off')

    data = [['A', 'B'], ['C', 'D']]
    tb = ax.table(cellText=data, loc='center')

    # Customize borders for specific cells
    styles = [('solid', 2), ('dashed', 1), ('dotted', 1), ('dashdot', 2)]
    idx = 0

    for pos, cell in tb.get_celld().items():
        style, width = styles[idx % len(styles)]
        cell.set_linestyle(style)
        cell.set_linewidth(width)
        idx += 1


def test_table_with_auto_labels():
    # Test table with automatically generated labels
    fig, ax = plt.subplots()
    ax.axis('off')

    # Generate data matrix
    data = np.arange(1, 17).reshape(4, 4)
    cell_text = [[str(val) for val in row] for row in data]

    # Generate automatic row/column labels
    row_labels = [f'Row {i + 1}' for i in range(data.shape[0])]
    col_labels = [f'Col {j + 1}' for j in range(data.shape[1])]

    ax.table(cellText=cell_text,
             rowLabels=row_labels,
             colLabels=col_labels,
             loc='center')
"""Catch all for categorical functions"""
import warnings

import pytest
import numpy as np

import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal


# class TestUnitData:
#     test_cases = [('single', (["hello world"], [0])),
#                   ('unicode', (["Здравствуйте мир"], [0])),
#                   ('mixed', (['A', "np.nan", 'B', "3.14", "мир"],
#                              [0, 1, 2, 3, 4]))]
#     ids, data = zip(*test_cases)
#
#     @pytest.mark.parametrize("data, locs", data, ids=ids)
#     def test_unit(self, data, locs):
#         unit = cat.UnitData(data)
#         assert list(unit._mapping.keys()) == data
#         assert list(unit._mapping.values()) == locs
#
#     def test_update(self):
#         data = ['a', 'd']
#         locs = [0, 1]
#
#         data_update = ['b', 'd', 'e']
#         unique_data = ['a', 'd', 'b', 'e']
#         updated_locs = [0, 1, 2, 3]
#
#         unit = cat.UnitData(data)
#         assert list(unit._mapping.keys()) == data
#         assert list(unit._mapping.values()) == locs
#
#         unit.update(data_update)
#         assert list(unit._mapping.keys()) == unique_data
#         assert list(unit._mapping.values()) == updated_locs
#
#     failing_test_cases = [("number", 3.14), ("nan", np.nan),
#                           ("list", [3.14, 12]), ("mixed type", ["A", 2])]
#
#     fids, fdata = zip(*test_cases)
#
#     @pytest.mark.parametrize("fdata", fdata, ids=fids)
#     def test_non_string_fails(self, fdata):
#         with pytest.raises(TypeError):
#             cat.UnitData(fdata)
#
#     @pytest.mark.parametrize("fdata", fdata, ids=fids)
#     def test_non_string_update_fails(self, fdata):
#         unitdata = cat.UnitData()
#         with pytest.raises(TypeError):
#             unitdata.update(fdata)
#
#
# class FakeAxis:
#     def __init__(self, units):
#         self.units = units
#
#
# class TestStrCategoryConverter:
#     """
#     Based on the pandas conversion and factorization tests:
#
#     ref: /pandas/tseries/tests/test_converter.py
#          /pandas/tests/test_algos.py:TestFactorize
#     """
#     test_cases = [("unicode", ["Здравствуйте мир"]),
#                   ("ascii", ["hello world"]),
#                   ("single", ['a', 'b', 'c']),
#                   ("integer string", ["1", "2"]),
#                   ("single + values>10", ["A", "B", "C", "D", "E", "F", "G",
#                                           "H", "I", "J", "K", "L", "M", "N",
#                                           "O", "P", "Q", "R", "S", "T", "U",
#                                           "V", "W", "X", "Y", "Z"])]
#
#     ids, values = zip(*test_cases)
#
#     failing_test_cases = [("mixed", [3.14, 'A', np.inf]),
#                           ("string integer", ['42', 42])]
#
#     fids, fvalues = zip(*failing_test_cases)
#
#     @pytest.fixture(autouse=True)
#     def mock_axis(self, request):
#         self.cc = cat.StrCategoryConverter()
#         # self.unit should be probably be replaced with real mock unit
#         self.unit = cat.UnitData()
#         self.ax = FakeAxis(self.unit)
#
#     @pytest.mark.parametrize("vals", values, ids=ids)
#     def test_convert(self, vals):
#         np.testing.assert_allclose(self.cc.convert(vals, self.ax.units,
#                                                    self.ax),
#                                    range(len(vals)))
#
#     @pytest.mark.parametrize("value", ["hi", "мир"], ids=["ascii", "unicode"])
#     def test_convert_one_string(self, value):
#         assert self.cc.convert(value, self.unit, self.ax) == 0
#
#     @pytest.mark.parametrize("fvals", fvalues, ids=fids)
#     def test_convert_fail(self, fvals):
#         with pytest.raises(TypeError):
#             self.cc.convert(fvals, self.unit, self.ax)
#
#     def test_axisinfo(self):
#         axis = self.cc.axisinfo(self.unit, self.ax)
#         assert isinstance(axis.majloc, cat.StrCategoryLocator)
#         assert isinstance(axis.majfmt, cat.StrCategoryFormatter)
#
#     def test_default_units(self):
#         assert isinstance(self.cc.default_units(["a"], self.ax), cat.UnitData)
#
#
# PLOT_LIST = [Axes.scatter, Axes.plot, Axes.bar]
# PLOT_IDS = ["scatter", "plot", "bar"]
#
#
# class TestStrCategoryLocator:
#     def test_StrCategoryLocator(self):
#         locs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#         unit = cat.UnitData([str(j) for j in locs])
#         ticks = cat.StrCategoryLocator(unit._mapping)
#         np.testing.assert_array_equal(ticks.tick_values(None, None), locs)
#
#     @pytest.mark.parametrize("plotter", PLOT_LIST, ids=PLOT_IDS)
#     def test_StrCategoryLocatorPlot(self, plotter):
#         ax = plt.figure().subplots()
#         plotter(ax, [1, 2, 3], ["a", "b", "c"])
#         np.testing.assert_array_equal(ax.yaxis.major.locator(), range(3))
#
#
# class TestStrCategoryFormatter:
#     test_cases = [("ascii", ["hello", "world", "hi"]),
#                   ("unicode", ["Здравствуйте", "привет"])]
#
#     ids, cases = zip(*test_cases)
#
#     @pytest.mark.parametrize("ydata", cases, ids=ids)
#     def test_StrCategoryFormatter(self, ydata):
#         unit = cat.UnitData(ydata)
#         labels = cat.StrCategoryFormatter(unit._mapping)
#         for i, d in enumerate(ydata):
#             assert labels(i, i) == d
#             assert labels(i, None) == d
#
#     @pytest.mark.parametrize("ydata", cases, ids=ids)
#     @pytest.mark.parametrize("plotter", PLOT_LIST, ids=PLOT_IDS)
#     def test_StrCategoryFormatterPlot(self, ydata, plotter):
#         ax = plt.figure().subplots()
#         plotter(ax, range(len(ydata)), ydata)
#         for i, d in enumerate(ydata):
#             assert ax.yaxis.major.formatter(i) == d
#         assert ax.yaxis.major.formatter(i+1) == ""
#
#
# def axis_test(axis, labels):
#     ticks = list(range(len(labels)))
#     np.testing.assert_array_equal(axis.get_majorticklocs(), ticks)
#     graph_labels = [axis.major.formatter(i, i) for i in ticks]
#     # _text also decodes bytes as utf-8.
#     assert graph_labels == [cat.StrCategoryFormatter._text(l) for l in labels]
#     assert list(axis.units._mapping.keys()) == [l for l in labels]
#     assert list(axis.units._mapping.values()) == ticks
#
#
# class TestPlotBytes:
#     bytes_cases = [('string list', ['a', 'b', 'c']),
#                    ('bytes list', [b'a', b'b', b'c']),
#                    ('bytes ndarray', np.array([b'a', b'b', b'c']))]
#
#     bytes_ids, bytes_data = zip(*bytes_cases)
#
#     @pytest.mark.parametrize("plotter", PLOT_LIST, ids=PLOT_IDS)
#     @pytest.mark.parametrize("bdata", bytes_data, ids=bytes_ids)
#     def test_plot_bytes(self, plotter, bdata):
#         ax = plt.figure().subplots()
#         counts = np.array([4, 6, 5])
#         plotter(ax, bdata, counts)
#         axis_test(ax.xaxis, bdata)
#
#
# class TestPlotNumlike:
#     numlike_cases = [('string list', ['1', '11', '3']),
#                      ('string ndarray', np.array(['1', '11', '3'])),
#                      ('bytes list', [b'1', b'11', b'3']),
#                      ('bytes ndarray', np.array([b'1', b'11', b'3']))]
#     numlike_ids, numlike_data = zip(*numlike_cases)
#
#     @pytest.mark.parametrize("plotter", PLOT_LIST, ids=PLOT_IDS)
#     @pytest.mark.parametrize("ndata", numlike_data, ids=numlike_ids)
#     def test_plot_numlike(self, plotter, ndata):
#         ax = plt.figure().subplots()
#         counts = np.array([4, 6, 5])
#         plotter(ax, ndata, counts)
#         axis_test(ax.xaxis, ndata)
#
#
# class TestPlotTypes:
#     @pytest.mark.parametrize("plotter", PLOT_LIST, ids=PLOT_IDS)
#     def test_plot_unicode(self, plotter):
#         ax = plt.figure().subplots()
#         words = ['Здравствуйте', 'привет']
#         plotter(ax, words, [0, 1])
#         axis_test(ax.xaxis, words)
#
#     @pytest.fixture
#     def test_data(self):
#         self.x = ["hello", "happy", "world"]
#         self.xy = [2, 6, 3]
#         self.y = ["Python", "is", "fun"]
#         self.yx = [3, 4, 5]
#
#     @pytest.mark.usefixtures("test_data")
#     @pytest.mark.parametrize("plotter", PLOT_LIST, ids=PLOT_IDS)
#     def test_plot_xaxis(self, test_data, plotter):
#         ax = plt.figure().subplots()
#         plotter(ax, self.x, self.xy)
#         axis_test(ax.xaxis, self.x)
#
#     @pytest.mark.usefixtures("test_data")
#     @pytest.mark.parametrize("plotter", PLOT_LIST, ids=PLOT_IDS)
#     def test_plot_yaxis(self, test_data, plotter):
#         ax = plt.figure().subplots()
#         plotter(ax, self.yx, self.y)
#         axis_test(ax.yaxis, self.y)
#
#     @pytest.mark.usefixtures("test_data")
#     @pytest.mark.parametrize("plotter", PLOT_LIST, ids=PLOT_IDS)
#     def test_plot_xyaxis(self, test_data, plotter):
#         ax = plt.figure().subplots()
#         plotter(ax, self.x, self.y)
#         axis_test(ax.xaxis, self.x)
#         axis_test(ax.yaxis, self.y)
#
#     @pytest.mark.parametrize("plotter", PLOT_LIST, ids=PLOT_IDS)
#     def test_update_plot(self, plotter):
#         ax = plt.figure().subplots()
#         plotter(ax, ['a', 'b'], ['e', 'g'])
#         plotter(ax, ['a', 'b', 'd'], ['f', 'a', 'b'])
#         plotter(ax, ['b', 'c', 'd'], ['g', 'e', 'd'])
#         axis_test(ax.xaxis, ['a', 'b', 'd', 'c'])
#         axis_test(ax.yaxis, ['e', 'g', 'f', 'a', 'b', 'd'])
#
#     def test_update_plot_heterogenous_plotter(self):
#         ax = plt.figure().subplots()
#         ax.scatter(['a', 'b'], ['e', 'g'])
#         ax.plot(['a', 'b', 'd'], ['f', 'a', 'b'])
#         ax.bar(['b', 'c', 'd'], ['g', 'e', 'd'])
#         axis_test(ax.xaxis, ['a', 'b', 'd', 'c'])
#         axis_test(ax.yaxis, ['e', 'g', 'f', 'a', 'b', 'd'])
#
#     failing_test_cases = [("mixed", ['A', 3.14]),
#                           ("number integer", ['1', 1]),
#                           ("string integer", ['42', 42]),
#                           ("missing", ['12', np.nan])]
#
#     fids, fvalues = zip(*failing_test_cases)
#
#     plotters = [Axes.scatter, Axes.bar,
#                 pytest.param(Axes.plot, marks=pytest.mark.xfail)]
#
#     @pytest.mark.parametrize("plotter", plotters)
#     @pytest.mark.parametrize("xdata", fvalues, ids=fids)
#     def test_mixed_type_exception(self, plotter, xdata):
#         ax = plt.figure().subplots()
#         with pytest.raises(TypeError):
#             plotter(ax, xdata, [1, 2])
#
#     @pytest.mark.parametrize("plotter", plotters)
#     @pytest.mark.parametrize("xdata", fvalues, ids=fids)
#     def test_mixed_type_update_exception(self, plotter, xdata):
#         ax = plt.figure().subplots()
#         with pytest.raises(TypeError):
#             plotter(ax, [0, 3], [1, 3])
#             plotter(ax, xdata, [1, 2])
#
#
# @mpl.style.context('default')
# @check_figures_equal()
# def test_overriding_units_in_plot(fig_test, fig_ref):
#     from datetime import datetime
#
#     t0 = datetime(2018, 3, 1)
#     t1 = datetime(2018, 3, 2)
#     t2 = datetime(2018, 3, 3)
#     t3 = datetime(2018, 3, 4)
#
#     ax_test = fig_test.subplots()
#     ax_ref = fig_ref.subplots()
#     for ax, kwargs in zip([ax_test, ax_ref],
#                           ({}, dict(xunits=None, yunits=None))):
#         # First call works
#         ax.plot([t0, t1], ["V1", "V2"], **kwargs)
#         x_units = ax.xaxis.units
#         y_units = ax.yaxis.units
#         # this should not raise
#         ax.plot([t2, t3], ["V1", "V2"], **kwargs)
#         # assert that we have not re-set the units attribute at all
#         assert x_units is ax.xaxis.units
#         assert y_units is ax.yaxis.units
#
#
# def test_no_deprecation_on_empty_data():
#     """
#     Smoke test to check that no deprecation warning is emitted. See #22640.
#     """
#     f, ax = plt.subplots()
#     ax.xaxis.update_units(["a", "b"])
#     ax.plot([], [])
#
#
# def test_hist():
#     fig, ax = plt.subplots()
#     n, bins, patches = ax.hist(['a', 'b', 'a', 'c', 'ff'])
#     assert n.shape == (10,)
#     np.testing.assert_allclose(n, [2., 0., 0., 1., 0., 0., 1., 0., 0., 1.])
#
#
# def test_set_lim():
#     # Numpy 1.25 deprecated casting [2.] to float, catch_warnings added to error
#     # with numpy 1.25 and prior to the change from gh-26597
#     # can be removed once the minimum numpy version has expired the warning
#     f, ax = plt.subplots()
#     ax.plot(["a", "b", "c", "d"], [1, 2, 3, 4])
#     with warnings.catch_warnings():
#         ax.set_xlim("b", "c")


class TestEmptyAndEdgeCases:
    def test_empty_data_handling(self):
        # Test how UnitData handles empty lists
        unit = cat.UnitData([])
        assert len(unit._mapping) == 0

        # Test updating with data after empty initialization
        unit.update(['a', 'b', 'c'])
        assert list(unit._mapping.keys()) == ['a', 'b', 'c']
        assert list(unit._mapping.values()) == [0, 1, 2]

    def test_duplicate_categories(self):
        # Test how duplicates are handled
        unit = cat.UnitData(['a', 'b', 'a', 'c', 'b'])
        assert list(unit._mapping.keys()) == ['a', 'b', 'c']
        assert list(unit._mapping.values()) == [0, 1, 2]

    def test_very_long_category_names(self):
        # Test with very long category names
        long_name = "x" * 1000
        unit = cat.UnitData([long_name, 'short'])

        ax = plt.figure().subplots()
        ax.plot([long_name, 'short'], [1, 2])

        # Verify formatter correctly handles long names
        formatter = cat.StrCategoryFormatter(unit._mapping)
        assert formatter(0, 0) == long_name


class TestCustomSorting:
    def test_custom_order(self):
        # Test plotting with custom category order
        ax = plt.figure().subplots()
        categories = ['medium', 'small', 'large']
        values = [5, 3, 8]

        ax.bar(categories, values)

        # Default order should be as provided
        np.testing.assert_array_equal(ax.xaxis.get_majorticklocs(), [0, 1, 2])
        assert [ax.xaxis.major.formatter(i, i) for i in range(3)] == categories

        # Now sort them in desired order
        sort_order = ['small', 'medium', 'large']
        # Here we'd need to use a custom categorical class or functionality
        # This is just checking the current behavior
        assert list(ax.xaxis.units._mapping.keys()) == categories


class TestSpecialCharacters:
    def test_extended_ascii(self):
        # Test with extended ASCII characters
        chars = ['é', 'ñ', 'ü', 'ç']
        ax = plt.figure().subplots()
        ax.bar(chars, [1, 2, 3, 4])

        # Verify correct mapping
        np.testing.assert_array_equal(ax.xaxis.get_majorticklocs(), [0, 1, 2, 3])
        assert [ax.xaxis.major.formatter(i, i) for i in range(4)] == chars

    def test_emoji_categories(self):
        # Test with emoji characters as categories
        emojis = ['😀', '🚀', '🐍', '📊']
        ax = plt.figure().subplots()
        ax.bar(emojis, [1, 2, 3, 4])

        # Verify correct mapping
        np.testing.assert_array_equal(ax.xaxis.get_majorticklocs(), [0, 1, 2, 3])
        assert [ax.xaxis.major.formatter(i, i) for i in range(4)] == emojis


@pytest.mark.parametrize("plot_func",
                         [Axes.pie, Axes.boxplot],
                         ids=["pie", "boxplot"])
def test_other_plot_types(plot_func):
    # Test categorical data with other plot types
    fig, ax = plt.subplots()  # Fixed: use plt.subplots() instead of plt.figure().subplots()
    categories = ['A', 'B', 'C']
    values = [3, 1, 2]

    if plot_func == Axes.pie:
        plot_func(ax, values, labels=categories)
        # Check that labels are used correctly
        texts = [t.get_text() for t in ax.texts]
        assert all(cat in texts for cat in categories)

    elif plot_func == Axes.boxplot:
        data = [[v] * (i+1) for i, v in enumerate(values)]
        plot_func(ax, data, tick_labels=categories)
        # Check that labels are set correctly
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == categories

### SET 2
class TestDataSerialization:
    def test_pickling_categorical_mapping(self):
        """Test that categorical mapping can be pickled and unpickled"""
        import pickle

        # Create categorical data and mapping
        categories = ["Category A", "Category B", "Category C"]
        unit_data = cat.UnitData(categories)

        # Pickle and unpickle just the mapping (avoids itertools warning)
        mapping = unit_data._mapping
        pickled_data = pickle.dumps(mapping)
        mapping_unpickled = pickle.loads(pickled_data)

        # Verify data survived serialization
        assert list(mapping_unpickled.keys()) == categories
        assert list(mapping_unpickled.values()) == [0, 1, 2]


class TestCategoricalTransformations:
    def test_log_scale_with_categories(self):
        """Test interaction between log scale and categorical axes"""
        fig, ax = plt.subplots()
        categories = ['A', 'B', 'C']
        values = [1, 10, 100]

        ax.bar(categories, values)
        ax.set_yscale('log')

        # Verify categories are properly set on x-axis
        assert list(ax.xaxis.units._mapping.keys()) == categories

        # Check tick labels match our categories
        tick_labels = [label.get_text() for label in ax.get_xticklabels()]
        assert tick_labels == categories

        # Y-axis should be log-scaled
        assert ax.get_yscale() == 'log'

    def test_categorical_with_polar_projection(self):
        """Test using categorical data with polar projection"""
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        categories = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        values = [4, 3, 2, 1, 5, 6, 7, 8]

        # Convert to radians for polar plot
        thetas = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)

        # Create bars at specific angles
        ax.bar(thetas, values)

        # Set the categorical labels
        ax.set_xticks(thetas)
        ax.set_xticklabels(categories)

        # Check the labels are correctly set
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == categories


class TestMultipleAxesAndSubplots:
    def test_linked_categorical_axes(self):
        """Test sharing categorical axes between subplots"""
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        categories = ['Jan', 'Feb', 'Mar', 'Apr']

        # Plot on both axes
        ax1.plot(categories, [1, 3, 2, 4])
        ax2.plot(categories, [4, 2, 3, 1])

        # Verify both axes share same mapping
        assert ax1.xaxis.units is ax2.xaxis.units
        assert list(ax1.xaxis.units._mapping.keys()) == categories

        # Verify ticks match on both plots
        np.testing.assert_array_equal(ax1.xaxis.get_majorticklocs(),
                                      ax2.xaxis.get_majorticklocs())

    def test_twin_axes_with_categories(self):
        """Test using twiny/twinx with categorical data"""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        # Primary axis with categories
        categories = ['Red', 'Green', 'Blue']
        ax1.bar(categories, [10, 20, 30], color='lightgray')

        # Secondary axis with same categories but different data
        ax2.plot(categories, [5, 15, 10], 'ro-')

        # Both axes should have same x categories
        assert list(ax1.xaxis.units._mapping.keys()) == categories
        assert list(ax2.xaxis.units._mapping.keys()) == categories

        # Should have different y-axis ranges (not comparing units which may not exist)
        assert ax1.get_ylim() != ax2.get_ylim()

        # Verify secondary y-axis has expected data points
        line = ax2.get_lines()[0]
        assert len(line.get_xdata()) == len(categories)
        assert np.allclose(line.get_ydata(), [5, 15, 10])


class TestAdvancedCategoricalFeatures:
    def test_categorical_with_annotations(self):
        """Test annotations with categorical axes"""
        fig, ax = plt.subplots()
        categories = ['Low', 'Medium', 'High']
        values = [3, 7, 2]

        ax.bar(categories, values)

        # Add annotations at specific category positions
        for i, (cat, val) in enumerate(zip(categories, values)):
            ax.annotate(f'{val}',
                        xy=(i, val),
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center')

        # Verify both annotations and categories
        assert len(ax.texts) == 3
        assert [ax.xaxis.major.formatter(i, i) for i in range(3)] == categories

    def test_categorical_colormap(self):
        """Test mapping categories to colors using a colormap"""
        fig, ax = plt.subplots()
        categories = ['Group A', 'Group B', 'Group C', 'Group D']
        values = [3, 7, 2, 5]

        # Create a colormap based on category index
        cmap = plt.cm.viridis
        norm = mpl.colors.Normalize(vmin=0, vmax=len(categories) - 1)
        colors = [cmap(norm(i)) for i in range(len(categories))]

        # Create bars with colors corresponding to categories
        bars = ax.bar(categories, values, color=colors)

        # Check category mapping and color assignment
        assert [ax.xaxis.major.formatter(i, i) for i in range(4)] == categories
        assert len(bars) == len(categories)
        for i, bar in enumerate(bars):
            assert np.allclose(bar.get_facecolor(), colors[i])


class TestCategoricalDataSources:
    def test_categorical_from_dataframe(self):
        """Test using pandas DataFrame as categorical data source"""
        pd = pytest.importorskip("pandas")

        # Create a test DataFrame
        df = pd.DataFrame({
            'category': ['Alpha', 'Beta', 'Gamma', 'Delta'],
            'value1': [10, 15, 7, 12],
            'value2': [5, 3, 8, 4]
        })

        fig, ax = plt.subplots()
        ax.bar(df['category'], df['value1'])

        # Add second series
        ax.plot(df['category'], df['value2'], 'ro-')

        # Check categories are preserved
        assert list(ax.xaxis.units._mapping.keys()) == df['category'].tolist()

    def test_categorical_from_dict(self):
        """Test using dictionaries as categorical data source"""
        data = {
            'A': 10,
            'B': 5,
            'C': 15,
            'D': 7
        }

        fig, ax = plt.subplots()
        ax.bar(list(data.keys()), list(data.values()))

        # Check categories match dictionary keys
        cats = list(data.keys())
        assert [ax.xaxis.major.formatter(i, i) for i in range(len(cats))] == cats


class TestCategoryFormatting:
    def test_custom_category_formatter(self):
        """Test applying custom formatting to category labels"""
        fig, ax = plt.subplots()
        categories = ['a', 'b', 'c', 'd']
        values = [3, 1, 4, 2]

        ax.bar(categories, values)

        # Create a formatter that uppercases labels
        def uppercase_formatter(x, pos):
            if x < len(categories):
                # Get the category at position x
                cat_list = list(ax.xaxis.units._mapping.keys())
                cat_positions = list(ax.xaxis.units._mapping.values())
                # Find the category at position x
                for cat, position in zip(cat_list, cat_positions):
                    if position == x:
                        return cat.upper()
            return ''

        formatter = mpl.ticker.FuncFormatter(uppercase_formatter)

        # Apply the formatter
        ax.xaxis.set_major_formatter(formatter)

        # Get the actual formatted tick labels
        tick_labels = [label.get_text() for label in ax.get_xticklabels()]

        # Verify formatter is applied
        assert tick_labels == ['A', 'B', 'C', 'D']

    def test_rotated_category_labels(self):
        """Test rotating category labels for better readability"""
        fig, ax = plt.subplots()
        long_categories = ['Category A with long name',
                           'Category B with longer name',
                           'Category C with even longer name']

        ax.bar(long_categories, [1, 2, 3])

        # Rotate labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Check rotation was applied
        for label in ax.get_xticklabels():
            assert label.get_rotation() == 45
            assert label.get_ha() == 'right'


class TestCategoryInteractions:
    def test_category_reordering(self):
        """Test reordering categories manually"""
        fig, ax = plt.subplots()
        categories = ['C', 'A', 'D', 'B']  # Initial order
        values = [3, 1, 4, 2]

        ax.bar(categories, values)

        # Get current axis info
        old_order = [ax.xaxis.major.formatter(i, i) for i in range(len(categories))]
        assert old_order == categories

        # Create new axis with desired order
        new_order = sorted(categories)  # Sort alphabetically

        # Create new plot with new order (we can't directly reorder the axis mapping)
        fig2, ax2 = plt.subplots()

        # Map values to their categories
        category_to_value = dict(zip(categories, values))
        reordered_values = [category_to_value[cat] for cat in new_order]

        ax2.bar(new_order, reordered_values)

        # Verify new order
        assert [ax2.xaxis.major.formatter(i, i) for i in range(len(new_order))] == new_order
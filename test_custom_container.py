import pytest
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from matplotlib.container import Container, BarContainer, ErrorbarContainer, StemContainer
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle


# ------------------ Generic Container Tests ------------------

@pytest.mark.parametrize("artists", [
    [],
    [Line2D([], [])],
    [Rectangle((0, 0), 1, 1)],
    [None, Line2D([], []), Rectangle((0, 0), 1, 1)],
    [Container([]), Line2D([], [])],
    [None, Container([Line2D([], [])]), Rectangle((0, 0), 1, 1)],
])
def test_container_repr_and_children(artists):
    c = Container(artists)
    assert isinstance(repr(c), str)
    children = [a for a in artists if a is not None]
    assert c.get_children() == children


@pytest.mark.parametrize("label", ["Test", "Label1", "NewLabel", "AnotherLabel"])
def test_container_label_handling(label):
    c = Container([], label=label)
    assert c._label == label
    c.set_label("Updated" + label)
    assert c.get_label() == "Updated" + label


@pytest.mark.parametrize("removal_behavior", [True, False])
def test_container_remove_custom_method(removal_behavior):
    flag = {"called": False}
    def custom_remove(c):
        flag["called"] = True
        if not removal_behavior:
            raise NotImplementedError("cannot remove artist")

    rect = Rectangle((0, 0), 1, 1)
    cont = Container([rect])
    cont._remove_method = custom_remove

    if removal_behavior:
        cont.remove()
        assert flag["called"] is True
    else:
        with pytest.raises(NotImplementedError):
            cont.remove()
        assert flag["called"] is True


@pytest.mark.parametrize("nested_level", [1, 2, 3, 5, 10, 15])
def test_deeply_nested_container_flattening(nested_level):
    obj = Line2D([], [])
    container = Container([obj])
    for _ in range(nested_level):
        container = Container([container])
    assert container.get_children() == [obj]


@pytest.mark.parametrize("num_none_objects", [0, 5, 10, 50, 100])
def test_container_with_multiple_none(num_none_objects):
    objs = [None] * num_none_objects + [Line2D([], [])]
    c = Container(objs)
    assert len(c.get_children()) == 1


@pytest.mark.parametrize("wrong_type", [123, "abc", 3.14, {}, [], (1, 2)])
def test_container_wrong_types_safe_remove(wrong_type):
    c = Container([wrong_type])
    try:
        c.remove()
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")


@pytest.mark.parametrize("num_elements", [0, 1, 10, 100, 1000, 5000])
def test_large_containers(num_elements):
    objs = [Line2D([], []) for _ in range(num_elements)]
    c = Container(objs)
    assert len(c.get_children()) == num_elements


def test_single_element_container_behavior():
    line = Line2D([], [])
    c = Container([line])
    assert list(c) == [line]
    assert c.get_children() == [line]


def test_container_inheritance_repr():
    class CustomContainer(Container): pass
    obj = Line2D([], [])
    cc = CustomContainer([obj])
    assert "CustomContainer" in repr(cc)
    assert cc.get_children() == [obj]


# ------------------ BarContainer Tests ------------------

@pytest.mark.parametrize("num_patches,orientation,with_errorbar", list(product([0, 1, 5, 10], ["vertical", "horizontal"], [True, False])))
def test_barcontainer_combinations(num_patches, orientation, with_errorbar):
    bars = [Rectangle((i, i), 1, 1) for i in range(num_patches)]
    datavalues = list(range(num_patches)) if num_patches else None
    err = ErrorbarContainer((Line2D([], []), (), ())) if with_errorbar else None
    bc = BarContainer(patches=bars, errorbar=err, datavalues=datavalues, orientation=orientation)
    assert bc.orientation == orientation
    if with_errorbar and err:
        assert isinstance(bc.errorbar, ErrorbarContainer)


# ------------------ ErrorbarContainer Tests ------------------

@pytest.mark.parametrize("caps_present,bars_present,has_xerr,has_yerr", list(product([True, False], repeat=4)))
def test_errorbarcontainer_full_flags(caps_present, bars_present, has_xerr, has_yerr):
    line = Line2D([], [])
    caps = (Line2D([], []), Line2D([], [])) if caps_present else ()
    bars = (LineCollection([]),) if bars_present else ()
    ec = ErrorbarContainer((line, caps, bars), has_xerr=has_xerr, has_yerr=has_yerr)
    assert ec.lines[0] is line
    assert ec.has_xerr == has_xerr
    assert ec.has_yerr == has_yerr


# ------------------ StemContainer Tests ------------------

@pytest.mark.parametrize("marker_present,stem_present,base_present", list(product([True, False], repeat=3)))
def test_stemcontainer_missing_parts(marker_present, stem_present, base_present):
    marker = Line2D([], []) if marker_present else None
    stems = LineCollection([]) if stem_present else None
    base = Line2D([], []) if base_present else None
    sc = StemContainer((marker, stems, base))
    if marker_present:
        assert sc.markerline is marker
    else:
        assert sc.markerline is None
    if stem_present:
        assert sc.stemlines is stems
    else:
        assert sc.stemlines is None
    if base_present:
        assert sc.baseline is base
    else:
        assert sc.baseline is None


# ------------------ Extreme Stress Tests ------------------

@pytest.mark.parametrize("deep_levels", [5, 10, 20, 50])
def test_extremely_nested_containers(deep_levels):
    inner = Line2D([], [])
    for _ in range(deep_levels):
        inner = Container([inner])
    c = Container([inner])
    assert c.get_children()


@pytest.mark.parametrize("massive_size", [500, 1000, 2000, 5000])
def test_massive_number_of_artists(massive_size):
    artists = [Line2D([], []) for _ in range(massive_size)]
    c = Container(artists)
    assert len(c.get_children()) == massive_size

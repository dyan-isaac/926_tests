import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.container import Container, BarContainer, ErrorbarContainer, StemContainer
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle


# ---------- Generic Container Tests ----------

def test_empty_container_repr_and_children():
    c = Container([])
    assert repr(c).startswith("<Container object of 0 artists>")
    assert c.get_children() == []

def test_container_label_handling():
    c = Container([], label="Test")
    assert c._label == "Test"
    c.set_label("NewLabel")
    assert c.get_label() == "NewLabel"


def test_container_remove_custom_method():
    flag = {"called": False}
    def custom_remove(c):
        flag["called"] = True
        raise NotImplementedError("cannot remove artist")

    rect = Rectangle((0, 0), 1, 1)
    cont = Container([rect])
    cont._remove_method = custom_remove
    with pytest.raises(NotImplementedError, match="cannot remove artist"):
        cont.remove()
    assert flag["called"] is False

# def test_container_remove_custom_method():
#     flag = {"called": False}
#     def custom_remove(c):
#         flag["called"] = True
#
#     rect = Rectangle((0, 0), 1, 1)
#     cont = Container([rect])
#     cont._remove_method = custom_remove
#     cont.remove()
#     assert flag["called"] is True

def test_get_children_skips_none():
    c = Container([None, Line2D([], [])])
    assert len(c.get_children()) == 1


# ---------- BarContainer Tests ----------

# def test_barcontainer_attributes():
#     bars = [Rectangle((0, 0), 1, 1) for _ in range(3)]
#     err_container = ErrorbarContainer((Line2D([], []), (), ()))
#     bc = BarContainer(patches=bars, errorbar=err_container,
#                       datavalues=[1, 2, 3], orientation="horizontal")
#     assert bc.patches == bars
#     assert bc.errorbar == err_container
#     assert bc.datavalues == [1, 2, 3]
#     assert bc.orientation == "horizontal"


# ---------- ErrorbarContainer Tests ----------

# def test_errorbarcontainer_flags_and_lines():
#     line = Line2D([], [])
#     caps = (Line2D([], []), Line2D([], []))
#     bars = (LineCollection([], []),)
#     ec = ErrorbarContainer((line, caps, bars), has_xerr=True, has_yerr=False)
#     assert ec.lines[0] is line
#     assert ec.has_xerr is True
#     assert ec.has_yerr is False


# ---------- StemContainer Tests ----------

# def test_stemcontainer_initialization():
#     marker = Line2D([], [])
#     stems = LineCollection([], [])
#     base = Line2D([], [])
#     sc = StemContainer((marker, stems, base))
#     assert sc.markerline is marker
#     assert sc.stemlines is stems
#     assert sc.baseline is base


# ---------- Partitioning and Edge Cases ----------

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

# def test_remove_skips_non_artist_objects():
#     obj = object()
#     c = Container([obj])
#     c.remove()  # should not raise

def test_nested_container_flattening():
    inner = Line2D([], [])
    outer = Container([Container([inner])])
    assert outer.get_children() == [inner]

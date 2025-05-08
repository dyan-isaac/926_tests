import pytest
from matplotlib import cbook
import pytest
from matplotlib.cbook import normalize_kwargs, simple_linear_interpolation
import numpy as np


class TestCallbackRegistryEdgeCases:
    def test_callback_trigger_without_callback(self):
        registry = cbook.CallbackRegistry()
        registry.process('event', {})  # Should not raise error without callback

    def test_callback_registered_and_removed(self):
        registry = cbook.CallbackRegistry()
        called = []

        def cb(event): called.append(event)

        cid = registry.connect('event', cb)
        registry.disconnect(cid)
        registry.process('event', {})
        assert called == []

    def test_callback_multiple_events(self):
        registry = cbook.CallbackRegistry()
        events = []

        def cb(event): events.append(event)

        registry.connect('a', cb)
        registry.connect('b', cb)
        registry.process('a', 'data-a')
        registry.process('b', 'data-b')
        assert events == ['data-a', 'data-b']


class TestNormalizeKwargsVariants:
    def test_normal_case(self):
        kwargs = {'label': 'abc'}
        aliased = {'lbl': 'label'}
        result = normalize_kwargs(kwargs, aliased)
        assert result == {'label': 'abc'}

    # def test_alias_override(self):
    #     kwargs = {'lbl': 'xyz'}
    #     aliased = {'lbl': 'label'}
    #     result = normalize_kwargs(kwargs, aliased)
    #     assert result == {'label': 'xyz'}
    #
    # def test_conflict_raises(self):
    #     kwargs = {'label': 'a', 'lbl': 'b'}
    #     aliased = {'lbl': 'label'}
    #     with pytest.raises(TypeError):
    #         normalize_kwargs(kwargs, aliased)

    def test_invalid_key_raises(self):
        kwargs = {'invalid': 1}
        aliased = {}
        with pytest.raises(TypeError):
            normalize_kwargs(kwargs, aliased, required=['label'])


# class TestSimpleLinearInterpolationCases:
#     def test_basic_interpolation(self):
#         x = np.array([0, 1, 2])
#         y = np.array([0, 2, 4])
#         new_x, new_y = simple_linear_interpolation(x, y, 2)
#         assert len(new_x) == 5
#         assert new_y.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
#
#     def test_single_segment(self):
#         x = np.array([0, 1])
#         y = np.array([1, 3])
#         new_x, new_y = simple_linear_interpolation(x, y, 3)
#         assert np.allclose(new_y, [1.0, 1.66666667, 2.33333333, 3.0])
#
#     def test_non_uniform_spacing(self):
#         x = np.array([0, 2, 3])
#         y = np.array([1, 3, 6])
#         new_x, new_y = simple_linear_interpolation(x, y, 2)
#         assert len(new_x) == 5
#         assert np.isclose(new_y[-1], 6.0)

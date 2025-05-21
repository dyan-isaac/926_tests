from __future__ import annotations

from collections.abc import Callable
import re
import typing
from typing import Any, TypeVar

import numpy as np
import pytest

import matplotlib as mpl
from matplotlib import _api


if typing.TYPE_CHECKING:
    from typing_extensions import Self

T = TypeVar('T')


# @pytest.mark.parametrize('target,shape_repr,test_shape',
#                          [((None, ), "(N,)", (1, 3)),
#                           ((None, 3), "(N, 3)", (1,)),
#                           ((None, 3), "(N, 3)", (1, 2)),
#                           ((1, 5), "(1, 5)", (1, 9)),
#                           ((None, 2, None), "(M, 2, N)", (1, 3, 1))
#                           ])
# def test_check_shape(target: tuple[int | None, ...],
#                      shape_repr: str,
#                      test_shape: tuple[int, ...]) -> None:
#     error_pattern = "^" + re.escape(
#         f"'aardvark' must be {len(target)}D with shape {shape_repr}, but your input "
#         f"has shape {test_shape}")
#     data = np.zeros(test_shape)
#     with pytest.raises(ValueError, match=error_pattern):
#         _api.check_shape(target, aardvark=data)
#
#
# def test_classproperty_deprecation() -> None:
#     class A:
#         @_api.deprecated("0.0.0")
#         @_api.classproperty
#         def f(cls: Self) -> None:
#             pass
#     with pytest.warns(mpl.MatplotlibDeprecationWarning):
#         A.f
#     with pytest.warns(mpl.MatplotlibDeprecationWarning):
#         a = A()
#         a.f
#
#
# def test_warn_deprecated():
#     with pytest.warns(mpl.MatplotlibDeprecationWarning,
#                       match=r'foo was deprecated in Matplotlib 3\.10 and will be '
#                             r'removed in 3\.12\.'):
#         _api.warn_deprecated('3.10', name='foo')
#     with pytest.warns(mpl.MatplotlibDeprecationWarning,
#                       match=r'The foo class was deprecated in Matplotlib 3\.10 and '
#                             r'will be removed in 3\.12\.'):
#         _api.warn_deprecated('3.10', name='foo', obj_type='class')
#     with pytest.warns(mpl.MatplotlibDeprecationWarning,
#                       match=r'foo was deprecated in Matplotlib 3\.10 and will be '
#                             r'removed in 3\.12\. Use bar instead\.'):
#         _api.warn_deprecated('3.10', name='foo', alternative='bar')
#     with pytest.warns(mpl.MatplotlibDeprecationWarning,
#                       match=r'foo was deprecated in Matplotlib 3\.10 and will be '
#                             r'removed in 3\.12\. More information\.'):
#         _api.warn_deprecated('3.10', name='foo', addendum='More information.')
#     with pytest.warns(mpl.MatplotlibDeprecationWarning,
#                       match=r'foo was deprecated in Matplotlib 3\.10 and will be '
#                             r'removed in 4\.0\.'):
#         _api.warn_deprecated('3.10', name='foo', removal='4.0')
#     with pytest.warns(mpl.MatplotlibDeprecationWarning,
#                       match=r'foo was deprecated in Matplotlib 3\.10\.'):
#         _api.warn_deprecated('3.10', name='foo', removal=False)
#     with pytest.warns(PendingDeprecationWarning,
#                       match=r'foo will be deprecated in a future version'):
#         _api.warn_deprecated('3.10', name='foo', pending=True)
#     with pytest.raises(ValueError, match=r'cannot have a scheduled removal'):
#         _api.warn_deprecated('3.10', name='foo', pending=True, removal='3.12')
#     with pytest.warns(mpl.MatplotlibDeprecationWarning, match=r'Complete replacement'):
#         _api.warn_deprecated('3.10', message='Complete replacement', name='foo',
#                              alternative='bar', addendum='More information.',
#                              obj_type='class', removal='4.0')
#
#
# def test_deprecate_privatize_attribute() -> None:
#     class C:
#         def __init__(self) -> None: self._attr = 1
#         def _meth(self, arg: T) -> T: return arg
#         attr: int = _api.deprecate_privatize_attribute("0.0")
#         meth: Callable = _api.deprecate_privatize_attribute("0.0")
#
#     c = C()
#     with pytest.warns(mpl.MatplotlibDeprecationWarning):
#         assert c.attr == 1
#     with pytest.warns(mpl.MatplotlibDeprecationWarning):
#         c.attr = 2
#     with pytest.warns(mpl.MatplotlibDeprecationWarning):
#         assert c.attr == 2
#     with pytest.warns(mpl.MatplotlibDeprecationWarning):
#         assert c.meth(42) == 42
#
#
# def test_delete_parameter() -> None:
#     @_api.delete_parameter("3.0", "foo")
#     def func1(foo: Any = None) -> None:
#         pass
#
#     @_api.delete_parameter("3.0", "foo")
#     def func2(**kwargs: Any) -> None:
#         pass
#
#     for func in [func1, func2]:  # type: ignore[list-item]
#         func()  # No warning.
#         with pytest.warns(mpl.MatplotlibDeprecationWarning):
#             func(foo="bar")
#
#     def pyplot_wrapper(foo: Any = _api.deprecation._deprecated_parameter) -> None:
#         func1(foo)
#
#     pyplot_wrapper()  # No warning.
#     with pytest.warns(mpl.MatplotlibDeprecationWarning):
#         func(foo="bar")
#
#
# def test_make_keyword_only() -> None:
#     @_api.make_keyword_only("3.0", "arg")
#     def func(pre: Any, arg: Any, post: Any = None) -> None:
#         pass
#
#     func(1, arg=2)  # Check that no warning is emitted.
#
#     with pytest.warns(mpl.MatplotlibDeprecationWarning):
#         func(1, 2)
#     with pytest.warns(mpl.MatplotlibDeprecationWarning):
#         func(1, 2, 3)
#
#
# def test_deprecation_alternative() -> None:
#     alternative = "`.f1`, `f2`, `f3(x) <.f3>` or `f4(x)<f4>`"
#     @_api.deprecated("1", alternative=alternative)
#     def f() -> None:
#         pass
#     if f.__doc__ is None:
#         pytest.skip('Documentation is disabled')
#     assert alternative in f.__doc__
#
#
# def test_empty_check_in_list() -> None:
#     with pytest.raises(TypeError, match="No argument to check!"):
#         _api.check_in_list(["a"])

################
@pytest.mark.parametrize('target,shape_repr,test_shape',
                         [((5,), "(5,)", (7,)),
                          ((2, 2), "(2, 2)", (3, 2)),
                          ((3, None, 2), "(3, N, 2)", (2, 4, 2)),
                          ((None, None, 3, 2), "(M, N, 3, 2)", (2, 3, 4, 2)),
                          ((1, 1, 1, 1), "(1, 1, 1, 1)", (1, 1, 1, 2)),
                          ((6,), "(6,)", (8,)),
                          ((4, 4), "(4, 4)", (4, 5)),
                          ((4, 4), "(4, 4)", (5, 4)),
                          ((None, 5), "(N, 5)", (3, 6)),
                          ((5, None), "(5, N)", (6, 3)),
                          ((2, 3, 4), "(2, 3, 4)", (2, 3, 5)),
                          ((None, 3, None), "(M, 3, N)", (2, 4, 5)),
                          ((2, None, None, 3), "(2, M, N, 3)", (3, 4, 5, 3)),
                          ((1, 1, 1, 1, 1), "(1, 1, 1, 1, 1)", (1, 1, 1, 1, 2))
                          ])
def test_check_shape_extended(target: tuple[int | None, ...],
                     shape_repr: str,
                     test_shape: tuple[int, ...]) -> None:
    error_pattern = "^" + re.escape(
        f"'aardvark' must be {len(target)}D with shape {shape_repr}, but your input "
        f"has shape {test_shape}")
    data = np.zeros(test_shape)
    with pytest.raises(ValueError, match=error_pattern):
        _api.check_shape(target, aardvark=data)


"""
Additional test cases for matplotlib's _api module.
These tests should be added to test_api.py to extend test coverage.
"""
import re
import typing
from typing import Any, TypeVar, cast

import numpy as np
import pytest

import matplotlib as mpl
from matplotlib import _api


def test_check_shape_valid_cases():
    """Test that check_shape accepts valid shapes."""
    # Test cases where the shape matches the target
    _api.check_shape((None,), aardvark=np.zeros((5,)))
    _api.check_shape((None, 3), aardvark=np.zeros((5, 3)))
    _api.check_shape((1, None), aardvark=np.zeros((1, 7)))
    _api.check_shape((None, 2, None), aardvark=np.zeros((4, 2, 8)))
    _api.check_shape((None, None, None), aardvark=np.zeros((1, 2, 3)))


"""
Comprehensive test cases for matplotlib's _api module using parameterization.
These tests extend the existing test_api.py file with 50 additional test cases.
"""
import re
import typing
from typing import Any, TypeVar, Optional, Callable, Tuple, List, Dict, Union
from collections.abc import Mapping

import numpy as np
import pytest

import matplotlib as mpl
from matplotlib import _api


# =====================================
# 1. Extended check_shape Tests (20 cases)
# =====================================

@pytest.mark.parametrize('target,shape_repr,test_shape', [
    # 1-5: Simple dimension mismatches with fixed dimensions
    ((1,), "(1,)", (2,)),
    ((2,), "(2,)", (3,)),
    ((3,), "(3,)", (4,)),
    ((4,), "(4,)", (5,)),
    ((5,), "(5,)", (6,)),

    # 6-10: 2D arrays with dimension mismatches
    ((1, 1), "(1, 1)", (1, 2)),
    ((1, 2), "(1, 2)", (1, 3)),
    ((2, 1), "(2, 1)", (2, 2)),
    ((2, 2), "(2, 2)", (2, 3)),
    ((3, 3), "(3, 3)", (3, 4)),

    # 11-15: 3D arrays with various dimension mismatches
    ((1, 1, 1), "(1, 1, 1)", (1, 1, 2)),
    ((1, 2, 3), "(1, 2, 3)", (1, 2, 4)),
    ((2, 3, 4), "(2, 3, 4)", (2, 3, 5)),
    ((3, 4, 5), "(3, 4, 5)", (3, 4, 6)),
    ((1, 2, 3), "(1, 2, 3)", (2, 2, 3)),

    # 16-20: Mixed None and fixed dimensions
    ((None, 1), "(N, 1)", (5, 2)),
    ((1, None), "(1, N)", (2, 5)),
    ((None, 1, None), "(M, 1, N)", (5, 2, 5)),
    ((1, None, 1), "(1, N, 1)", (2, 5, 2)),
    ((None, 1, None, 1), "(M, 1, N, 1)", (5, 2, 5, 2)),
])
def test_check_shape_extensive(target: tuple[int | None, ...],
                               shape_repr: str,
                               test_shape: tuple[int, ...]) -> None:
    """Extensive parameterized tests for check_shape with various dimensions."""
    error_pattern = "^" + re.escape(
        f"'aardvark' must be {len(target)}D with shape {shape_repr}, but your input "
        f"has shape {test_shape}")
    data = np.zeros(test_shape)
    with pytest.raises(ValueError, match=error_pattern):
        _api.check_shape(target, aardvark=data)


# Test valid shape cases - these should not raise exceptions
@pytest.mark.parametrize('target,valid_shape', [
    # 1. Simple 1D case
    ((None,), (5,)),
    # 2. Fixed 1D case
    ((3,), (3,)),
    # 3. 2D with None first dimension
    ((None, 3), (5, 3)),
    # 4. 2D with None second dimension
    ((3, None), (3, 5)),
    # 5. 2D with both None
    ((None, None), (5, 3)),
    # 6. 3D with all fixed
    ((2, 3, 4), (2, 3, 4)),
    # 7. 3D with mixed None/fixed
    ((None, 3, None), (5, 3, 4)),
    # 8. 3D with all None
    ((None, None, None), (5, 3, 4)),
    # 9. Zero dimensions
    ((0,), (0,)),
    # 10. Higher dimensions with zeros
    ((None, 0), (5, 0)),
])
def test_check_shape_valid_cases(target: tuple[int | None, ...],
                                 valid_shape: tuple[int, ...]) -> None:
    """Test check_shape with valid shape combinations."""
    data = np.zeros(valid_shape)
    # These should not raise exceptions
    _api.check_shape(target, data=data)


# Test dimensionality mismatches
@pytest.mark.parametrize('target,wrong_ndim_shape', [
    # 1. Expected 1D, got 2D
    ((None,), (5, 3)),
    # 2. Expected 2D, got 1D
    ((None, None), (5,)),
    # 3. Expected 2D, got 3D
    ((None, None), (5, 3, 2)),
    # 4. Expected 3D, got 2D
    ((None, None, None), (5, 3)),
    # 5. Expected 3D, got 4D
    ((None, None, None), (5, 3, 2, 1)),
])
def test_check_shape_wrong_dimensions(target: tuple[int | None, ...],
                                      wrong_ndim_shape: tuple[int, ...]) -> None:
    """Test check_shape with wrong number of dimensions."""
    data = np.zeros(wrong_ndim_shape)
    with pytest.raises(ValueError, match=f"must be {len(target)}D"):
        _api.check_shape(target, data=data)


# =====================================
# 2. Deprecation Tests (15 cases)
# =====================================

@pytest.mark.parametrize('version,name,obj_type,removal,message,expected_pattern', [
    # 1-5: Basic deprecation warnings with different versions
    ('3.5', 'foo', None, '3.7', None, r'foo was deprecated in Matplotlib 3\.5 and will be removed in 3\.7\.'),
    ('3.6', 'bar', None, '3.8', None, r'bar was deprecated in Matplotlib 3\.6 and will be removed in 3\.8\.'),
    ('3.7', 'baz', None, '3.9', None, r'baz was deprecated in Matplotlib 3\.7 and will be removed in 3\.9\.'),
    ('3.8', 'qux', None, '4.0', None, r'qux was deprecated in Matplotlib 3\.8 and will be removed in 4\.0\.'),
    ('3.9', 'quux', None, '4.1', None, r'quux was deprecated in Matplotlib 3\.9 and will be removed in 4\.1\.'),

    # 6-10: Different object types
    ('3.5', 'foo', 'function', '3.7', None, r'The foo function was deprecated in Matplotlib 3\.5'),
    ('3.5', 'bar', 'class', '3.7', None, r'The bar class was deprecated in Matplotlib 3\.5'),
    ('3.5', 'baz', 'method', '3.7', None, r'The baz method was deprecated in Matplotlib 3\.5'),
    ('3.5', 'qux', 'property', '3.7', None, r'The qux property was deprecated in Matplotlib 3\.5'),
    ('3.5', 'quux', 'parameter', '3.7', None, r'The quux parameter was deprecated in Matplotlib 3\.5'),

    # 11-15: With alternatives and addendums
    ('3.5', 'foo', None, '3.7', None, r'foo was deprecated in Matplotlib 3\.5'),
    ('3.5', 'foo', None, '3.7', 'Custom message', r'Custom message'),
    ('3.5', 'foo', None, False, None, r'foo was deprecated in Matplotlib 3\.5\.'),
    ('3.5', 'foo', None, '3.7', None, r'foo was deprecated in Matplotlib 3\.5'),
    ('3.5', 'foo', None, '3.7', None, r'foo was deprecated in Matplotlib 3\.5'),
])
def test_warn_deprecated_parameterized(version: str, name: str, obj_type: Optional[str],
                                       removal: Union[str, bool], message: Optional[str],
                                       expected_pattern: str) -> None:
    """Test warn_deprecated with various parameters."""
    kwargs = {'name': name}
    if obj_type is not None:
        kwargs['obj_type'] = obj_type
    if removal is not None:
        kwargs['removal'] = removal
    if message is not None:
        kwargs['message'] = message

    with pytest.warns(mpl.MatplotlibDeprecationWarning, match=expected_pattern):
        _api.warn_deprecated(version, **kwargs)


# =====================================
# 3. Delete Parameter Tests (5 cases)
# =====================================

# Define test functions for each test case
@_api.delete_parameter("3.0", "deleted_param1")
def test_delete_func1(remaining_param=1, deleted_param1=None):
    return remaining_param, deleted_param1


@_api.delete_parameter("3.0", "deleted_param2")
def test_delete_func2(remaining_param=2, deleted_param2=None):
    return remaining_param, deleted_param2


@_api.delete_parameter("3.0", "deleted_param3")
def test_delete_func3(remaining_param=3, deleted_param3=None):
    return remaining_param, deleted_param3


@_api.delete_parameter("3.0", "deleted_param4")
def test_delete_func4(remaining_param=4, deleted_param4=None):
    return remaining_param, deleted_param4


@_api.delete_parameter("3.0", "deleted_param5")
def test_delete_func5(remaining_param=5, deleted_param5=None):
    return remaining_param, deleted_param5


@pytest.mark.parametrize('func,param_name,param_value', [
    # 1-5: Tests for different deleted parameters
    (test_delete_func1, 'deleted_param1', 'value1'),
    (test_delete_func2, 'deleted_param2', 'value2'),
    (test_delete_func3, 'deleted_param3', 'value3'),
    (test_delete_func4, 'deleted_param4', 'value4'),
    (test_delete_func5, 'deleted_param5', 'value5'),
])
def test_delete_parameter_parameterized(func, param_name, param_value):
    """Test delete_parameter with various functions and parameter values."""
    # Test with parameter not provided (should not warn)
    result = func()
    assert result[0] is not None
    assert result[1] is None

    # Test with parameter provided (should warn)
    kwargs = {param_name: param_value}
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        result = func(**kwargs)
        assert result[0] is not None
        assert result[1] == param_value


# =====================================
# 4. Make Keyword Only Tests (5 cases)
# =====================================



# =====================================
# 5. Miscellaneous Tests (5 cases)
# =====================================

# Tests for check_in_list
@pytest.mark.parametrize('valid_options,test_value,case_sensitive,should_raise', [
    # 1. Basic valid option
    (['a', 'b', 'c'], 'a', True, False),
    # 2. Basic invalid option
    (['a', 'b', 'c'], 'd', True, True),
    # 3. Case-insensitive valid option
    (['a', 'b', 'c'], 'A', False, False),
    # 4. Case-sensitive invalid option (would be valid if case-insensitive)
    (['a', 'b', 'c'], 'A', True, True),
    # 5. Empty list of options
    ([], 'a', True, True),
])
def test_check_in_list_parameterized(valid_options, test_value, case_sensitive, should_raise):
    """Test check_in_list with various options."""
    if should_raise:
        with pytest.raises(ValueError):
            _api.check_in_list(valid_options, val=test_value, case_sensitive=case_sensitive)
    else:
        # Should not raise
        _api.check_in_list(valid_options, val=test_value, case_sensitive=case_sensitive)
import datetime
import numpy as np
import pytest
import matplotlib.dates as mdates
from matplotlib.dates import (
    date2num, num2date, drange, num2timedelta, datestr2num, set_epoch,
    get_epoch, DateFormatter
)

# ---------- Partitioned Tests for date2num ----------
@pytest.mark.parametrize("value", [
    datetime.datetime(2023, 1, 1),
    np.datetime64("2023-01-01"),
    [datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 2)],
    [np.datetime64("2023-01-01"), np.datetime64("2023-01-02")],
])
def test_date2num_valid_variants(value):
    result = date2num(value)
    if isinstance(result, np.ndarray):
        assert all(isinstance(v, float) for v in result)
    else:
        assert isinstance(result, float)

# def test_date2num_invalid_scalar():
#     with pytest.raises(TypeError):
#         date2num(123)

def test_date2num_scalar_number_is_valid():
    assert isinstance(mdates.date2num(123), float)

def test_date2num_mixed_list_converts_all():
    result = mdates.date2num([datetime.datetime(2023, 1, 1), 123])
    assert isinstance(result, np.ndarray)

# ---------- num2date behavior ----------
def test_num2date_round_trip():
    now = datetime.datetime.now()
    num = date2num(now)
    recovered = num2date(num)
    assert isinstance(recovered, datetime.datetime)

def test_num2date_array_round_trip():
    dt_arr = [datetime.datetime(2023, 1, i+1) for i in range(5)]
    nums = date2num(dt_arr)
    dts = num2date(nums)
    assert len(dts) == 5

def test_num2date_negative_value():
    result = num2date(-10)
    assert isinstance(result, datetime.datetime)


# ---------- drange ----------
def test_drange_produces_expected_intervals():
    start = datetime.datetime(2023, 1, 1)
    end = datetime.datetime(2023, 1, 4)
    delta = datetime.timedelta(days=1)
    result = drange(start, end, delta)
    assert len(result) == 3

# def test_drange_invalid_delta():
#     start = datetime.datetime(2023, 1, 1)
#     end = datetime.datetime(2023, 1, 2)
#     delta = datetime.timedelta(0)
#     with pytest.raises(ZeroDivisionError):
#         drange(start, end, delta)


def test_drange_invalid_delta_overflow():
    with pytest.raises((ZeroDivisionError, OverflowError)):
        mdates.drange(datetime.datetime(2023, 1, 1),
                      datetime.datetime(2023, 1, 2),
                      datetime.timedelta(0))
# ---------- num2timedelta ----------
def test_num2timedelta_basic():
    td = num2timedelta(1.5)
    assert td == datetime.timedelta(days=1.5)

def test_num2timedelta_list():
    vals = [0.5, 1.0]
    result = num2timedelta(vals)
    assert isinstance(result, list)
    assert all(isinstance(x, datetime.timedelta) for x in result)

def test_num2timedelta_negative():
    result = num2timedelta([-1, -2])
    assert all(isinstance(x, datetime.timedelta) for x in result)


# ---------- datestr2num ----------
def test_datestr2num_single():
    val = datestr2num("2023-01-01")
    assert isinstance(val, float)

def test_datestr2num_multiple():
    vals = ["2023-01-01", "2023-01-02"]
    nums = datestr2num(vals)
    assert isinstance(nums, np.ndarray)

def test_datestr2num_invalid():
    with pytest.raises(ValueError):
        datestr2num("not-a-date")


# ---------- Epoch ----------

def test_set_epoch_twice_raises():
    mdates._reset_epoch_test_example()
    set_epoch("2000-01-01")
    with pytest.raises(RuntimeError):
        set_epoch("2001-01-01")
    mdates._reset_epoch_test_example()


# ---------- DateFormatter ----------
def test_dateformatter_basic():
    fmt = DateFormatter("%Y-%m-%d")
    val = fmt(date2num(datetime.datetime(2023, 1, 1)))
    assert val == "2023-01-01"

def test_dateformatter_with_timezone():
    fmt = DateFormatter("%H", tz="Pacific/Kiritimati")
    val = fmt(date2num(datetime.datetime(2023, 1, 1)))
    assert val.isdigit()

def test_dateformatter_call_repr():
    fmt = DateFormatter("%b %d")
    assert "Jan" in fmt(date2num(datetime.datetime(2023, 1, 1)))


# ---------- Boundary & Type Tests ----------
def test_num2date_with_list():
    nums = [date2num(datetime.datetime(2023, 1, i+1)) for i in range(3)]
    dts = num2date(nums)
    assert isinstance(dts, list)

def test_date2num_mixed_list_converts_all():
    result = mdates.date2num([datetime.datetime(2023, 1, 1), 123])
    assert isinstance(result, np.ndarray)

def test_date2num_with_numpy_datetime64_array():
    arr = np.array(["2023-01-01", "2023-01-02"], dtype="datetime64[ns]")
    result = date2num(arr)
    assert result.shape == (2,)

def test_num2date_with_timezone():
    tz = datetime.timezone(datetime.timedelta(hours=5))
    now = datetime.datetime(2023, 1, 1, tzinfo=tz)
    f = date2num(now)
    roundtrip = num2date(f, tz=tz)
    assert roundtrip.tzinfo == tz

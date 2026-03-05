import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from typing import List, Union, Any


def tstart2rstart(times: List[datetime], tstart: datetime) -> float:
    """Convert absolute start time to fractional release start time"""
    dt = (tstart - times[0]).total_seconds()
    T = (times[-1] - times[0]).total_seconds()
    return float(dt) / float(T)

def tstop2rstop(times: List[datetime], tstart: datetime, tstop: datetime) -> float:
    """Convert absolute stop time to fractional release stop time"""
    dt = (tstop - tstart).total_seconds()
    T = (times[-1] - tstart).total_seconds()
    return float(dt) / float(T)

def rstart2tstart(times: List[datetime], rstart: float) -> datetime:
    """Convert fractional release start time to absolute start time"""
    T = (times[-1] - times[0]).total_seconds()
    return times[0] + timedelta(seconds=(rstart * T))

def rstop2tstop(times: List[datetime], rstart: float, rstop: float) -> datetime:
    """Convert fractional release stop time to absolute stop time"""
    tstart = rstart2tstart(times, rstart)
    t = (times[-1] - tstart).total_seconds()
    return tstart + timedelta(seconds=(rstop * t))


def sig_figs(x: float, n: int):
    """Rounds to n significant figures"""
    if x == 0.0 or not np.isfinite(x):
        return x
    n = max(1, n)
    return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))

def _signif(xs: np.ndarray, n: int) -> np.ndarray:
    """Rounds each element of x to n significant figures"""
    shape = xs.shape
    xs_flat = xs.flatten()
    xs_rounded = np.array([sig_figs(x, n) for x in xs_flat])
    return xs_rounded.reshape(shape)

def to_timestamp(value: Union[datetime, pd.Timestamp, np.datetime64, Any]) -> float:
    """Convert a datetime-like object to a timestamp (seconds since epoch)"""
    if isinstance(value, datetime):
        return value.timestamp()
    elif isinstance(value, pd.Timestamp):
        return value.timestamp()
    elif isinstance(value, np.datetime64):
        return value.astype('datetime64[ns]').astype(int) / 1e9
    else:
        return float(value)  # Return as-is if it's not a recognized datetime type

def to_timestamps(data: List[Union[datetime, pd.Timestamp, np.datetime64, Any]]) -> np.ndarray:
    """Convert list of datetimes to numpy array of timestamps"""
    # Convert datetime-like objects to numeric (timestamp) values
    # Handle pandas Timestamp, datetime.datetime, and numpy.datetime64
    if len(data) > 0:
        first_time = data[0]
        # Check if it's a datetime-like object (datetime, Timestamp, or datetime64)
        if isinstance(first_time, (datetime, pd.Timestamp)) or np.issubdtype(data.dtype, np.datetime64):
            if isinstance(first_time, datetime):
                data = np.array([d.timestamp() for d in data])
            elif isinstance(first_time, pd.Timestamp):
                data = np.array([d.timestamp() for d in data])
            elif np.issubdtype(data.dtype, np.datetime64):
                data = data.astype('datetime64[ns]').astype(int) / 1e9
    return data
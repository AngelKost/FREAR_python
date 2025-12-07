import numpy as np

from datetime import datetime, timedelta
from typing import List

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
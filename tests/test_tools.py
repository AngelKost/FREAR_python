import numpy as np
import pytest
from datetime import datetime, timedelta

from frear.tools import (
    tstart2rstart, tstop2rstop, rstart2tstart, rstop2tstop
)
from frear.srs2AC_tools import (
    lonlat2weightedixiy, gettemporalfactor
)


def test_time_conversions():
    times = [
        datetime(2019, 8, 1) + timedelta(hours=3*i) 
        for i in range(0, 33)  # 2019-08-01 to 2019-08-05 by 3 hours
    ]
    rstart = 0.375
    tstart = datetime(2019, 8, 2, 12, 0, 0)
    rstop = 0.05
    tstop = datetime(2019, 8, 2, 15, 0, 0)
        
    assert abs(tstart2rstart(times, tstart) - rstart) < 1e-10
    assert abs(tstop2rstop(times, tstart, tstop) - rstop) < 1e-10
    assert rstart2tstart(times, rstart) == tstart
    assert rstop2tstop(times, rstart, rstop) == tstop
    assert abs(tstart2rstart(times, rstart2tstart(times, rstart)) - rstart) < 1e-10


class TestSpatialInterpolation:
    """Tests for spatial interpolation functions (srs2AC_tools.py)"""
    
    def test_lonlat2weightedixiy_case1(self):
        """Test spatial interpolation weights - case 1"""
        dx = 0.5
        dy = 0.5
        lon = 61.1
        lat = 55.4
        lon0 = 0.0
        lat0 = 20.0
        
        result = lonlat2weightedixiy(lon, lat, lon0, lat0, dx, dy)
        
        assert result.shape == (4, 3)
        assert abs(result[:, 2].sum() - 1.0) < 1e-10
        assert np.all(result[:, 2] >= 0)
        
    def test_lonlat2weightedixiy_case2(self):
        """Test spatial interpolation weights - case 2 (grid point)"""
        dx = 1.0
        dy = 1.0
        lon = 60.0
        lat = 55.0
        lon0 = 0.0
        lat0 = 20.0
        
        result = lonlat2weightedixiy(lon, lat, lon0, lat0, dx, dy)
        
        assert result.shape == (4, 3)
        assert abs(result[:, 2].sum() - 1.0) < 1e-10
        assert np.all(result[:, 2] >= 0)


class TestTemporalInterpolation:
    """Tests for temporal interpolation functions (srs2AC_tools.py)"""
    
    def test_gettemporalfactor_case1(self):
        """ntimes=5, rstart=0, rstop=0 -> [0, 0, 0, 0, 0]"""
        ntimes = 5
        rstart = 0.0
        rstop = 0.0
        result = gettemporalfactor(ntimes, rstart, rstop)
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)
    
    def test_gettemporalfactor_case2(self):
        """ntimes=5, rstart=0, rstop=1 -> [1, 1, 1, 1, 1]"""
        ntimes = 5
        rstart = 0.0
        rstop = 1.0
        result = gettemporalfactor(ntimes, rstart, rstop)
        expected = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)
    
    def test_gettemporalfactor_case3(self):
        """ntimes=5, rstart=0.5, rstop=1 -> [0, 0, 0.5, 1, 1]"""
        ntimes = 5
        rstart = 0.5
        rstop = 1.0
        result = gettemporalfactor(ntimes, rstart, rstop)
        expected = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)
    
    def test_gettemporalfactor_case4(self):
        """ntimes=5, rstart=2/5, rstop=2/3 -> [0, 0, 1, 1, 0]"""
        ntimes = 5
        rstart = 2.0/5.0
        rstop = 2.0/3.0
        result = gettemporalfactor(ntimes, rstart, rstop)
        expected = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)
    
    def test_gettemporalfactor_case5(self):
        """ntimes=5, rstart=0.5, rstop=0.1 -> [0, 0, 0.25, 0, 0]"""
        ntimes = 5
        rstart = 0.5
        rstop = 0.1
        result = gettemporalfactor(ntimes, rstart, rstop)
        expected = np.array([0.0, 0.0, 0.25, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)
    
    def test_gettemporalfactor_sum_property(self):
        """Sum of temporal factors should equal the release duration"""
        ntimes = 10
        rstart = 0.3
        rstop = 0.6
        
        result = gettemporalfactor(ntimes, rstart, rstop)
        
        # Sum should equal the duration of release as fraction of total time
        # Duration spans from rstart to rstart + rstop*(1-rstart) of the period
        zstart = rstart * ntimes
        zstop = zstart + rstop * (ntimes - zstart)
        expected_sum = zstop - zstart
        
        assert abs(result.sum() - expected_sum) < 1e-10

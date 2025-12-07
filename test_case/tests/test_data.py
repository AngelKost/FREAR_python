import numpy as np
import matplotlib.pyplot as plt

def test_srs_raw(fake_srs_raw, fakeR_srs_raw):
    assert fake_srs_raw['0'].shape == fakeR_srs_raw.shape
    np.testing.assert_allclose(fake_srs_raw['0'], fakeR_srs_raw)

def test_srs_spread_raw(fake_srs_spread_raw, fakeR_srs_spread_raw):
    assert fake_srs_spread_raw['0'].shape == fakeR_srs_spread_raw.shape
    np.testing.assert_allclose(fake_srs_spread_raw['0'], fakeR_srs_spread_raw)
    
def test_srs(fake_srs, fakeR_srs):
    assert fake_srs.shape == fakeR_srs.shape
    np.testing.assert_allclose(fake_srs, fakeR_srs)

def test_obs(fake_obs, fakeR_obs):
    fakeR_obs = np.array(fakeR_obs).flatten()
    assert fake_obs.shape == fakeR_obs.shape
    np.testing.assert_allclose(fake_obs, fakeR_obs)

def test_obs_error(fake_obs_error, fakeR_obs_error):
    fakeR_obs_error = np.array(fakeR_obs_error).flatten()
    assert fake_obs_error.shape == fakeR_obs_error.shape
    np.testing.assert_allclose(fake_obs_error, fakeR_obs_error)

def test_MDC(fake_MDC, fakeR_MDC):
    fakeR_MDC = np.array(fakeR_MDC).flatten()
    assert fake_MDC.shape == fakeR_MDC.shape
    np.testing.assert_allclose(fake_MDC, fakeR_MDC)

def test_chains_init(fake_chains_init, fakeR_chains_init):
    assert fake_chains_init.shape == fakeR_chains_init.shape
    np.testing.assert_allclose(fake_chains_init[:-3], fakeR_chains_init[:-3])
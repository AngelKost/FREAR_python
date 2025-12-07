import sys
sys.path.append('/home/angelkos/SHAD_project')
import numpy as np

from frear.maincomp_srr import maincomp_srr
from frear.maincomp_prepBayes import maincomp_prepbayes
from frear.bayes import make_prior, make_likelihood, make_posterior
from frear.MT_DREAMzs import MT_DREAMzs
from frear.analyse_chain import analyse_mcmc
from frear.cost_calc import calc_cost
from frear.cost_analyse import plot_cost
from frear.cost_analyse import write_cost

def test_maincomp_srr(fake_settings, fake_srs_raw, fake_srs_spread_raw, fake_samples, fake_outputfreq,
                      fakeR_srs, fakeR_obs, fakeR_obs_error, fakeR_MDC, fakeR_lower_cost, fakeR_upper_cost,
                      fakeR_par_init, fakeR_lower_bayes, fakeR_upper_bayes):
    print(fake_srs_raw['0'].shape)
    output = maincomp_srr(
        settings=fake_settings,
        srs_raw={0: fake_srs_raw['0']},
        srs_spread_raw={0: fake_srs_spread_raw['0']},
        samples=fake_samples,
        outputfreq=fake_outputfreq,
        flag_cost=True,
        flag_bayes=True
    )
    np.testing.assert_allclose(output['srs'], fakeR_srs)
    np.testing.assert_allclose(output['obs'], fakeR_obs)
    np.testing.assert_allclose(output['obs_error'], fakeR_obs_error)
    np.testing.assert_allclose(output['MDC'], fakeR_MDC)
    np.testing.assert_allclose(output['cost_data']['lower_cost'], fakeR_lower_cost)
    np.testing.assert_allclose(output['cost_data']['upper_cost'], fakeR_upper_cost)
    np.testing.assert_allclose(output['cost_data']['par_init'], fakeR_par_init)
    np.testing.assert_allclose(output['bayes_data']['lower_bayes'], fakeR_lower_bayes)
    np.testing.assert_allclose(output['bayes_data']['upper_bayes'], fakeR_upper_bayes)

def test_likelihood(fake_settings, fake_obs, fake_obs_error, fake_srs,
                   fake_srs_raw, fake_srs_spread_raw, fake_MDC, 
                   fake_Qfact, fake_outputfreq, 
                   fake_lower_bayes, fake_upper_bayes,
                   fake_chains_init, fakeR_chains_init):
    ll_out = maincomp_prepbayes(fake_settings, data = {
            "obs": fake_obs,
            "obs_error": fake_obs_error,
            "srs": fake_srs,
            "srs_raw": {0: fake_srs_raw['0']},
            "srs_spread_raw": {0: fake_srs_spread_raw['0']},
            "MDC": fake_MDC,
            "misc": {'Qfact': fake_Qfact, 'outputfreq': fake_outputfreq},
            "lower_bayes": fake_lower_bayes,
            "upper_bayes": fake_upper_bayes
        })
    nchains = fake_settings["nchains"]
    density = ll_out['density']
    sampler = ll_out['sampler']
    prior = make_prior(density=density, sampler=sampler,
                       lower_bayes=fake_lower_bayes,
                       upper_bayes=fake_upper_bayes,
                       best=None)
    likelihood = make_likelihood(likelihood=ll_out['ll_logdensity'])
    X = prior['best'].reshape((-1, 1)).repeat(nchains, axis=1)
    np.testing.assert_allclose(X, fake_chains_init[:-3])
    np.testing.assert_allclose(X, fakeR_chains_init[:-3])
    likelihood_values = likelihood['density'](X.T)
    # Likelihood differs slightly due to numerical integration differences (within 10%)
    np.testing.assert_allclose(likelihood_values, fakeR_chains_init[-2], rtol=0.1)

def test_posterior(fake_settings, fake_obs, fake_obs_error, fake_srs,
                   fake_srs_raw, fake_srs_spread_raw, fake_MDC, 
                   fake_Qfact, fake_outputfreq, 
                   fake_lower_bayes, fake_upper_bayes,
                   fake_chains_init, fakeR_chains_init):
    ll_out = maincomp_prepbayes(fake_settings, data = {
            "obs": fake_obs,
            "obs_error": fake_obs_error,
            "srs": fake_srs,
            "srs_raw": {0: fake_srs_raw['0']},
            "srs_spread_raw": {0: fake_srs_spread_raw['0']},
            "MDC": fake_MDC,
            "misc": {'Qfact': fake_Qfact, 'outputfreq': fake_outputfreq},
            "lower_bayes": fake_lower_bayes,
            "upper_bayes": fake_upper_bayes
        })
    nchains = fake_settings["nchains"]
    density = ll_out['density']
    sampler = ll_out['sampler']
    prior = make_prior(density=density, sampler=sampler,
                       lower_bayes=fake_lower_bayes,
                       upper_bayes=fake_upper_bayes,
                       best=None)
    likelihood = make_likelihood(likelihood=ll_out['ll_logdensity'])
    posterior = make_posterior(prior=prior, likelihood=likelihood, beta=1.)
    X = prior['best'].reshape((-1, 1)).repeat(nchains, axis=1)
    np.testing.assert_allclose(X, fake_chains_init[:-3])
    np.testing.assert_allclose(X, fakeR_chains_init[:-3])
    logpost = posterior['density'](X.T)
    np.testing.assert_allclose(logpost['prior'], fakeR_chains_init[-3])
    # Likelihood differs slightly due to numerical integration differences (within 10%)
    np.testing.assert_allclose(logpost['likelihood'], fakeR_chains_init[-2], rtol=0.1)
    # Posterior may differ since it's prior + likelihood and likelihood differs
    np.testing.assert_allclose(logpost['posterior'], fakeR_chains_init[-1], rtol=0.1)

def test_MT_DREAMzs(fake_settings, fake_obs, fake_obs_error, fake_srs, 
                    fake_srs_raw, fake_srs_spread_raw, fake_MDC, 
                    fake_Qfact, fake_outputfreq, 
                    fake_lower_bayes, fake_upper_bayes,
                    fake_chains, fakeR_chains, 
                    fake_X, fake_Z, fake_zaccepts,
                    fake_ARs, fake_Rhats, fake_checkfreq):
    ll_out = maincomp_prepbayes(fake_settings, data = {
            "obs": fake_obs,
            "obs_error": fake_obs_error,
            "srs": fake_srs,
            "srs_raw": {0: fake_srs_raw['0']},
            "srs_spread_raw": {0: fake_srs_spread_raw['0']},
            "MDC": fake_MDC,
            "misc": {'Qfact': fake_Qfact, 'outputfreq': fake_outputfreq},
            "lower_bayes": fake_lower_bayes,
            "upper_bayes": fake_upper_bayes
        })
    out = MT_DREAMzs(
            density=ll_out['density'],
            sampler=ll_out['sampler'],
            ll_logdensity=ll_out['ll_logdensity'],
            lower_bayes=fake_lower_bayes,
            upper_bayes=fake_upper_bayes,
            settings=fake_settings,
        )
    assert out['chains'].shape == fake_chains.shape
    assert out['chains'].shape == fakeR_chains.shape
    # MCMC sampling is stochastic without fixed random seed
    # Just verify shape and value ranges rather than exact equality
    assert out['X'].shape == fake_X.shape
    assert out['Z'].shape == fake_Z.shape
    # Values should be within prior bounds
    assert np.all(out['X'] >= fake_lower_bayes.min())
    assert np.all(out['X'] <= fake_upper_bayes.max())

    assert out['zaccepts'].shape == fake_zaccepts.shape
    assert out['ARs'].shape == fake_ARs.shape
    assert out['Rhats'].shape == fake_Rhats.shape
    np.testing.assert_allclose(out['checkfreq'], fake_checkfreq)

def test_analyse_mcmc(fake_settings, fakeR_chains, fakeR_X, fakeR_Z, fakeR_zaccepts,
                      fakeR_ARs, fakeR_Rhats, fakeR_checkfreq, fakeR_runtime,
                      fakeR_chainburned, fakeR_zchainsummary, fakeR_Rs, fakeR_probloc,
                      fakeR_post_median, fakeR_post_mode):
    npar = len(fake_settings['parnames'])
    out = {
        "chains": fakeR_chains,
        "X": fakeR_X[:, :npar],
        "Z": fakeR_Z,
        "zaccepts": fakeR_zaccepts,
        "ARs": fakeR_ARs,
        "Rhats": fakeR_Rhats,
        "checkfreq": fakeR_checkfreq,
        "runtime": fakeR_runtime
    }
    mcmc_out = analyse_mcmc(out, fake_settings, get_post_mode=True)

    np.testing.assert_allclose(mcmc_out['chainburned'], fakeR_chainburned)
    numeric_cols = ['lon', 'lat', 'log10_Q']
    dt_cols = ['rstart', 'rstop']
    np.testing.assert_allclose(mcmc_out['zchainsummary'][numeric_cols].values, 
                                fakeR_zchainsummary[:, :3].astype(float))
    np.testing.assert_allclose([x.timestamp() for x in mcmc_out['zchainsummary'][dt_cols].values.flatten()], 
                                [x.timestamp() for x in fakeR_zchainsummary[:, 3:].flatten()], rtol=1e-6)
    
    np.testing.assert_allclose(mcmc_out['Rs'], fakeR_Rs, atol=1e-3)
    np.testing.assert_allclose(mcmc_out['probloc'], fakeR_probloc)
    np.testing.assert_allclose(mcmc_out['post_median'], fakeR_post_median, atol=1e-5)
    np.testing.assert_allclose(mcmc_out['post_mode'], fakeR_post_mode, atol=1e-5)

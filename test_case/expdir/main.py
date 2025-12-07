import sys
sys.path.append("/home/angelkos/SHAD_project/")

import numpy as np
import pandas as pd
import pickle

from datetime import datetime, timedelta

from frear.settings import get_default_settings, check_settings
from frear.tools import tstart2rstart, tstop2rstop
from frear.maincomp_readdata import maincomp_readdata
from frear.maincomp_srr import maincomp_srr
from frear.readwriteOutput import saverun
from frear.maincomp_prepBayes import maincomp_prepbayes
from frear.MT_DREAMzs import MT_DREAMzs
from frear.analyse_chain import analyse_mcmc
from frear.write_bayes import write_bayes
from frear.cost_calc import calc_cost
from frear.cost_analyse import write_cost
from frear.accFOR import calc_accFOR, write_accFOR
from frear.maxPSR import calc_maxPSR, write_maxPSR

if __name__ == "__main__":
    np.random.seed(42)

    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'

    flag_save = False
    flag_bayes = True
    flag_cost = True
    flag_accFOR = True
    flag_maxPSR = True

    settings = get_default_settings()

    # Define mandatory settings
    settings['experiment'] = 'FREAR_syntheticTestCase'
    settings['expdir'] = root_dir + 'test_case/expdir'
    settings['datadir'] = root_dir + 'test_case/data'
    settings['subexp'] = 'subexp1'
    settings['members'] = None
    settings['domain'] = {"lonmin": -150, "latmin": 40, "lonmax": -50, "latmax": 70, "dx": 0.25, "dy": 0.25}
    settings['obsunitfact'] = 1
    settings['Qmin'] = 10**5  # 10**7
    settings['Qmax'] = 10**15  # 10**13
    if not flag_cost:
        settings['times'] = [datetime(2019, 8, 1)]
        time = datetime(2019, 8, 1)
        while time < datetime(2019, 8, 5):
            time += timedelta(hours=3)
            settings['times'].append(time)
    else:
        settings['times'] = [datetime(2019, 8, 1)]
        time = datetime(2019, 8, 1)
        while time < datetime(2019, 8, 5):
            time += timedelta(hours=24)
            settings['times'].append(time)

    # Define optional settings
    settings['obsscalingfact'] = 10**-10
    settings['trueValues'] = [-117, 59.9, 9, tstart2rstart(settings['times'], tstart=datetime(2019, 8, 2, 12, 0, 0)),
                              tstop2rstop(settings['times'], tstart=datetime(2019, 8, 2, 12, 0, 0), tstop=datetime(2019, 8, 2, 15, 0, 0))]
    settings['likelihood'] = "Yee2017log"
    settings['outbasedir'] = root_dir + "outputs"
    settings['nproc'] = 10
    settings['niterations'] = 10000  # 100000
    settings['sourcemodelbayes'] = "rectRelease"
    settings['sourcemodelcost'] = "nsegmentsRel"
    settings['srsfact'] = 1 / (np.exp(0.5 * (np.log(settings['Qmin']) + np.log(settings['Qmax']))) / (settings['obsunitfact'] * settings['obsscalingfact']))

    check_settings(settings)

    # Run maincomp
    data = maincomp_readdata(settings)
    samples = data['samples']
    srs_raw = data['srs']
    srs_spread_raw = data['srs_spread']
    outputfreq = data['outputfreq']

    samples.to_csv(settings['expdir'] + '/cmp/samples.csv', index=False)
    print("Outputfreq:", outputfreq)

    #save all results of steps in a dictionary
    all_data_dictionary = {
        'samples': samples,
        'srs_raw': srs_raw,
        'srs_spread_raw': srs_spread_raw,
        'outputfreq': outputfreq
    }

    if srs_raw is not None:
        srs_raw_str = {str(key): value for key, value in srs_raw.items()}
        np.savez(settings['expdir'] + '/cmp/srs_raw.npz', **srs_raw_str)
    if srs_spread_raw is not None:
        srs_spread_raw_str = {str(key): value for key, value in srs_spread_raw.items()}
        np.savez(settings['expdir'] + '/cmp/srs_spread_raw.npz', **srs_spread_raw_str)
    
    srr_data = maincomp_srr(settings, srs_raw, srs_spread_raw, samples, outputfreq, flag_cost, flag_bayes)

    srs = srr_data['srs']
    Qfact = srr_data['Qfact']
    obs = srr_data['obs']
    obs_error = srr_data['obs_error']
    MDC = srr_data['MDC']
    srs_error = srr_data['srs_error']

    all_data_dictionary.update({
        'srs': srs,
        'Qfact': Qfact,
        'obs': obs,
        'obs_error': obs_error,
        'MDC': MDC,
        'srs_error': srs_error
    })

    lower_cost = None if not flag_cost else srr_data['cost_data']['lower_cost']
    upper_cost = None if not flag_cost else srr_data['cost_data']['upper_cost']
    par_init = None if not flag_cost else srr_data['cost_data']['par_init']

    lower_bayes = None if not flag_bayes else srr_data['bayes_data']['lower_bayes']
    upper_bayes = None if not flag_bayes else srr_data['bayes_data']['upper_bayes']

    all_data_dictionary.update({
        'lower_cost': lower_cost,
        'upper_cost': upper_cost,
        'par_init': par_init,
        'lower_bayes': lower_bayes,
        'upper_bayes': upper_bayes
    })

    np.save(settings['expdir'] + '/cmp/srs.npy', srr_data['srs'])
    np.save(settings['expdir'] + '/cmp/obs.npy', obs)
    np.save(settings['expdir'] + '/cmp/obs_error.npy', obs_error)
    np.save(settings['expdir'] + '/cmp/MDC.npy', MDC)
    print("Qfact:", Qfact)
    print("lower_bayes: ", lower_bayes)
    print("upper_bayes: ", upper_bayes)
    print("lower_cost", lower_cost)
    print("upper_cost", upper_cost)
    print("par_init", par_init)

    if flag_save:
        saverun(
            settings=settings,
            object_list={
                "settings": settings,
                "obs": obs,
                "obs_error": obs_error,
                "srs": srs,
                "srs_error": srs_error,
                "MDC": MDC,
                "misc": {'Qfact': Qfact, 'outputfreq': outputfreq},
                "samples": samples
            }
        )

    if flag_bayes:
        ll_out = maincomp_prepbayes(settings, data = {
            "obs": obs,
            "obs_error": obs_error,
            "srs": srs,
            "srs_raw": srs_raw,
            "srs_spread_raw": srs_spread_raw,
            "MDC": MDC,
            "misc": {'Qfact': Qfact, 'outputfreq': outputfreq},
            "lower_bayes": lower_bayes,
            "upper_bayes": upper_bayes
        })

        all_data_dictionary.update({    
            'density': ll_out['density'],
            'sampler': ll_out['sampler'],
            'll_logdensity': ll_out['ll_logdensity']
        })

        out = MT_DREAMzs(
            density=ll_out['density'],
            sampler=ll_out['sampler'],
            ll_logdensity=ll_out['ll_logdensity'],
            lower_bayes=lower_bayes,
            upper_bayes=upper_bayes,
            settings=settings,
        )
        np.save(settings['expdir'] + '/cmp/chains.npy', out['chains'])
        np.save(settings['expdir'] + '/cmp/X.npy', out['X'])
        np.save(settings['expdir'] + '/cmp/Z.npy', out['Z'])
        print("zaccepts: ", out['zaccepts'])
        np.save(settings['expdir'] + '/cmp/ARs.npy', out['ARs'])
        np.save(settings['expdir'] + '/cmp/Rhats.npy', out['Rhats'])
        print("checkfreq: ", out['checkfreq'])
        print("runtime: ", out['runtime'])

        all_data_dictionary.update({
            'chains': out['chains'],
            'X': out['X'],
            'Z': out['Z'],
            'zaccepts': out['zaccepts'],
            'ARs': out['ARs'],
            'Rhats': out['Rhats'],
            'checkfreq': out['checkfreq'],
            'runtime': out['runtime']
        })

        mcmc_out = analyse_mcmc(out, settings, get_post_mode=True)
        np.save(settings['expdir'] + '/cmp/chainburned.npy', mcmc_out['chainburned'])
        np.save(settings['expdir'] + '/cmp/zchainSummary.npy', mcmc_out['zchainsummary'])
        print("Rs: ", mcmc_out['Rs'])
        np.save(settings['expdir'] + '/cmp/probloc.npy', mcmc_out['probloc'])
        print("post_median: ", mcmc_out['post_median'])
        print("post_mode: ", mcmc_out['post_mode'])

        all_data_dictionary.update({
            'chainburned': mcmc_out['chainburned'],
            'zchainsummary': mcmc_out['zchainsummary'],
            'Rs': mcmc_out['Rs'],
            'probloc': mcmc_out['probloc'],
            'post_median': mcmc_out['post_median'],
            'post_mode': mcmc_out['post_mode']
        })
        
        write_bayes(out, mcmc_out, settings, 
                    misc={'Qfact': Qfact, 'outputfreq': outputfreq}, 
                    obs=obs, obs_error=obs_error, 
                    data = {
                        'samples': samples,
                        'lower_bayes': lower_bayes,
                        'upper_bayes': upper_bayes,
                        'MDC': MDC,
                        'ACfun_bayes': ll_out['ACfun_bayes']
                    }, lsave = flag_save)

    if flag_cost:
        mod_error_cost = MDC + obs_error
        optsed = calc_cost(srs=srs, obs=obs, Qfact=Qfact, sigma=mod_error_cost, lower_cost=lower_cost,
                            upper_cost=upper_cost, par_init=par_init, nproc=settings['nproc'], settings=settings)
        np.save(settings['expdir'] + '/cmp/optsed_costs.npy', optsed['cost'])
        np.save(settings['expdir'] + '/cmp/optsed_Qs.npy', optsed['Qs'])
        np.save(settings['expdir'] + '/cmp/optsed_accQ.npy', optsed['accQ'])
        
        write_cost(optsed, mod_error_cost, settings, data = {
            'srs': srs,
            'Qfact': Qfact,
            'samples': samples,
            'obs': obs,
            'obs_error': obs_error,
            'MDC': MDC
        }, flag_save=flag_save)

    if flag_accFOR:
        accFOR = calc_accFOR(srs=srs, obs=obs)
        np.save(settings['expdir'] + '/cmp/accFOR.npy', accFOR)
        
        write_accFOR(accFOR, ndet = np.sum(obs > 0), settings=settings, flag_save=flag_save)

    if flag_maxPSR:
        maxPSR = calc_maxPSR(srs=srs, obs=obs, method="spearman")
        np.save(settings['expdir'] + '/cmp/maxPSR.npy', maxPSR)

        write_maxPSR(maxPSR=maxPSR, settings=settings, flag_save=flag_save)
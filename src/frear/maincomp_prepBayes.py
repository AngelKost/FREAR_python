import numpy as np
import scipy.special as sc

from typing import Any, Dict

from frear.bayes import density_ffact, sampler_ffact
from frear.ll_Yee2017 import getLikes_Yee2017, preCalc_ll_noRelease as preCalc_ll_noRelease_, preCalc_ll_factors as preCalc_ll_factors_
from frear.ll_Yee2017_usinglog import getLikes_Yee2017log, preCalc_ll_noRelease as preCalc_ll_noRelease_log, preCalc_ll_factors as preCalc_ll_factors_log
from frear.modelErr import calcModErr_invGamma
from frear.srs2AC_aux import AC_ffact_bayes

def maincomp_prepbayes(settings: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Main function to prepare Bayesian likelihood and prior functions based on settings and data
    
    Parameters:
        settings (Dict[str, Any]): Configuration settings
            - likelihood (str): Type of likelihood to use
            - lllfactors (bool): Flag to indicate if likelihood factors should be pre-calculated
            - domain: Domain information for AC calculations
        data (Dict[str, Any]): dictionary containing necessary data
            - lower_bayes (np.ndarray): Lower bounds of the parameters
            - upper_bayes (np.ndarray): Upper bounds of the parameters
            - srs (np.ndarray): Source-receptor relationship data
            - srs_raw (Optional[np.ndarray]): Raw source-receptor relationship data
            - srs_spread_raw (Optional[np.ndarray]): Spread of raw source-receptor relationship
            - obs (np.ndarray): Observed data
            - obs_error (np.ndarray): Observation error data
            - MDC (np.ndarray): Minimum Detectable Concentration data
            - misc (Dict[str, Any]): Miscellaneous data such as Qfact and output frequency
    Returns:
        ll_out (Dict[str, Any]): dictionary containing likelihood and prior functions
            - density (Callable[[np.ndarray], float]): density function
            - sampler (Callable[[int], np.ndarray]): sampler function
            - ACfun_bayes (Callable[[np.ndarray], np.ndarray]): function to compute modeled AC
            - ll_logdensity (Callable[[np.ndarray], Dict[str, Any]]): log-density function
            - alphas (np.ndarray): alpha parameters for likelihood
            - betas (np.ndarray): beta parameters for likelihood
            - Ys (np.ndarray): Y parameters for likelihood
            - ll_noRelease (np.ndarray): likelihood for no release scenario
    """
    lower_bayes = data['lower_bayes']
    upper_bayes = data['upper_bayes']

    density = density_ffact(upper_bayes, lower_bayes)
    sampler = sampler_ffact(upper_bayes, lower_bayes)

    likelihood_mode = settings["likelihood"]

    if likelihood_mode == "Yee2017log":
        ll = getLikes_Yee2017log
        preCalc_ll_factors = preCalc_ll_factors_log
        preCalc_ll_noRelease = preCalc_ll_noRelease_log
    elif likelihood_mode == "Yee2017":
        ll = getLikes_Yee2017
        preCalc_ll_factors = preCalc_ll_factors_
        preCalc_ll_noRelease = preCalc_ll_noRelease_
    else:
        ll = None

    srs = data['srs']
    srs_raw = data.get('srs_raw', None)
    srs_spread_raw = data.get('srs_spread_raw', None)
    obs = data['obs']
    obs_error = data['obs_error']
    MDC = data['MDC']
    misc = data['misc']

    if likelihood_mode in ("Yee2017log", "Yee2017"):
        out = calcModErr_invGamma(settings, srs, srs_raw, srs_spread_raw)
        srs_error = None  # use mod_error_bayes instead of srs_error
        mod_error_bayes = out["sigmas"]
        alphas = out["alphas"]
        betas = out["betas"]

        if settings["likelihood"] == "Yee2017log":
            scaling = np.maximum(np.asarray(obs), 8 * np.asarray(MDC))
            mod_error_bayes = np.asarray(mod_error_bayes) * scaling[None, :]
        else:
            scaling = np.maximum(np.asarray(obs), 4 * np.asarray(MDC))
            mod_error_bayes = np.asarray(mod_error_bayes) * scaling[None, :]

        if len(obs) > 1:
            mod_error_bayes = mod_error_bayes.T
        else:
            mod_error_bayes = mod_error_bayes.reshape(len(mod_error_bayes), 1)

    else:
        raise ValueError(f"No error model available for likelihood {settings['likelihood']}")


    ACfun_bayes = AC_ffact_bayes(
        settings=settings,
        obs=obs,
        srs=srs,
        obs_error=obs_error,
        srs_error=srs_error,
        domain=settings["domain"],
        misc=misc,
        MDC=MDC,
        mod_error_bayes=mod_error_bayes
    )

    ll_out = {}
    ll_out['density'] = density
    ll_out['sampler'] = sampler
    ll_out['ACfun_bayes'] = ACfun_bayes

    if settings["likelihood"] in ("Yee2017log", "Yee2017"):

        Ys = alphas**betas * sc.gamma(betas + 0.5) / (np.sqrt(2 * np.pi) * sc.gamma(betas))

        ll_out["alphas"] = alphas
        ll_out["betas"] = betas
        ll_out["Ys"] = Ys

        ll_out["ll_noRelease"] = None
        if 'll_factors' not in ll_out.keys():
            ll_out['ll_factors'] = None

        ll_noRelease = preCalc_ll_noRelease(
            settings=settings,
            lower_bayes=lower_bayes,
            upper_bayes=upper_bayes,
            ACfun_bayes=ACfun_bayes,
            ll_out=ll_out
        )
        
        ll_out["ll_noRelease"] = ll_noRelease
    
        if settings["lllfactors"]:
            ll_factors = preCalc_ll_factors(
                settings=settings,
                lower_bayes=lower_bayes,
                upper_bayes=upper_bayes,
                ACfun_bayes=ACfun_bayes,
                ll_out=ll_out
            )
        else:
            ll_factors = np.ones(len(obs))

        ll_out["ll_factors"] = ll_factors

    def ll_logdensity(par: np.ndarray) -> Dict[str, Any]:
        ac = ACfun_bayes(par=par)
        return np.sum(ll(ac=ac, ll_out=ll_out)["likes"])
    
    ll_out['ll_logdensity'] = ll_logdensity

    return ll_out

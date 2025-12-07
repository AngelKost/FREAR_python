import numpy as np

from typing import Dict, Any

from frear.srs2AC_tools import srs_spatialInterpol, gettemporalfactor

def srs2AC_bayes_rectRelease(par: np.ndarray, srs: np.ndarray, 
                             Qfact: float, domain: Dict[str, Any]) -> np.ndarray:
    """Rectangular release bayes source-model port.

    Parameters:
        par (np.ndarray): Parameter array
        srs (np.ndarray): Source-receptor relationship data
        Qfact (float): Scaling factor for source term
        domain (Dict[str, Any]): Domain information for AC calculations
    Returns:
        AC (np.ndarray): Computed activity concentration
    """
    if srs.ndim != 4:
        raise ValueError('srs must be 4-D array')
    
    ntimes = srs.shape[0]
    nobs = srs.shape[3]

    if len(par) == 5:
        multipliers = 10**np.zeros(nobs)
    elif len(par) == 5 + nobs:
        multipliers = 10.0 ** np.asarray(par[5:])
    else:
        raise ValueError('Unknown number of parameters in par')

    M = srs_spatialInterpol(srs, par_lon=par[0], par_lat=par[1], domain=domain)

    srsfactor = gettemporalfactor(ntimes=ntimes, rstart=par[3], rstop=par[4])
    M = M * srsfactor.reshape((ntimes, 1))

    Q = 10.0 ** (par[2])
    Q /= np.sum(srsfactor)

    ac = np.sum(Q * Qfact * M, axis=0) * multipliers
    return ac

def srs2AC_bayes_rectRelease_setup(settings: Dict[str, Any], nobs: int) -> Dict[str, Any]:
    """Setup function for rectangular release bayes source model
    
    Parameters:
        settings (Dict[str, Any]): Configuration settings
            - domain (Dict[str, Any]): Domain information for AC calculations
            - Qmax (float): Maximum source term
            - Qmin (float): Minimum source term
        nobs (int): Number of observations
    Returns:
        setup (Dict[str, Any]): Setup information including parameter bounds
            - lower_bayes (List[float]): Lower bounds for bayes parameters
            - upper_bayes (List[float]): Upper bounds for bayes parameters
    """
    upper_bayes = [settings['domain']['lonmax'], settings['domain']['latmax'], np.log10(settings['Qmax']), 0.95, 1.0]
    lower_bayes = [settings['domain']['lonmin'], settings['domain']['latmin'], np.log10(settings['Qmin']), 0.0, 0.1]

    if settings['lmultipliers']:
        upper_bayes = np.concatenate([upper_bayes, np.ones(nobs)])
        lower_bayes = np.concatenate([lower_bayes, -np.ones(nobs)])

        if len(settings["parnames"]) < len(upper_bayes):
            settings["parnames"] += [f"multiplier_{i+1}" for i in range(nobs)]

    if settings['lfixedLocation']:
        lon0 = settings["trueValues"][0]
        lat0 = settings["trueValues"][1]

        upper_bayes[0: 1] = [lon0, lat0]
        lower_bayes[0: 1] = [lon0, lat0]
        
        if settings['adaptation'] != 0:
            settings['adaptation'] = 0
            print("Warning: adaptation set to 0 due to fixed location in bayes source model")
        
    if settings['lfixedTime']:
        rstart0 = settings["trueValues"][3]
        rstop0 = settings["trueValues"][4]

        upper_bayes[3: 4] = [rstart0, rstop0]
        lower_bayes[3: 4] = [rstart0, rstop0]

        if settings['adaptation'] != 0:
            settings['adaptation'] = 0
            print("Warning: adaptation set to 0 due to fixed time in bayes source model")

    settings['sourcemodel_bayes_exec'] = srs2AC_bayes_rectRelease

    return {
        'lower_bayes': lower_bayes,
        'upper_bayes': upper_bayes
    }
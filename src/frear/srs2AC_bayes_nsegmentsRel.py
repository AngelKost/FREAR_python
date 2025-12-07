import numpy as np

from typing import Dict, Any

from frear.srs2AC_tools import srs_spatialInterpol

def srs2AC_bayes_nsegmentsRel(par: np.ndarray, srs: np.ndarray, 
                              Qfact: float, domain: Dict[str, Any]) -> np.ndarray:
    """Compute activity concentration for n-segment release model
    
    Parameters:
        par (np.ndarray): Parameter array
        srs (np.ndarray): Source-receptor relationship data
        Qfact (float): Scaling factor for source term
        domain (Dict[str, Any]): Domain information for AC calculations
    Returns:
        AC (np.ndarray): Computed activity concentration
    """
    if srs.ndim != 4:
        raise ValueError("srs has wrong dimensions")
    ntimes = srs.shape[0]
    nobs = srs.shape[3]

    if len(par) == (2 + ntimes):
        multipliers = 10 ** np.zeros(nobs)
    elif len(par) == (2 + ntimes + nobs):
        multipliers = 10 ** par[2 + ntimes:]
    else:
        raise ValueError(f"Unknown number of parameters: npar is {len(par)} and nobs is {nobs}")

    M = srs_spatialInterpol(
        srs=srs,
        par_lon=par[0],
        par_lat=par[1],
        domain=domain
    )

    Qs = 10 ** par[2:2 + ntimes]

    scaled = (Qs * Qfact)[:, None] * M
    ac = np.sum(scaled, axis=0) * multipliers

    return ac

def srs2AC_bayes_nsegmentsRel_setup(settings: Dict[str, Any], nobs: int) -> Dict[str, Any]:
    """Setup function for the 'nsegmentsRel' Bayesian model
    
    Parameters:
        settings (Dict[str, Any]): Configuration settings
            - domain (Dict[str, Any]): Domain information for AC calculations
            - times (List[float]): Time intervals for segments
            - Qmax (float): Maximum source term
            - Qmin (float): Minimum source term
        nobs (int): Number of observations
    Returns:
        setup (Dict[str, Any]): Setup information including parameter bounds
            - lower_bayes (List[float]): Lower bounds for bayes parameters
            - upper_bayes (List[float]): Upper bounds for bayes parameters
    """
    ntimes = len(settings['times']) - 1
    settings['parnames'] = ['lon', 'lat'] + [f'log10_Q{i+1}' for i in range(ntimes)]
    settings['trueValues'] = [None] * len(settings['parnames'])

    upper_bayes = [settings['domain']['lonmax'], settings['domain']['latmax']] + [np.log10(settings['Qmax'])] * ntimes
    lower_bayes = [settings['domain']['lonmin'], settings['domain']['latmin']] + [np.log10(settings['Qmin'])] * ntimes

    settings['sourcemodel_bayes_exec'] = srs2AC_bayes_nsegmentsRel

    return {
        'lower_bayes': lower_bayes,
        'upper_bayes': upper_bayes
    }
import numpy as np

from typing import Any, Dict

from frear.srs2AC_tools import srs_spatialInterpol


def srs2AC_bayes_contRel(par: np.ndarray, srs: np.ndarray, Qfact: float, domain: Dict[str, Any]) -> np.ndarray:
    """Compute activity concentration for a continuous release
    
    Parameters:
        par (np.ndarray): Parameter array [longitude, latitude, log10_Q]
        srs (np.ndarray): Source-receptor relationship data
        Qfact (float): Scaling factor for source term
        domain (Dict[str, Any]): Domain information for AC calculations
    Returns:
        AC (np.ndarray): Computed activity concentration
    """
    # Spatial interpolation of SRS at source location
    M = srs_spatialInterpol(
        srs=srs,
        par_lon=par[0],
        par_lat=par[1],
        domain=domain
    )

    if isinstance(M, np.ndarray) and M.ndim > 1:
        # Average over the first axis (time or equivalent)
        M = M.mean(axis=0)

    Q = 10 ** par[2]

    return Q * Qfact * M

def srs2AC_bayes_contRel_setup(settings: Dict[str, Any], nobs: int) -> Dict[str, Any]:
    """Setup function for continuous release bayes source model
    
    Parameters:
        settings (Dict[str, Any]): Configuration settings
            - domain (Dict[str, Any]): Domain information for AC calculations
        nobs (int): Number of observations
    Returns:
        setup (Dict[str, Any]): Setup information including parameter bounds
            - lower_bayes (List[float]): Lower bounds for bayes parameters
            - upper_bayes (List[float]): Upper bounds for bayes parameters
    """
    settings['parnames'] = ['lon', 'lat', 'log10_Q']
    settings['trueValues'] = [None] * len(settings['parnames'])
    upper_bayes = [settings['domain']['lonmax'], settings['domain']['latmax'], np.log10(settings['Qmax'])]
    lower_bayes = [settings['domain']['lonmin'], settings['domain']['latmin'], np.log10(settings['Qmin'])]
    settings['sourcemodel_bayes_exec'] = srs2AC_bayes_contRel
    return {
        'lower_bayes': lower_bayes,
        'upper_bayes': upper_bayes
    }

import numpy as np

from typing import Dict, Any, Optional, Callable

def AC_ffact_bayes(settings: Dict[str, Any], obs: np.ndarray, srs: np.ndarray, 
                   obs_error: np.ndarray, srs_error: np.ndarray, 
                   domain: Dict[str, Any], misc: Dict[str, Any], 
                   MDC: np.ndarray, 
                   mod_error_bayes: Optional[np.ndarray] = None) -> Callable[[np.ndarray], Dict[str, Any]]:
    """Generate function to compute AC for Bayesian source model
    
    Parameters:
        settings (Dict[str, Any]): Configuration settings
            - sourcemodel_bayes_exec (Callable): Function to compute AC for Bayesian source model
        obs (np.ndarray): Observed data
        srs (np.ndarray): Source-receptor relationship data
        obs_error (np.ndarray): Observation error data
        srs_error (np.ndarray): Error in SRS data
        domain (Dict[str, Any]): Domain information for AC calculations
        misc (Dict[str, Any]): Miscellaneous data such as Qfact and output frequency
            - Qfact (float): Scaling factor for source term
            - outputfreq: Output frequency information
        MDC (np.ndarray): Minimum Detectable Concentration data
        mod_error_bayes (Optional[np.ndarray]): Model error for Bayesian source model
    Returns:
        inner (Callable[[np.ndarray], Dict[str, Any]]): Function that computes modeled AC
    """
    Qfact = misc["Qfact"]
    srs2AC_bayes = settings['sourcemodel_bayes_exec']

    def inner(par):
        mod = srs2AC_bayes(par=par, srs=srs, Qfact=Qfact, domain=domain)

        if mod_error_bayes is None:
            sigma_mod = srs2AC_bayes(par=par, srs=srs_error, Qfact=Qfact, domain=domain)
        else:
            sigma_mod = np.mean(mod_error_bayes, axis=1)

        return {
            'obs': obs,
            'sigma_obs': obs_error,
            'MDC': MDC,
            'mod': mod,
            'sigma_mod': sigma_mod
        }

    return inner

def AC_ffact_cost(settings: Dict[str, Any], 
                  obs: np.ndarray, srs: np.ndarray, 
                  obs_error: np.ndarray, srs_error: np.ndarray, 
                  domain: Dict[str, Any], misc: Dict[str, Any], 
                  MDC: np.ndarray, 
                  mod_error_cost: Optional[np.ndarray] = None) -> Callable[[np.ndarray], Dict[str, Any]]:
    """Generate function to compute AC for cost source model
    
    Parameters:
        settings (Dict[str, Any]): Configuration settings
            - sourcemodel_cost_exec (Callable): Function to compute AC for cost source model
        obs (np.ndarray): Observed data
        srs (np.ndarray): Source-receptor relationship data
        obs_error (np.ndarray): Observation error data
        srs_error (np.ndarray): Error in SRS data
        domain (Dict[str, Any]): Domain information for AC calculations
        misc (Dict[str, Any]): Miscellaneous data such as Qfact and output frequency
            - Qfact (float): Scaling factor for source term
            - outputfreq: Output frequency information
        MDC (np.ndarray): Minimum Detectable Concentration data
        mod_error_cost (Optional[np.ndarray]): Model error for cost source model
    Returns:
        inner (Callable[[np.ndarray], Dict[str, Any]]): Function that computes modeled AC
    """
    Qfact = misc["Qfact"]
    srs2AC_cost = settings['sourcemodel_cost_exec']

    def inner(par):
        mod = srs2AC_cost(par=par, srs=srs, Qfact=Qfact, domain=domain)
        if mod_error_cost is None:
            sigma_mod = srs2AC_cost(par=par, srs=srs_error, Qfact=Qfact, domain=domain)
        else:
            sigma_mod = np.mean(mod_error_cost, axis=1)
        return {
            'obs': obs,
            'sigma_obs': obs_error,
            'MDC': MDC,
            'mod': mod,
            'sigma_mod': sigma_mod
        }

    return inner

def getACsubset(ac: Dict[str, Any], ind: np.ndarray) -> Dict[str, Any]:
    """Get subset of AC data based on provided indices
    
    Parameters:
        ac (Dict[str, Any]): AC data dictionary
            - obs (np.ndarray): Observed data
            - sigma_obs (np.ndarray): Observation error data
            - MDC (np.ndarray): Minimum Detectable Concentration data
            - mod (np.ndarray): Modeled data
            - sigma_mod (np.ndarray): Model error data
        ind (np.ndarray): Indices for subsetting
    Returns:
        subset (Dict[str, Any]): Subset of AC data
    """
    return {
        'obs': ac['obs'][ind],
        'sigma_obs': ac['sigma_obs'][ind],
        'MDC': ac['MDC'][ind],
        'mod': ac['mod'][ind],
        'sigma_mod': ac['sigma_mod'][ind]
    }
import numpy as np

from typing import Dict, Any, Optional

from frear.srs2AC_bayes_contRel import srs2AC_bayes_contRel_setup
from frear.srs2AC_bayes_nsegmentsRel import srs2AC_bayes_nsegmentsRel_setup
from frear.srs2AC_bayes_rectRelease import srs2AC_bayes_rectRelease_setup
from frear.srs2AC_cost_contRel import srs2AC_cost_contRel_setup
from frear.srs2AC_cost_nsegmentsRel import srs2AC_cost_nsegmentsRel_setup

def maincomp_srr(settings: Dict[str, Any], srs_raw: np.ndarray, srs_spread_raw: Optional[np.ndarray], 
                 samples: Dict[str, Any], outputfreq: Any, 
                 flag_cost: bool, flag_bayes: bool) -> Dict[str, Any]:
    """
    Main function to compute the source-receptor relationship (SRR).

    Parameters:
        settings (Dict[str, Any]): Configuration settings
            - fcalcsrs (Callable): Function to calculate SRR from raw data
            - srsfact (float): Scaling factor for SRR
            - obsunitfact (float): Unit conversion factor for observations
            - obsscalingfact (float): Scaling factor for observations
            - sourcemodelcost (str): Type of source model cost to use
            - sourcemodelbayes (str): Type of source model Bayesian prior to use
        srs_raw (np.ndarray): Raw source-receptor relationship data
        srs_spread_raw (Optional[np.ndarray]): Spread of raw source-receptor relationship
        samples (Dict[str, Any]): Samples data
        outputfreq (Any): Output frequency information
        flag_cost (bool): Flag to enable cost model computation
        flag_bayes (bool): Flag to enable Bayesian model computation
    Returns:
        data (Dict[str, Any]): A dictionary containing computed SRR and related data
            - srs (np.ndarray): Processed source-receptor relationship data
            - Qfact (float): Scaling factor for SRR
            - obs (np.ndarray): Observed data
            - obs_error (np.ndarray): Observation error data
            - MDC (np.ndarray): Minimum Detectable Concentration data
            - misc (Dict[str, Any]): Miscellaneous data such as Qfact and output frequency
            - srs_error (Optional[np.ndarray]): Error in SRR data
            - cost_data (Optional[Dict[str, Any]]): Updated cost model data if flag_cost is True
            - bayes_data (Optional[Dict[str, Any]]): Updated bayesian model data if flag_bayes is True
    """

    srs = settings['fcalcsrs'](srs_raw)
    Qfact = settings['srsfact'] / (settings['obsunitfact'] * settings['obsscalingfact'])

    obs = samples['Value'] / settings['obsscalingfact']
    obs_error = samples['Uncertainty'] / settings['obsscalingfact']
    MDC = samples['MDC Value'] / settings['obsscalingfact']

    misc = {'Qfact': Qfact, 'outputfreq': outputfreq}
    srs_error = None

    if flag_cost: #lower_cost, upper_cost, par_init, settings['sourcemodelcost_exec']
        ntimes = srs.shape[0]
        if settings['sourcemodelcost'] == 'contRel':
            cost_data = srs2AC_cost_contRel_setup(settings, Qfact, ntimes)
        elif settings['sourcemodelcost'] == 'nsegmentsRel':
            cost_data = srs2AC_cost_nsegmentsRel_setup(settings, Qfact, ntimes)
        else:
            raise ValueError(f"Unknown sourcemodelcost: {settings['sourcemodelcost']}")

    nobs = srs.shape[3]

    if flag_bayes: #lower_bayes, upper_bayes, settings['sourcemodelbayes_exec']
        if settings['sourcemodelbayes'] == 'contRel':
            bayes_data = srs2AC_bayes_contRel_setup(settings, nobs)
        elif settings['sourcemodelbayes'] == 'nsegmentsRel':
            bayes_data = srs2AC_bayes_nsegmentsRel_setup(settings, nobs)
        elif settings['sourcemodelbayes'] == 'rectRelease':
            bayes_data = srs2AC_bayes_rectRelease_setup(settings, nobs)
        else:
            raise ValueError(f"Unknown sourcemodelbayes: {settings['sourcemodelbayes']}")
        
    return {
        'srs': srs,
        'Qfact': Qfact,
        'obs': obs,
        'obs_error': obs_error,
        'MDC': MDC,
        'misc': misc,
        'srs_error': srs_error,
        'cost_data': cost_data if flag_cost else None,
        'bayes_data': bayes_data if flag_bayes else None
    }

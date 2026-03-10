import numpy as np

from typing import Dict, Any, Optional

from frear.srs2AC_bayes_contRel import srs2AC_bayes_contRel_setup
from frear.srs2AC_bayes_nsegmentsRel import srs2AC_bayes_nsegmentsRel_setup
from frear.srs2AC_bayes_rectRelease import srs2AC_bayes_rectRelease_setup
from frear.srs2AC_cost_contRel import srs2AC_cost_contRel_setup
from frear.srs2AC_cost_nsegmentsRel import srs2AC_cost_nsegmentsRel_setup

def maincomp_srr(settings: Dict[str, Any], srs_raw: np.ndarray, 
                 samples: Dict[str, Any], outputfreq: Any) -> Dict[str, Any]:
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
        samples (Dict[str, Any]): Samples data
        outputfreq (Any): Output frequency information
    Returns:
        data (Dict[str, Any]): A dictionary containing computed SRR and related data
            - srs (np.ndarray): Processed source-receptor relationship data
            - Qfact (float): Scaling factor for SRR
            - obs (np.ndarray): Observed data
            - obs_error (np.ndarray): Observation error data
            - MDC (np.ndarray): Minimum Detectable Concentration data
            - misc (Dict[str, Any]): Miscellaneous data such as Qfact and output frequency
            - srs_error (Optional[np.ndarray]): Error in SRR data
    """

    srs = settings['fcalcsrs'](srs_raw)
    Qfact = settings['srsfact'] / (settings['obsunitfact'] * settings['obsscalingfact'])

    obs = samples['Value'] / settings['obsscalingfact']
    obs_error = samples['Uncertainty'] / settings['obsscalingfact']
    MDC = samples['MDC Value'] / settings['obsscalingfact']

    misc = {'Qfact': Qfact, 'outputfreq': outputfreq}
    srs_error = None
        
    return {
        'srs': srs,
        'Qfact': Qfact,
        'obs': obs,
        'obs_error': obs_error,
        'MDC': MDC,
        'misc': misc,
        'srs_error': srs_error
    }

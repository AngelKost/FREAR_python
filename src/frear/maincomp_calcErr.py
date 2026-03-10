import numpy as np

from typing import Dict, Any

from frear.modelErr import calcModErr_invGamma

def maincomp_calcErr(settings: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Function to calculate likelihood parameters based on the source-receptor relationship (SRR) and the specified likelihood model
    
    Parameters:
        settings (Dict[str, Any]): Configuration settings
            - likelihood (str): Type of likelihood to use
        data (Dict[str, Any]): dictionary containing necessary data
            - srs (np.ndarray): Source-receptor relationship data
            - srs_raw (Optional[np.ndarray]): Raw source-receptor relationship data
            - srs_spread_raw (Optional[np.ndarray]): Spread of raw source-receptor relationship
    Returns:
        out (Dict[str, Any]): dictionary containing likelihood parameters
            - alphas (np.ndarray): alpha parameters for likelihood
            - betas (np.ndarray): beta parameters for likelihood
            - sigmas (np.ndarray): sigma parameters for likelihood
    """
    srs = data['srs']
    srs_raw = data.get('srs_raw', None)
    srs_spread_raw = data.get('srs_spread_raw', None)

    likelihood_mode = settings["likelihood"]

    if likelihood_mode in ("Yee2017log", "Yee2017"):
        out = calcModErr_invGamma(settings, srs, srs_raw, srs_spread_raw)
    else:
        raise ValueError(f"No error model available for likelihood {settings['likelihood']}")
    
    return out
    

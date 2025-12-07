import numpy as np

from typing import Dict, Any, Optional

from frear.domain import lon2ix, lat2iy

def srs2AC_cost_contRel(par: np.ndarray, Qfact: float, M: Optional[np.ndarray], 
                        srs: Optional[np.ndarray] = None,  
                        lon: Optional[float] = None, lat: Optional[float] = None, 
                        ix: Optional[int] = None, iy: Optional[int] = None,
                        domain: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Compute cost for continuous release source model
    
    Parameters:
        par (np.ndarray): Parameter array
        Qfact (float): Scaling factor for source term
        M (Optional[np.ndarray]): Pre-extracted SRS data at source location
        srs (Optional[np.ndarray]): Source-receptor relationship data
        lon (Optional[float]): Longitude of source location
        lat (Optional[float]): Latitude of source location
        ix (Optional[int]): Index in x-direction for source location
        iy (Optional[int]): Index in y-direction for source location
        domain (Optional[Dict[str, Any]]): Domain information for AC calculations
    Returns:
        AC (np.ndarray): Computed activity concentration
    """
    if M is None:
        if ix is None:
            ix = lon2ix(lon=lon, lon0=domain['lonmin'], dx=domain['dx'])
        if iy is None:
            iy = lat2iy(lat=lat, lat0=domain['latmin'], dy=domain['dy'])
        M = srs[:, ix, iy, :]
    if isinstance(M, np.ndarray) and M.ndim >= 2:
        M = np.mean(M, axis=0)

    return par * Qfact * M

def srs2AC_cost_contRel_setup(settings: Dict[str, Any], Qfact: float, ntimes: int) -> Dict[str, Any]:
    """Setup function for continuous release source model cost computation
    
    Parameters:
        settings (Dict[str, Any]): Configuration settings
            - Qmax (float): Maximum source term
            - Qmin (float): Minimum source term
        Qfact (float): Scaling factor for source term
        ntimes (int): Number of time steps (placeholder for compatibility)
    Returns:
        setup (Dict[str, Any]): Setup information including parameter bounds
            - lower_cost (float): Lower bound for cost parameter
            - upper_cost (float): Upper bound for cost parameter
            - par_init (float): Initial guess for cost parameter
    """
    lower_cost = settings['Qmin'] * Qfact
    upper_cost = settings['Qmax'] * Qfact
    par_init = 10 ** np.mean([np.log10(lower_cost), np.log10(upper_cost)])
    settings['sourcemodelcost_exec'] = srs2AC_cost_contRel

    return {
        'lower_cost': lower_cost,
        'upper_cost': upper_cost,
        'par_init': par_init
    }
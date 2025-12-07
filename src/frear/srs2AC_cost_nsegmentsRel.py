import numpy as np

from typing import Dict, Any, Optional

from frear.domain import lon2ix, lat2iy

def srs2AC_cost_nsegmentsRel(par: np.ndarray, M: Optional[np.ndarray], Qfact: float, 
                             srs: Optional[np.ndarray] = None, 
                             lon: Optional[float] = None, lat: Optional[float] = None, 
                             ix: Optional[int] = None, iy: Optional[int] = None,
                             domain: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Compute AC for n-segments cost source model
    
    Parameters:
        par (np.ndarray): Parameter array of length ntimes
        M (Optional[np.ndarray]): Pre-extracted SRS data at source location
        Qfact (float): Scaling factor for source term
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
    if M.shape[0] != len(par):
        raise ValueError(f"Number of release time steps ({len(par)}) does not match M rows ({M.shape[0]})")
    
    # par: (ntimes,), M: (ntimes, nsegments) -> par[:, None] * M: (ntimes, nsegments)
    # Then sum over time axis to get nsegments output values
    # R equivalent: colSums(par * Qfact * M)
    return np.sum(par[:, None] * Qfact * M, axis=0)

def srs2AC_cost_nsegmentsRel_setup(settings: Dict[str, Any], Qfact: float, ntimes: int) -> Dict[str, Any]:
    """Setup function for n-segments release cost source model AC computation
    
    Parameters:
        settings (Dict[str, Any]): Configuration settings
            - Qmax (float): Maximum source term
            - Qmin (float): Minimum source term
        Qfact (float): Scaling factor for source term
        ntimes (int): Number of time steps / segments
    Returns:
        setup (Dict[str, Any]): Setup information including parameter bounds
            - lower_cost (np.ndarray): Lower bounds for cost parameters
            - upper_cost (np.ndarray): Upper bounds for cost parameters
            - par_init (np.ndarray): Initial guess for cost parameters
    """
    lower_cost = np.full(ntimes, settings["Qmin"] * Qfact)
    upper_cost = np.full(ntimes, settings["Qmax"] * Qfact)

    par_init = 10 ** np.mean(
        np.column_stack([np.log10(lower_cost), np.log10(upper_cost)]),
        axis=1
    )
    settings['sourcemodelcost_exec'] = srs2AC_cost_nsegmentsRel

    return {
        'lower_cost': lower_cost,
        'upper_cost': upper_cost,
        'par_init': par_init
    }
import numpy as np

from typing import Dict, Any

from frear.domain import lon2ix, lat2iy

def lonlat2weightedixiy(lon: float, lat: float, lon0: float, lat0: float, dx: float, dy: float) -> np.ndarray:
    """Convert lon, lat to ix, iy, weights for bilinear interpolation"""

    left = int(np.floor((lon - lon0) / dx))
    bottom = int(np.floor((lat - lat0) / dy))
    
    u = (lon - (lon0 + left * dx)) / dx
    v = (lat - (lat0 + bottom * dy)) / dy
    
    indices = np.array([
        [left,     bottom],
        [left,     bottom + 1],
        [left + 1, bottom],
        [left + 1, bottom + 1]
    ])
    
    weights = np.array([
        (1 - u) * (1 - v),
        (1 - u) * v,
        u * (1 - v),
        u * v
    ])
    
    return np.column_stack((indices[:, 0], indices[:, 1], weights))

def gettemporalfactor(ntimes: int, rstart: float, rstop: float) -> np.ndarray:
    """Get temporal factor for rectangular release between rstart and rstop (fractions of total time)"""
    temporalfactor = np.zeros(ntimes, dtype=float)
    zstart = rstart * ntimes
    zstop = zstart + rstop * (ntimes - zstart)
    istart = int(max(np.ceil(zstart), 1))
    istop = int(max(np.ceil(zstop), istart))
    if istart == istop:
        temporalfactor[istart - 1] = zstop - zstart
    else:
        temporalfactor[istart - 1:istop] = 1.0
        temporalfactor[istart - 1] = istart - zstart
        temporalfactor[istop - 1] = 1 - (istop - zstop)
    return temporalfactor

def srs_spatialInterpol(srs: np.ndarray, 
                        par_lon: float, par_lat: float, 
                        domain: Dict[str, Any]) -> np.ndarray:
    """Calculate spatial interpolation of SRS at given lon, lat"""
    weighted = lonlat2weightedixiy(
        lon=par_lon,
        lat=par_lat,
        lon0=domain['lonmin'],
        lat0=domain['latmin'],
        dx=domain['dx'],
        dy=domain['dy']
    )

    maxix = lon2ix(domain['lonmax'], domain['lonmin'], domain['dx']) - 1
    maxiy = lat2iy(domain['latmax'], domain['latmin'], domain['dy']) - 1

    weighted[weighted[:, 0] == (maxix + 1), 0] = maxix
    weighted[weighted[:, 1] == (maxiy + 1), 1] = maxiy

    ntimes = srs.shape[0]
    nobs   = srs.shape[3]
    M = np.zeros((ntimes, nobs))

    for ix, iy, weight in weighted:
        ix = int(ix)
        iy = int(iy)
        M += srs[:, ix - 1, iy - 1, :] * weight

    return M

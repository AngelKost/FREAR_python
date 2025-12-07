import numpy as np

from typing import Dict, Any

def ix2lon(ix: int, lon0: float, dx: float) -> float:
    """Convert index ix to longitude of cell centre"""
    return lon0 + (ix + 0.5) * dx

def iy2lat(iy: int, lat0: float, dy: float) -> float:
    """Convert index iy to latitude of cell centre"""
    return lat0 + (iy + 0.5) * dy

def lon2ix(lon, lon0: float, dx: float):
    """Convert longitude(s) to ix index"""
    lon_arr = np.asarray(lon)
    ix = np.floor((lon_arr - lon0) / dx + 1e-10).astype(int)
    if np.isscalar(ix) or len(ix) == 1:
        return int(ix)
    return ix

def lat2iy(lat, lat0: float, dy: float):
    """Convert latitude(s) to iy index"""
    lat_arr = np.asarray(lat)
    iy = np.floor((lat_arr - lat0) / dy + 1e-10).astype(int)
    if np.isscalar(iy) or len(iy) == 1:
        return int(iy)
    return iy

def domain_add_nxny(domain: Dict[str, Any]) -> Dict[str, Any]:
    """Add nx and ny (number of grid cells) to a domain dict"""
    if domain['lonmin'] > domain['lonmax']:
        domain['nx'] = round((domain['lonmax'] - domain['lonmin'] + 360) / domain['dx'] + 1)
    else:
        domain['nx'] = round((domain['lonmax'] - domain['lonmin']) / domain['dx'] + 1)
    domain['ny'] = round((domain['latmax'] - domain['latmin']) / domain['dy'] + 1)
    return domain

def make_domain(lonmin: float, latmin: float, lonmax: float, latmax: float, dx: float, dy: float) -> Dict[str, Any]:
    """Create domain dict"""
    domain = {'lonmin': lonmin, 'latmin': latmin, 'lonmax': lonmax, 'latmax': latmax, 'dx': dx, 'dy': dy}
    return domain_add_nxny(domain)

def FPheader2domain(header: Dict[str, Any]) -> Dict[str, Any]: #FIXME: header columns
    """Convert FREAR Python FP header dict to domain dict"""
    domain = {
        'lonmin': header['outlon0'],
        'latmin': header['outlat0'],
        'lonmax': header['outlon0'] + (header['numxgrid'] - 1) * header['dxout'],
        'latmax': header['outlat0'] + (header['numygrid'] - 1) * header['dyout'],
        'dx': header['dxout'],
        'dy': header['dyout']
    }
    return domain_add_nxny(domain)

def convert_domain(matrix1: np.ndarray, domain1: Dict[str, Any], domain2: Dict[str, Any]) -> np.ndarray:
    """Convert matrix1 defined on domain1 to matrix2 on domain2 as average"""
    if domain1 == domain2:
        return matrix1.copy()
    
    lonmin = max(domain1['lonmin'], domain2['lonmin'])
    latmin = max(domain1['latmin'], domain2['latmin'])
    lonmax = min(domain1['lonmax'] + domain1['dx'], domain2['lonmax'] + domain2['dx'])
    latmax = min(domain1['latmax'] + domain1['dy'], domain2['latmax'] + domain2['dy'])

    if lonmin > lonmax or latmin > latmax:
        raise ValueError("Domains do not overlap")
    
    if domain2['lonmin'] < domain1['lonmin']:
        raise ValueError("Requested domain is smaller than source domain in lonmin")
    if domain2['lonmax'] + domain2['dx'] > domain1['lonmax'] + domain1['dx']:
        raise ValueError("Requested domain is greater than source domain in lonmax")
    if domain2['latmin'] < domain1['latmin']:
        raise ValueError("Requested domain is smaller than source domain in latmin")
    if domain2['latmax'] + domain2['dy'] > domain1['latmax'] + domain1['dy']:
        raise ValueError("Requested domain is greater than source domain in latmax")
    
    jx1 = lon2ix(lon=lonmin, lon0=domain1['lonmin'], dx=domain1['dx'])
    jy1 = lat2iy(lat=latmin, lat0=domain1['latmin'], dy=domain1['dy'])
    jx2 = lon2ix(lon=lonmax - domain1['dx'], lon0=domain1['lonmin'], dx=domain1['dx'])
    jy2 = lat2iy(lat=latmax - domain1['dy'], lat0=domain1['latmin'], dy=domain1['dy'])

    ix1 = lon2ix(lon=lonmin, lon0=domain2['lonmin'], dx=domain2['dx'])
    iy1 = lat2iy(lat=latmin, lat0=domain2['latmin'], dy=domain2['dy'])
    ix2 = lon2ix(lon=lonmax - domain2['dx'], lon0=domain2['lonmin'], dx=domain2['dx'])
    iy2 = lat2iy(lat=latmax - domain2['dy'], lat0=domain2['latmin'], dy=domain2['dy'])

    matrix2 = np.zeros((domain2['ny'], domain2['nx']), dtype=matrix1.dtype)

    if round(domain1['dx'], 6) == round(domain2['dx'], 6) and round(domain1['dy'], 6) == round(domain2['dy'], 6):
        matrix2[ix1:ix2+1, iy1:iy2+1] = matrix1[jx1:jx2+1, jy1:jy2+1]
    else:
        mx = domain2['dx'] / domain1['dx']
        my = domain2['dy'] / domain1['dy']
        for ix in range(ix1, ix2 + 1):
            for iy in range(iy1, iy2 + 1):
                jx_start = int((ix - ix1) * mx + jx1)
                jx_end = int((ix - ix1 + 1) * mx + jx1)
                jy_start = int((iy - iy1) * my + jy1)
                jy_end = int((iy - iy1 + 1) * my + jy1)
                matrix2[ix, iy] = np.mean(matrix1[jx_start:jx_end, jy_start:jy_end])
    return matrix2

def smooth2D(arr: np.ndarray, n: int) -> np.ndarray:
    """Apply 2D moving average smoothing"""
    nx, ny = arr.shape
    arr_smoothed = np.zeros_like(arr)
    for ix in range(nx):
        for iy in range(ny):
            ix_min = max(0, ix - n)
            ix_max = min(nx, ix + n + 1)
            iy_min = max(0, iy - n)
            iy_max = min(ny, iy + n + 1)
            arr_smoothed[ix, iy] = np.mean(arr[ix_min:ix_max, iy_min:iy_max])
    return arr_smoothed
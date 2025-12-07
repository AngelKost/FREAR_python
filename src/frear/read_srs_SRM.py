import pandas as pd
import numpy as np
import os
import pickle
import gzip

from typing import Dict, Any, Optional, List, Union
from datetime import datetime

def opener(file: str):
    """Open a file, handling gzip if necessary
    
    Parameters:
        file (str): Path to the file to open
    Returns:
        file object: Opened file object
    """
    if file.endswith('.gz'):
        return gzip.open(file, 'rt')
    else:
        return open(file, 'r+')

def readSRM_header(srmfile: str) -> Dict[str, Any]:
    """
    Reads the header information from a SRM file.

    Parameters:
        srmfile (str): The path to the SRM file.
    Returns:
        header (Dict[str, Any]): A dictionary containing the header information.
    """
    expected_columns = [
        "statlon", "statlat", "sampStartDate", "sampStartHour",
        "sampEndDate", "sampEndHour", "mass", "nsimhours",
        "outputfreq", "aveTime", "dx", "dy", "stat"
    ]

    with opener(srmfile) as f:
        line = f.readline().strip()

    values = line.split()

    if len(values) != len(expected_columns):
        raise ValueError(f"Expected {len(expected_columns)} columns, got {len(values)}")
    
    header = dict(zip(expected_columns, values))

    header['dx'] = float(header['dx'])
    header['dy'] = float(header['dy'])

    return header

def get_srs_fixedperiod(srm_fixedperiod: pd.DataFrame, srm_spread_fixedperiod: Optional[pd.DataFrame],
                        nx: int, ny: int) -> Dict[str, Any]:
    """Get srs and srs_spread for a fixed period from SRM data
    
    Parameters:
        srm_fixedperiod (pd.DataFrame): DataFrame with SRM data for the fixed period
        srm_spread_fixedperiod (Optional[pd.DataFrame]): DataFrame with SRM spread data for the fixed period
        nx (int): Number of grid points in x direction
        ny (int): Number of grid points in y direction
    Returns:
        out (Dict[str, Any]): Dictionary with srs data
            - srs_fixedPeriod (np.ndarray): 2D array of srs values [nx, ny]
            - srs_spread_fixedPeriod (Optional[np.ndarray]): 2D array of srs spread values
    """
    has_spread = srm_spread_fixedperiod is not None

    ixiys = srm_fixedperiod[["ix", "iy"]].drop_duplicates()

    srs_fixedPeriod = np.zeros((nx, ny))
    srs_spread_fixedPeriod = np.zeros((nx, ny)) if has_spread else None

    srm_fixedperiod = srm_fixedperiod.sort_values(["ix", "iy"]).reset_index(drop=True)

    if has_spread:
        srm_spread_fixedperiod = (
            srm_spread_fixedperiod.sort_values(["ix", "iy"]).reset_index(drop=True)
        )

    ixiys = ixiys.sort_values(["ix", "iy"]).reset_index(drop=True)
    ix_values = srm_fixedperiod["ix"].values
    change_points = np.where(np.diff(ix_values) != 0)[0] + 1
    ix_intervals = list(change_points) + [len(srm_fixedperiod)]

    ix_start = 0
    for ix_end in ix_intervals:
        ix = int(srm_fixedperiod.iloc[ix_start]["ix"])
        # Extract block corresponding to this ix
        srm_fixed_ix = srm_fixedperiod.iloc[ix_start:ix_end]

        if has_spread:
            srm_spread_fixed_ix = srm_spread_fixedperiod.iloc[ix_start:ix_end]

        iys = ixiys[ixiys["ix"] == ix]["iy"]

        for iy in iys:
            srm_fixed_ixiy = srm_fixed_ix[srm_fixed_ix["iy"] == iy]["conc"].to_numpy()

            if has_spread:
                srm_spread_fixed_ixiy = srm_spread_fixed_ix.loc[srm_spread_fixed_ix["iy"] == iy]["spread"].to_numpy()

            ix0 = ix - 1
            iy0 = iy - 1
            srs_fixedPeriod[ix0, iy0] = np.mean(srm_fixed_ixiy)
            if has_spread:
                srs_spread_fixedPeriod[ix0, iy0] = np.mean(srm_spread_fixed_ixiy)

            if np.isnan(srs_fixedPeriod[ix0, iy0]):
                raise ValueError(f"Nan produced at (ix, iy) = ({ix}, {iy})")

        ix_start = ix_end

    return {
        "srs_fixedPeriod": srs_fixedPeriod,
        "srs_spread_fixedPeriod": srs_spread_fixedPeriod,
    }

def read_srm(srmfile: str, times: Optional[List[datetime]] = None, 
             domain: Optional[Dict[str, Any]] = None, 
             srsfact: Optional[float] = None, halflife: Optional[float] = None) -> Dict[str, Any]:
    """Read a SRM file and return srs and srs_spread arrays along with metadata.

    Parameters:
        srmfile (str): Path to the SRM file.
        times (Optional[List[datetime]]): List of datetime objects representing time steps.
        domain (Optional[Dict[str, Any]]): Domain dictionary
        srsfact (Optional[float]): Scaling factor for SRS values
        halflife (Optional[float]): Half-life for decay adjustment
    Returns:
        data (Dict[str, Any]): A dictionary containing
            - header (Dict[str, Any]): Header information from the SRM file
            - srs (np.ndarray): 3D array of srs values [ntimes, nx, ny]
            - srs_spread (Optional[np.ndarray]): 3D array of srs spread values [ntimes, nx, ny]
            - times (List[datetime]): List of datetime objects representing time steps
            - srsfact (Optional[float]): Scaling factor used for SRS values
    """
    if times is not None and times != sorted(times):
        raise ValueError("readSRM: input 'times' is not sorted!")
    header = readSRM_header(srmfile)

    if domain is not None:
        if domain['dx'] < header['dx'] or domain['dy'] < header['dy']:
            raise ValueError("readSRM: domain dx/dy smaller than SRM dx/dy!")
        else:
            if domain['dx'] > header['dx'] or domain['dy'] > header['dy']:
                print(f"domain dx/dy larger than SRM dx/dy -> converting ({header['dx']}, {header['dy']}) to ({domain['dx']}, {domain['dy']})")

    if times is not None and len(times) < 2:
        raise ValueError("readSRM: input 'times' must have at least two points (start and end)")
    
    srm = pd.read_csv(srmfile, skiprows=1, header=None, sep=r"\s+")
    has_spread = (srm.shape[1] == 5)
    columns = ["lat", "lon", "itime", "conc"]
    if has_spread:
        columns.append("spread")
    srm.columns = columns

    simstart = datetime.strptime(f"{header['sampEndDate']}-{header['sampEndHour']}", "%Y%m%d-%H")
    simend = simstart - pd.Timedelta(hours=int(header['nsimhours']))
    srm['date'] = simstart - srm['itime'] * int(header['outputfreq']) * pd.Timedelta(hours=1)

    if times is None:
        freq = pd.Timedelta(hours=int(header['outputfreq']))
        times = list(pd.date_range(start=simend, end=simstart, freq=freq).to_pydatetime())
        times = times[:-1]

    srm.loc[srm.lon > 180, 'lon'] -= 360
    
    if domain is not None:
        if domain['lonmin'] > domain['lonmax']:
            srm = srm[((srm.lon >= domain['lonmin']) | (srm.lon <= domain['lonmax'])) &
                      (srm.lat >= domain['latmin']) & (srm.lat <= domain['latmax'])]
        else:
            srm = srm[(srm.lon >= domain['lonmin']) & (srm.lon <= domain['lonmax']) &
                      (srm.lat >= domain['latmin']) & (srm.lat <= domain['latmax'])]

        srm['ix'] = ((srm['lon'] - domain['lonmin']) / domain['dx'] + 1).astype(int)
        srm['iy'] = ((srm['lat'] - domain['latmin']) / domain['dy'] + 1).astype(int)
        srm.loc[srm.ix < 0, 'ix'] += int(360 / domain['dx'])
        nx = domain['nx']
        ny = domain['ny']
    else:
        srm['ix'] = ((srm['lon'] - srm['lon'].min()) / header['dx'] + 1).astype(int)
        srm['iy'] = ((srm['lat'] - srm['lat'].min()) / header['dy'] + 1).astype(int)
        nx = srm['ix'].max()
        ny = srm['iy'].max()

    if halflife is not None:
        srm['conc'] *= 2 ** (-(srm['itime'] * int(header['outputfreq']) * 3600) / halflife)
    
    ntimes = len(times) - 1
    srs = np.zeros((ntimes, nx, ny))
    srs_spread = np.zeros((ntimes, nx, ny)) if has_spread else None

    timestart = times[0]
    for itime in range(1, len(times)):
        print(f"Processing time interval {itime} / {len(times)-1}... \n", end="")
        timestop = times[itime]
        srm_fixedPeriod = srm[(srm.date > timestart) & (srm.date <= timestop)][["conc", "ix", "iy"]]
        srm_spread_fixedPeriod = None
        if has_spread:
            srm_spread_fixedPeriod = srm[(srm.date > timestart) & (srm.date <= timestop)][["spread", "ix", "iy"]]
        out_srs = get_srs_fixedperiod(srm_fixedPeriod, srm_spread_fixedPeriod, nx, ny)
        srs[itime - 1, :, :] = out_srs["srs_fixedPeriod"]
        if has_spread:
            srs_spread[itime - 1, :, :] = out_srs["srs_spread_fixedPeriod"]
        timestart = timestop

    if srsfact is None:
        srsfact = 1 / float(header['mass'])
    else:
        srs = srs / (srsfact * float(header['mass']))
        if has_spread:
            srs_spread = srs_spread / (srsfact * float(header['mass']))
    
    return {
        "header": header,
        "srs": srs,
        "srs_spread": srs_spread,
        "times": times,
        "srsfact": srsfact
    }

def read_all_srm(srmpaths: Union[Dict[Any, List[np.ndarray]], List[np.ndarray]], 
                 times: Optional[List[datetime]] = None,
                 domain: Optional[Dict[str, Any]] = None, srsfact: Optional[float] = None,
                 halflife: Optional[float] = None, nproc: int = 1,
                 srm_processed: Optional[Union[Dict[Any, List[str]], List[str]]] = None) -> Dict[str, Any]:
    """Read all SRM files for multiple members

    Parameters:
        srmpaths (Union[Dict[Any, List[np.ndarray]], List[np.ndarray]]): Paths to SRM files, either as a dictionary mapping member identifiers to lists of file paths or as a list of file paths.
        times (Optional[List[datetime]]): List of datetime objects representing time steps.
        domain (Optional[Dict[str, Any]]): Domain dictionary
        srsfact (Optional[float]): Scaling factor for SRS values
        halflife (Optional[float]): Half-life for decay adjustment
        nproc (int): Number of parallel processes to use
        srm_processed (Optional[Union[Dict[Any, List[str]], List[str]]]): Optional dictionary mapping member identifiers to lists of processed SRM file paths
    Returns:
        data (Dict[str, Any]): A dictionary containing
            - srs (Dict[str, np.ndarray]): Dictionary mapping member identifiers to 3D arrays of SRS values
            - srs_spread (Dict[str, Optional[np.ndarray]]): Dictionary mapping member identifiers to 3D arrays of SRS spread values or None
            - outputfreq (int): Output frequency in seconds
            - header (Dict[str, Any]): Header information from the first SRM file
    """
    srs = {}
    srs_spread = {}
    members = None if not isinstance(srmpaths, dict) else list(srmpaths.keys())

    if members is None:
        members = [0]

    for member in members:
        print(f"INFO: Reading {len(srmpaths[member])} srs files for member {member} ({members.index(member)+1} of {len(members)})")
        
        srs_single_member = []
        srs_spread_single_member = []

        if nproc <= 1:
            for ipath in range(len(srmpaths[member])):
                print(f"{ipath + 1}... \n", end="")
                paths = srmpaths[member][ipath] # List of paths for this entry (one or many)
                proc_paths = srm_processed[member][ipath] if srm_processed is not None else None

                if srm_processed is not None and os.path.exists(proc_paths[0]):
                    out = pickle.load(open(proc_paths[0], 'rb'))
                else:
                    out = read_srm(srmfile=paths[0], times=times, domain=domain,
                                   srsfact=srsfact, halflife=halflife)
                    if srm_processed is not None:
                        pickle.dump(out, open(proc_paths[0], 'wb'))
                out_srs = out['srs']
                out_srs_spread = out['srs_spread']
                if len(paths) > 1:
                    for jpath in range(1, len(paths)):
                        if srm_processed is not None and os.path.exists(proc_paths[jpath]):
                            out = pickle.load(open(proc_paths[jpath], 'rb'))
                        else:
                            out = read_srm(srmfile=paths[jpath], times=times, domain=domain,
                                           srsfact=srsfact, halflife=halflife)
                            if srm_processed is not None:
                                pickle.dump(out, open(proc_paths[jpath], 'wb'))
                        out_srs += out['srs']
                        if out['srs_spread'] is not None:
                            if out_srs_spread is None:
                                out_srs_spread = out['srs_spread']
                            else:
                                out_srs_spread += out['srs_spread']
                srs_single_member.append(out_srs / len(paths))
                if out_srs_spread is not None:
                    srs_spread_single_member.append(out_srs_spread / len(paths))
                else:
                    srs_spread_single_member = None
            print("done")
        else:
            raise NotImplementedError("Parallel processing not implemented") #FIXME: implement parallel reading
    
        ntimes = len(srs_single_member[0][:, 0, 0])
        nx = len(srs_single_member[0][0, :, 0])
        ny = len(srs_single_member[0][0, 0, :])
        nsamples = len(srs_single_member)
        srs_single_member = np.stack(srs_single_member, axis=-1)
        srs[member] = srs_single_member
        if srs_spread_single_member is not None:
            srs_spread_single_member = np.stack(srs_spread_single_member, axis=-1)
            srs_spread[member] = srs_spread_single_member
        else:
            srs_spread = None
    header = readSRM_header(srmpaths[members[0]][0][0])
    outputfreq = abs(int(header['outputfreq'])) * 3600

    return {
        "srs": srs,
        "srs_spread": srs_spread,
        "outputfreq": outputfreq,
        "header": header
        }


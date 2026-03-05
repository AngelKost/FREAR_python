"""
Data reader for combination of .nc and .csv files (instead of set of .txt.gz files).
Replicates data conversion and maincomp_readdata pipeline.
"""

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import os
import gc

from datetime import datetime
from typing import Dict, List, Optional, Any
from scipy.ndimage import gaussian_filter

from frear.domain import domain_add_nxny

def read_observations_csv(csv_path: str) -> pd.DataFrame:
    """Read observations from CSV file"""
    df = pd.read_csv(csv_path)
    
    # Ensure date column is datetime-type if it exists
    if 'COLLECT_STOP' in df.columns:
        df['COLLECT_STOP'] = pd.to_datetime(df['COLLECT_STOP'])
    
    return df.reset_index(drop=True)


def infer_station_names(obs_df: pd.DataFrame, station_coords: List[tuple], station_names: List[str]) -> pd.DataFrame:
    """
    Assign station names to observations based on nearest coordinates.

    Parameters:
        obs_df (pd.DataFrame): Observation DataFrame with LAT, LON columns
        station_coords (list of tuples): [(lat1, lon1), (lat2, lon2), ...]
        station_names (list of str): Names corresponding to station_coords

    Returns:
        pd.DataFrame: obs_df with new column 'Station'
    """
    obs_df = obs_df.copy()
    obs_lats = obs_df['LAT'].values
    obs_lons = obs_df['LON'].values
    
    station_array = np.array(station_coords)  # shape (n_stations, 2)
    
    assigned_stations = []
    for lat, lon in zip(obs_lats, obs_lons):
        dists = (station_array[:,0] - lat)**2 + (station_array[:,1] - lon)**2
        nearest_idx = np.argmin(dists)
        assigned_stations.append(station_names[nearest_idx])
    
    obs_df['Station'] = assigned_stations
    return obs_df


def delete_duplicate_observations(obs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for duplicate observations in the DataFrame.

    Parameters:
        obs_df (pd.DataFrame): Observation DataFrame

    Returns:
        pd.DataFrame: DataFrame with duplicate observations replaced by mean values
    """
    def mean_observation(group):
        if len(group) > 1:
            print(f"Found {len(group)} duplicate observations for Station {group.name[0]} at {group.name[1]}")
        row = group.iloc[0].copy()
        row['AVE_ACTIV'] = group['AVE_ACTIV'].mean()
        row['AVE_ACTIV_ERR'] = group['AVE_ACTIV_ERR'].mean()
        return row

    obs_df_no_duplicates = obs_df.groupby(['Station', 'COLLECT_STOP'], as_index=False, sort=False).apply(mean_observation).reset_index(drop=True)

    return obs_df_no_duplicates

def apply_gaussian_smoothing(srs_array: np.ndarray, sigma: float) -> None:
    """Apply Gaussian smoothing to the SRS array (inplace modification)"""
    gaussian_filter(srs_array, sigma=(0, sigma, sigma), output=srs_array) # No smoothing across time dimension, only spatial dimensions

def maincomp_readdata(
    srm_nc_path: str,
    concentrations_csv_path: str,
    times: List[datetime],
    domain: Dict[str, float],
    lats: np.ndarray,
    lons: np.ndarray,
    srsfact: float,
    mass_factor: float,
    mdc_value: float,
    srs_storage: str = 'memmap',
    srs_memmap_path: Optional[str] = None,
    gaussian_smoothing: bool = False,
    gaussian_sigma: Optional[float] = None,
    regrid_method: str = 'bilinear'
) -> Dict[str, Any]:
    """
    Main function to read and process data from .nc and .csv files, replicating maincomp_readdata pipeline.
    
    Parameters:
        srm_nc_path (str): Path to SRM.nc file
        concentrations_csv_path (str): Path to concentrations CSV file
        times (List[datetime]): List of datetime objects for time dimension (settings['times'])
        domain (Dict[str, float]): Domain settings with keys 'lonmin', 'latmin', 'lonmax', 'latmax', 'dx', 'dy'
        lats (np.ndarray): 1D array of stations' latitudes
        lons (np.ndarray): 1D array of stations' longitudes
        srsfact (float): Scaling factor for SRS values
        mass_factor (float): Mass factor for scaling SRS values
        mdc_value (float): Minimum detectable concentration value
        srs_storage (str): Storage method for SRS data ('memmap' or 'RAM')
        srs_memmap_path (Optional[str]): Path for memory-mapped SRS data if srs_storage is 'memmap'
        gaussian_smoothing (bool): Whether to apply Gaussian smoothing to SRS data
        gaussian_sigma (Optional[float]): Standard deviation for Gaussian smoothing filter (if enabled)
        regrid_method (str): Method for regridding SRM data to target grid (default 'bilinear')
        
    Returns:
        data (Dict[str, Any]): 
        Dictionary containing:
            - 'samples': pd.DataFrame of observations with columns ['LAT', 'LON', 'COLLECT_STOP', 'CONC']
            - 'srs': np.ndarray of shape (sensor_i, ts, td, ny, nx) with processed SRS values
            - 'outputfreq': int, frequency of output time steps (in seconds)
            - 'srs_spread': None (placeholder for compatibility with original reader output, SRM.nc is assumed to have no spread dimension)
    """

    # Generate station names and indices based on lat/lon for entity inference
    station_names = [f'STA_{lat:.3f}_{lon:.3f}' for lat, lon in zip(lats, lons)]
    sensor_indices = {station_names[idx]: idx for idx in range(len(station_names))}

    # Read observations from CSV and build samples DataFrame
    obs_df = read_observations_csv(concentrations_csv_path)
    obs_df = infer_station_names(obs_df, list(zip(lats, lons)), station_names)
    obs_df = delete_duplicate_observations(obs_df)

    print("  Building samples DataFrame...")
    samples_list = []
    for idx, row in obs_df.iterrows():
        row_lat = row['LAT']
        row_lon = row['LON']
        
        entity = row['Station']
        
        metric = row.get('NAME', 'I-131')
        date = row['COLLECT_STOP']
        value = row.get('AVE_ACTIV', 0.0)
        uncertainty = row.get('AVE_ACTIV_ERR', 0.0)
        
        samples_list.append({
            'Entity': entity,
            'Metric': metric,
            'Date': date,
            'Value': value,
            'Uncertainty': uncertainty,
            'MDC Value': mdc_value
        })
    samples_df = pd.DataFrame(samples_list)
    print(f"  Samples: {len(samples_df)} observations")

    # Setup domain with grid dimensions
    domain = domain_add_nxny(domain)
    
    target_lon = domain['lonmin'] + np.arange(domain['nx']) * domain['dx']
    target_lat = domain['latmin'] + np.arange(domain['ny']) * domain['dy']

    # Read and regrid SRM.nc data
    print("  Reading .nc data...")

    ds = xr.open_dataset(srm_nc_path)
    ts_times = pd.to_datetime(ds['ts'].values)
    td_times = pd.to_datetime(ds['td'].values)
    time_indices = {td_times[i]: i for i in range(len(td_times))}
    
    n_sensors = ds['srm_conc'].shape[0]
    n_ts = ds['srm_conc'].shape[1]
    n_td = ds['srm_conc'].shape[2]

    # Setup regridder for interpolation to target grid
    chunk_dims = {'sensor_i': 1, 'td': 1}  # Process one sensor and one time step at a time to reduce memory usage
    ds = ds.chunk(chunk_dims)
    
    lon2d, lat2d = np.meshgrid(target_lon, target_lat)
    
    # xESMF regridder with original and target grids
    regridder = xe.Regridder(
        xr.Dataset({'lon': (('y_i','x_i'), ds['lons_map'].values),
                    'lat': (('y_i','x_i'), ds['lats_map'].values)}),
        xr.Dataset({'lon': (('y_i','x_i'), lon2d),
                    'lat': (('y_i','x_i'), lat2d)}),
        regrid_method,
        periodic=True
    )
    
    nsensors = ds.sizes['sensor_i']
    nts = ds.sizes['ts']
    ntd = ds.sizes['td']
    ny = len(target_lat)
    nx = len(target_lon)

    # Infer time step interval from ts_times
    if len(ts_times) > 1:
        outputfreq_hours = int((ts_times[1] - ts_times[0]).total_seconds() // 3600)
    else:
        outputfreq_hours = 24  # Default if only one time point
    print(f"  Output frequency: {outputfreq_hours} hours")
    
    # Allocate output SRS array depending on storage method
    ntimes = len(times) - 1
    n_samples = len(obs_df)
    srs_shape = (ntimes, domain['nx'], domain['ny'], n_samples)

    if srs_storage == 'memmap':
        if srs_memmap_path is None:
            srs_memmap_path = 'srs_readdata.mmap'
        mmap_dir = os.path.dirname(srs_memmap_path)
        if mmap_dir:
            os.makedirs(mmap_dir, exist_ok=True)
        srs_all = np.memmap(srs_memmap_path, dtype=np.float64, mode='w+', shape=srs_shape)
        srs_all[:] = 0.0
        srs_all.flush()
        print(f"  SRS storage: memmap at {srs_memmap_path}")
    else:
        srs_all = np.zeros(srs_shape, dtype=np.float64)
        print("  SRS storage: RAM")

    # Maincomp_readdata processing: loop through samples and aggregate SRS values into output array
    print(f"  Processing {n_samples} samples...")
    for sample_idx in range(n_samples):
        obs_row = obs_df.iloc[sample_idx]
        sample_date = pd.to_datetime(obs_row['COLLECT_STOP'])
        obs_lat = obs_row['LAT']
        obs_lon = obs_row['LON']
        
        # Find matching sensor index
        sensor_idx = sensor_indices[obs_row['Station']]
        
        if sensor_idx == -1:
            print(f"    Warning: No matching sensor for sample {sample_idx} at ({obs_lat}, {obs_lon})")
            continue
        
        # Find td index for sample date
        td_key = sample_date
        td_idx = time_indices.get(td_key, -1)
        
        if td_idx == -1:
            print(f"    Warning: No matching TD time for sample {sample_idx} date {sample_date}")
            continue
        
        # Extract and regrid SRM for this sensor/date: shape (ts, y_i, x_i)
        data_slice = ds['srm_conc'].isel(sensor_i=sensor_idx, td=td_idx).values
        srm_sensor_td = regridder(data_slice)
        srm_sensor_td[np.isnan(srm_sensor_td)] = 0.0 # patch method makes nans where there is no data
        srm_sensor_td[srm_sensor_td < 0] = 0.0 # Ensure values are non-negative after regridding (patch metos produced errors)
        
        # Optional Gaussian smoothing
        if gaussian_smoothing:
            if gaussian_sigma is None:
                gaussian_sigma = 1.0  # Default sigma if not provided
            apply_gaussian_smoothing(srm_sensor_td, sigma=gaussian_sigma)
        
        # Find time range:
        # start_date = end_date - 1 day or earliest non-zero ts time
        start_date = sample_date - pd.Timedelta(days=1)
        nonzero_ts_idx = np.where(np.any(srm_sensor_td != 0, axis=(1, 2)))[0]
        if len(nonzero_ts_idx) > 0:
            start_date = ts_times[nonzero_ts_idx[0]]
        
        # Filter ts to time range [start_date, sample_date]
        ts_mask = (ts_times <= sample_date) & (ts_times >= start_date)
        srm_filtered = srm_sensor_td[ts_mask, :, :]
        ts_timestamps = ts_times[ts_mask]
        
        if len(ts_timestamps) == 0:
            print(f"    Warning: No data in time range for sample {sample_idx}")
            continue
        
        print(f"    Sample {sample_idx}: Sensor {sensor_idx}, TD {td_idx}, ts times {len(ts_timestamps)}")
        
        srs_regridded = srm_filtered.transpose(0, 2, 1)  # shape (ts, x_i, y_i)
        
        # Aggregate across time intervals as in read_srs_SRM.py
        srs_aggregated = np.zeros((ntimes, domain['nx'], domain['ny']), dtype=np.float64)
        
        # As in .txt.gz files, times are based on simstart - itime * outputfreq
        # Where simstart = sample_date and itime is the index in the filtered ts times
        reconstructed_dates = np.array([
            sample_date - pd.Timedelta(hours=i * outputfreq_hours)
            for i in range(len(ts_timestamps))
        ])
        
        for interval_idx in range(ntimes):
            timestart = times[interval_idx]
            timestop = times[interval_idx + 1]
            
            # Find ts points in this time interval using reconstructed dates
            interval_mask = (reconstructed_dates > timestart) & (reconstructed_dates <= timestop)
            interval_indices = np.where(interval_mask)[0]
            
            if len(interval_indices) > 0:
                # Average values for each grid cell across time points in interval
                # Matching get_srs_fixedperiod behavior
                interval_data = srs_regridded[interval_indices, :, :]  # shape (n_times, nx, ny)
                srs_aggregated[interval_idx, :, :] = np.mean(interval_data, axis=0)  # shape (nx, ny)
        
        # Store result in output array
        srs_all[:, :, :, sample_idx] = srs_aggregated
        if isinstance(srs_all, np.memmap):
            srs_all.flush()
    
    # Apply scaling matching read_srm.py: srs <- srs / (srsfact * header$mass)
    combined_factor = srsfact * mass_factor
    if combined_factor != 1.0:
        srs_all /= combined_factor
        if isinstance(srs_all, np.memmap):
            srs_all.flush()

    #Delete all intermediate variables to free memory
    del regridder, ds
    gc.collect()

    return {
        'samples': samples_df,
        'srs': {0: srs_all},
        'outputfreq': outputfreq_hours * 3600,  # in seconds
        'srs_spread': None  # Placeholder for compatibility with original reader output
    }

import dill
import numpy as np
import os
import pandas as pd

from typing import Dict, Union, Any

from frear.settings import get_default_settings
from frear.domain import domain_add_nxny

def load_settings(logdir: str) -> Dict[str, Any]:
    """
    Load settings from a log directory.
    
    Args:
        logdir (str): directory where settings.pkl is located
    Returns:
        dict of settings (Dict[str, Any]): settings loaded from settings.pkl
    """
    settings_path = os.path.join(logdir, 'settings.pkl') # path to file containing settings
    if os.path.exists(settings_path):
        with open(settings_path, "rb") as f:
            settings = dill.load(f)
        print(f"Settings loaded from {settings_path}")
        return settings
    else:
        raise FileNotFoundError(f"No settings.pkl found at {settings_path} - no settings for maincomp_readdata checkpoint")

def save_dict(srs: Dict[Union[int, str], Union[np.ndarray, np.memmap]], logdir: str, prefix: str = 'srs'):
    """
    Save a dictionary of SRS/SRS_spread arrays (np.array or memmap) to memmap files.
    
    Args:
        srs (Dict[Union[int, str], Union[np.ndarray, np.memmap]]): {key: np.ndarray or np.memmap}
        logdir (str): directory to save files
        prefix (str): prefix for filenames
    """
    os.makedirs(logdir, exist_ok=True)
    shapes_dict = {}
    for key, arr in srs.items():
        key_str = str(key)
        file_path = os.path.join(logdir, f"{prefix}_{key_str}.mmap")

        if isinstance(arr, np.memmap):
            # Copy memmap to new path if not already there
            if os.path.abspath(arr.filename) != os.path.abspath(file_path):
                new_arr = np.memmap(file_path, dtype=arr.dtype, mode='w+', shape=arr.shape)
                for idx in range(arr.shape[0]):
                    new_arr[idx] = arr[idx] # copy in chunks
                new_arr.flush()
                arr = new_arr  # now arr points to new memmap file
            else:
                arr.flush()  # already at correct location
        else:
            # Create memmap and copy data
            mmap_arr = np.memmap(file_path, dtype=arr.dtype, mode='w+', shape=arr.shape)
            for idx in range(arr.shape[0]):
                mmap_arr[idx] = arr[idx]
            mmap_arr.flush()
            arr = mmap_arr  # replace array with memmap

        shapes_dict[key_str] = arr.shape  # store shape in npz for loading

    # Save info for all keys in a .npz (shapes and key mapping)
    np.savez(os.path.join(logdir, f"{prefix}.npz"), **shapes_dict)
    print(f"SRS dictionary saved to {logdir}")

def load_dict(logdir: str, prefix: str = 'srs'):
    """
    Load SRS memmaps from directory previously saved by save_dict.
    
    Returns:
        dict of {key: np.memmap}
    """
    npz_path = os.path.join(logdir, f"{prefix}.npz")
    if not os.path.exists(npz_path):
        print(f"No saved metadata for {prefix} at {npz_path}")
        return None

    srs_dict = {}
    with np.load(npz_path, allow_pickle=True) as info:
        for key_str, shape in info.items():
            file_path = os.path.join(logdir, f"{prefix}_{key_str}.mmap")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing memmap file {file_path}")
            arr = np.memmap(file_path, dtype=np.float64, mode='r+', shape=tuple(shape))
            srs_dict[int(key_str)] = arr
    return srs_dict

def load_checkpoint(logdir: str, pipeline_checkpoint: str = 'maincomp_readdata', pipeline_settings: Dict[str, Any] = {'flag_bayes': False}) -> Dict[str, Any]:
    """
    Load checkpoint data for a given pipeline stage.

    Pipeline stages:
    - maincomp_readdata: loads inital data and settings
    - maincomp_srr: loads data after source-receptor relationship calculations
    - maincomp_calcErr: loads data after calculating likelihood parameters
    - **Last possible point of change for true values and fixed time/location in bayes model setup**
    - maincomp_model_setup: loads data after cost/bayes model setup
    - maincomp_prepbayes: loads data after bayes model setup and before MT-DREAMzs run
    
    Args:
        logdir (str): directory where checkpoints are stored (settings.pkl is required, then uses settings['logdir'] for other files)
        pipeline_checkpoint (str): name of the checkpoint stage to load
        pipeline_settings (Dict[str, Any]): settings to determine which data to load
    Returns:
        dict of loaded data relevant to the checkpoint stage
    """
    checkpoint_index = 0
    if pipeline_checkpoint == 'maincomp_readdata':
        checkpoint_index = 1
    elif pipeline_checkpoint == 'maincomp_srr':
        checkpoint_index = 2
    elif pipeline_checkpoint == 'maincomp_calcErr':
        checkpoint_index = 3
    elif pipeline_checkpoint == 'maincomp_model_setup':
        checkpoint_index = 4
    elif pipeline_checkpoint == 'maincomp_prepbayes':
        checkpoint_index = 5
    
    loader = {}

    if checkpoint_index >= 0:
        loader['settings'] = get_default_settings() # load default settings for all stages, will be updated with loaded settings if checkpoint_index >= 1

    if checkpoint_index >= 1:
        # Load data created from maincomp_readdata stage
        # Data includes:
        # - settings (from settings.pkl)
        # - samples (from samples.csv)
        # - srs_raw (from srs_*.mmap)
        # - srs_spread_raw (from srs_spread_*.mmap)
        # - outputfreq (from outputfreq.txt)
        loader['settings'] = load_settings(logdir)
        
        loader['samples'] = pd.read_csv(loader['settings']['logdir'] + 'samples.csv')
        loader['srs_raw'] = load_dict(loader['settings']['logdir'], prefix='srs')
        loader['srs_spread_raw'] = load_dict(loader['settings']['logdir'], prefix='srs_spread')
        loader['outputfreq'] = int(open(loader['settings']['logdir'] + 'outputfreq.txt', 'r').read().strip())
    
        if 'nx' not in loader['settings']['domain'] or 'ny' not in loader['settings']['domain']:
            # if skipping reading data, may lack nx, ny in domain
            loader['settings']['domain'] = domain_add_nxny(loader['settings']['domain'])

    if checkpoint_index >= 2:
        # Load data created from maincomp_srr stage
        # Maincomp_readdata stage results are already loaded
        # Data includes:
        # - srs (from srs.npy)
        # - srs_error (from srs_error.npy), optional
        # - Qfact (from Qfact.txt)
        # - obs (from obs.npy)
        # - obs_error (from obs_error.npy)
        # - MDC (from MDC.npy)
        loader['srs'] = np.load(loader['settings']['logdir'] + 'srs.npy')
        if os.path.exists(loader['settings']['logdir'] + 'srs_error.npy'):  
            loader['srs_error'] = np.load(loader['settings']['logdir'] + 'srs_error.npy')
        else:
            loader['srs_error'] = None
        loader['Qfact'] = float(open(loader['settings']['logdir'] + 'Qfact.txt', 'r').read().strip())
        loader['obs'] = np.load(loader['settings']['logdir'] + 'obs.npy')
        loader['obs_error'] = np.load(loader['settings']['logdir'] + 'obs_error.npy')
        loader['MDC'] = np.load(loader['settings']['logdir'] + 'MDC.npy')
    
    if checkpoint_index >= 3:
        # Load data created from maincomp_calcErr stage
        # Maincomp_readdata and maincomp_srr stage results are already loaded
        # Data includes:
        # - alphas (from alphas.npy)
        # - betas (from betas.npy)
        # - sigmas (from sigmas.npy)
        loader['ll_parameters'] = {
            'alphas': np.load(loader['settings']['logdir'] + 'alphas.npy'),
            'betas': np.load(loader['settings']['logdir'] + 'betas.npy'),
            'sigmas': np.load(loader['settings']['logdir'] + 'sigmas.npy')
        }

    if checkpoint_index >= 4:
        # Load data created from maincomp_model_setup stage
        # Maincomp_readdata, maincomp_srr and maincomp_calcErr stage results are already loaded
        # Data includes:
        # - setup_data (from setup_data.pkl) - dictionary of model setup, saved with dill
        setup_data_path = os.path.join(loader['settings']['logdir'], 'setup_data.pkl')
        if os.path.exists(setup_data_path):
            with open(setup_data_path, "rb") as f:
                loader['setup_data'] = dill.load(f)
            print(f"Model setup data loaded from {setup_data_path}")
        else:
            raise FileNotFoundError(f"No setup_data.pkl found at {setup_data_path} - no model setup data for maincomp_model_setup checkpoint")

    if checkpoint_index >= 5:
        # Load data created from maincomp_prepbayes stage
        # Maincomp_readdata, maincomp_srr and maincomp_calcErr stage results are already loaded
        # Data includes:
        # - ll_out (from ll_out.pkl) - dictionary of bayes model setup, saved with dill
        ll_out_path = os.path.join(loader['settings']['logdir'], 'll_out.pkl')
        if os.path.exists(ll_out_path):
            with open(ll_out_path, "rb") as f:
                loader['ll_out'] = dill.load(f)
            print(f"Bayes model setup loaded from {ll_out_path}")
        else:
            raise FileNotFoundError(f"No ll_out.pkl found at {ll_out_path} - no bayes model setup for maincomp_prepbayes checkpoint")
        
    # The rest of the stages have only one function to run and save results, so no additional data to load from their checkpoints
    return loader

def save_checkpoint(logdir: str, pipeline_checkpoint: str, data: Dict[str, Any], pipeline_settings: Dict[str, Any]) -> None:
    """
    Save checkpoint data for a given pipeline stage.

    Pipeline stages:
    - maincomp_readdata: loads inital data and settings
    - maincomp_srr: loads data after source-receptor relationship calculations
    - maincomp_calcErr: loads data after calculating likelihood parameters
    - **Last possible point of change for true values and fixed time/location in bayes model setup**
    - maincomp_model_setup: loads data after cost/bayes model setup
    - maincomp_prepbayes: loads data after bayes model setup and before MT-DREAMzs run
    
    Args:
        logdir (str): directory where checkpoints are stored (settings.pkl is saved there, then uses settings['logdir'] for other files)
        pipeline_checkpoint (str): name of the checkpoint stage to save
        data (Dict[str, Any]): dictionary of data to save for the checkpoint stage
        pipeline_settings (Dict[str, Any]): dictionary of pipeline settings (flag_cost, flag_bayes) to determine which data to save
    """
    checkpoint_index = 0
    if pipeline_checkpoint == 'maincomp_readdata':
        checkpoint_index = 1
    elif pipeline_checkpoint == 'maincomp_srr':
        checkpoint_index = 2
    elif pipeline_checkpoint == 'maincomp_calcErr':
        checkpoint_index = 3
    elif pipeline_checkpoint == 'maincomp_model_setup':
        checkpoint_index = 4
    elif pipeline_checkpoint == 'maincomp_prepbayes':
        checkpoint_index = 5

    settings_path = os.path.join(logdir, 'settings.pkl')
    dill.dump(data['settings'], open(settings_path, "wb"))
    print(f"Settings saved to {settings_path}")

    if checkpoint_index >= 1:
        # Save data created from maincomp_readdata stage
        # Data includes:
        # - settings (saved above)
        # - samples (to samples.csv)
        # - srs_raw (to srs_*.mmap using save_dict)
        # - srs_spread_raw (to srs_spread_*.mmap using save_dict)
        # - outputfreq (to outputfreq.txt)
        data['samples'].to_csv(data['settings']['logdir'] + 'samples.csv', index=False)
        print(data['outputfreq'], file=open(data['settings']['logdir'] + 'outputfreq.txt', 'w+'))
        if data.get('srs_raw') is not None:
            save_dict(data['srs_raw'], data['settings']['logdir'], prefix='srs')
        if data.get('srs_spread_raw') is not None:
            save_dict(data['srs_spread_raw'], data['settings']['logdir'], prefix='srs_spread')
    
    if checkpoint_index >= 2:
        # Save data created from maincomp_srr stage
        # Maincomp_readdata stage results are already saved
        # Data includes:
        # - srs (to srs.npy)
        # - srs_error (to srs_error.npy), optional
        # - Qfact (to Qfact.txt)
        # - obs (to obs.npy)
        # - obs_error (to obs_error.npy)
        # - MDC (to MDC.npy)
        np.save(data['settings']['logdir'] + 'srs.npy', data['srr_data']['srs'])
        if data['srr_data'].get('srs_error') is not None:
            np.save(data['settings']['logdir'] + 'srs_error.npy', data['srr_data']['srs_error'])
        np.save(data['settings']['logdir'] + 'obs.npy', data['srr_data']['obs'])
        np.save(data['settings']['logdir'] + 'obs_error.npy', data['srr_data']['obs_error'])
        np.save(data['settings']['logdir'] + 'MDC.npy', data['srr_data']['MDC'])
        print(data['srr_data']['Qfact'], file=open(data['settings']['logdir'] + 'Qfact.txt', 'w+'))

    if checkpoint_index >= 3:
        # Save data created from maincomp_calcErr stage
        # Maincomp_readdata and maincomp_srr stage results are already saved
        # Data includes:
        # - alphas (to alphas.npy)
        # - betas (to betas.npy)
        # - sigmas (to sigmas.npy)
        np.save(data['settings']['logdir'] + 'alphas.npy', data['ll_parameters']['alphas'])
        np.save(data['settings']['logdir'] + 'betas.npy', data['ll_parameters']['betas'])
        np.save(data['settings']['logdir'] + 'sigmas.npy', data['ll_parameters']['sigmas'])

    if checkpoint_index >= 4:
        # Save data created from maincomp_model_setup stage
        # Maincomp_readdata, maincomp_srr and maincomp_calcErr stage results are already saved
        # Data includes:
        # - setup_data (to setup_data.pkl) - dictionary of model setup, saved with dill
        setup_data_path = os.path.join(data['settings']['logdir'], 'setup_data.pkl')
        with open(setup_data_path, "wb") as f:
            dill.dump(data['setup_data'], f)
        print(f"Model setup data saved to {setup_data_path}")

    if checkpoint_index >= 5:
        # Save data created from maincomp_prepbayes stage
        # Maincomp_readdata, maincomp_srr, and maincomp_calcErr stage results are already saved
        # Data includes:
        # - ll_out (to ll_out.pkl) - dictionary of bayes model setup, saved with dill
        ll_out_path = os.path.join(data['settings']['logdir'], 'll_out.pkl')
        with open(ll_out_path, "wb") as f:
            dill.dump(data['ll_out'], f)
        print(f"Bayes model setup saved to {ll_out_path}")

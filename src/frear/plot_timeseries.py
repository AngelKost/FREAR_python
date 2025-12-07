import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from typing import Dict, Any, Optional

def plot_ac_timeseries(ac: Optional[Dict[str, Any]] = None, obs: Optional[np.ndarray] = None,  
                       obs_min: Optional[np.ndarray] = None, obs_max: Optional[np.ndarray] = None,
                       mod: Optional[np.ndarray] = None, 
                       mod_min: Optional[np.ndarray] = None, mod_max: Optional[np.ndarray] = None, 
                       MDC: Optional[np.ndarray] = None, 
                       obsunitfact: float = 10**-6, obsscalingfact: float = 1, 
                       title: str = "", labels: str = "", prefix: str = "",
                       outpath: Optional[str] = None, show: bool = True,
                       ax: Optional[plt.Axes] = None) -> None:
    """Plot activity concentration time series
    
    Parameters:
        ac (Optional[Dict[str, Any]]): dictionary of data
            - obs (np.ndarray): Observations array
            - sigma_obs (np.ndarray): Observation uncertainties array
            - mod (np.ndarray): Model values array
            - sigma_mod (np.ndarray): Model uncertainties array
        obs (Optional[np.ndarray]): Observations array
        obs_min (Optional[np.ndarray]): Minimum observation values array
        obs_max (Optional[np.ndarray]): Maximum observation values array
        mod (Optional[np.ndarray]): Model values array
        mod_min (Optional[np.ndarray]): Minimum model values array
        mod_max (Optional[np.ndarray]): Maximum model values array
        MDC (Optional[np.ndarray]): Minimum detectable concentration array
        obsunitfact (float): Unit conversion factor for observations
        obsscalingfact (float): Scaling factor for observations
        title (str): Title for the plot
        labels (str): Labels for the x-axis
        prefix (str): Prefix for the output filename
        outpath (Optional[str]): Output path to save the plot
        show (bool): Whether to display the plot
        ax (Optional[plt.Axes]): Matplotlib Axes object to plot on
    """
    aux = np.log10(obsunitfact * obsscalingfact)
    unitlist = np.arange(-12, 7, 3)
    iunit = np.argmin((aux - unitlist)**2)
    scalingdiff = 10**(aux - unitlist[iunit])

    if ac is not None:
        obs = ac['obs']
        obs_min = ac['obs'] - ac['sigma_obs']
        obs_max = ac['obs'] + ac['sigma_obs']
        mod = ac['mod']
        mod_min = ac['mod'] - ac['sigma_mod']
        mod_max = ac['mod'] + ac['sigma_mod']
    
    if obs is not None:
        if obs_min is None:
            obs_min = obs
        if obs_max is None:
            obs_max = obs
        obs = obs * scalingdiff
        obs_min = obs_min * scalingdiff
        obs_max = obs_max * scalingdiff
    
    if mod is not None:
        if mod_min is None:
            mod_min = mod
        if mod_max is None:
            mod_max = mod
        mod = mod * scalingdiff
        mod_min = mod_min * scalingdiff
        mod_max = mod_max * scalingdiff
    if MDC is not None:
        MDC = MDC * scalingdiff

    ylabels = [
        'Activity concentration [pBq/m³]',
        'Activity concentration [nBq/m³]',
        'Activity concentration [μBq/m³]',
        'Activity concentration [mBq/m³]',
        'Activity concentration [Bq/m³]',
        'Activity concentration [kBq/m³]',
        'Activity concentration [MBq/m³]',
    ]
    ylab = ylabels[iunit]
    eps = min(np.min(obs[obs != 0]), np.inf if mod is None or len(mod[mod != 0]) == 0 else np.min(mod[mod != 0])) / 1e3
    ymax = max(np.max(obs), np.max(obs_max),
            0 if mod is None else np.max(mod),
            0 if mod_max is None else np.max(mod_max),
            0 if MDC is None else np.max(MDC))
    nsamples = len(obs)

    x = np.arange(1, nsamples+1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = None

    ax.errorbar(x, obs, yerr=[obs-obs_min, obs_max-obs], fmt='o', color='black', label='Observation', capsize=3)

    if mod is not None:
        ax.errorbar(x, mod, yerr=[mod-mod_min, mod_max-mod], fmt='s', color='red', label='Model', capsize=3)

    if MDC is not None:
        ax.scatter(x, MDC, color='magenta', marker='D', label='MDC')

    if labels is not None and nsamples >= 15:
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        for ul in unique_labels:
            idx = np.where(labels == ul)[0]
            if len(idx) > 1:
                ax.hlines(y=eps, xmin=idx[0]+1, xmax=idx[-1]+1, color='gray', linewidth=1)

    ax.set_yscale('log')
    ax.set_xlabel('Sample')
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.set_xticks(x)
    if labels is not None:
        ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)

    if fig is not None:
        fig.tight_layout()
        if outpath is not None:
            fig.savefig(os.path.join(outpath, f"{prefix}_ac_timeseries.pdf"))
        if show:
            plt.show()
        else:
            plt.close(fig)
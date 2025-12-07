import numpy as np

from typing import Dict, Optional, Any
from scipy.stats import pearsonr, spearmanr

from frear.plot_geofield import plot_2D_direct

# Calculate the maximum-in-time PSR for each grid box of the output domain
# for a given srs field and given observations
# srs: array with SRS values [ntimes, nx, ny, nsamples]
# obs: vector of observations
# method: pearson, spearman (see ?cor)

def calc_maxPSR(srs: np.ndarray, obs: np.ndarray, method: str = "spearman") -> np.ndarray:
    """Calculate maximum probability of source release (maxPSR) over the domain
    
    Parameters:
        srs (np.ndarray): Source-receptor sensitivity array [ntimes, nx, ny, nsamples]
        obs (np.ndarray): Observed data array
        method (str): Correlation method to use ("pearson" or "spearman")
    Returns:
        maxPSR (np.ndarray): Maximum PSR array [nx, ny]
    """
    def h(x: np.ndarray, obs: np.ndarray, method: str) -> float:
        if np.sum(np.abs(x)) == 0:
            return 0.0
        else:
            if method == "pearson":
                return pearsonr(x, obs).correlation
            elif method == "spearman":
                return spearmanr(x, obs).correlation
            else:
                raise ValueError(f"Unknown method: {method}")

    ntimes, nx, ny, nsamples = srs.shape
    aux = np.zeros((ntimes, nx, ny), dtype=float)
    for itime in range(ntimes):
        for ix in range(nx):
            for iy in range(ny):
                aux[itime, ix, iy] = h(srs[itime, ix, iy, :], obs, method)

    maxPSR = np.max(aux, axis=0)
    return maxPSR

def plot_maxPSR(
    maxPSR: np.ndarray,
    domain: dict,
    title: str = "",
    IMSfile: str = None,
    reactorfile: str = None,
    prefix: str = "",
    outpath: str = None,
    show: bool = True
):
    """Plot maximum probability of source release (maxPSR) over the domain
    
    Parameters:
        maxPSR (np.ndarray): Maximum PSR array [nx, ny]
        domain (dict): Domain information for plotting
        title (str): Title for the plot
        IMSfile (str): IMS file 
        reactorfile (str): Reactor file
        prefix (str): Prefix for the output filename
        outpath (str): Output path for saving the plot
        show (bool): Whether to display the plot
    """
    colors = ["white", "royalblue", "yellow", "red"]
    levels = np.arange(0, 1.1, 0.1)

    plot_2D_direct(
        data=maxPSR.T,
        domain=domain,
        title=title,
        colors=colors,
        levels=levels,
        IMSfile=IMSfile,
        reactorfile=reactorfile,
        labels=["Longitude", "Latitude", "Max PSR"],
        show=show,
        outpath=outpath,
        filename=f"{prefix}maxPSR_plot.pdf"
    )

def write_maxPSR(maxPSR: np.ndarray, settings: Dict[str, Any], 
                 maxPSR_pearson: Optional[np.ndarray] = None, flag_save: bool = True):
    """Write and plot maxPSR data
    
    Parameters:
        maxPSR (np.ndarray): Maximum PSR array [nx, ny]
        settings (Dict[str, Any]): Settings dictionary
            - outpath (str): Output path for saving files
            - domain (dict): Domain information for plotting
            - IMSfile (str): IMS file
            - reactorfile (str): Reactor file
        maxPSR_pearson (Optional[np.ndarray]): Optional maximum PSR array using Pearson correlation
        flag_save (bool): Whether to save the maxPSR data to a file
    """
    if flag_save:
        np.save(settings['outpath'] + '/maxPSR.npy', maxPSR)
    
    plot_maxPSR(
        maxPSR=maxPSR,
        domain=settings['domain'],
        title="maximum-in-time PSR",
        IMSfile=settings['IMSfile'],
        reactorfile=settings['reactorfile'],
        outpath=settings['outpath'],
        show=True
    )

    if maxPSR_pearson is not None:
        plot_maxPSR(
            maxPSR=maxPSR_pearson,
            domain=settings['domain'],
            title="maximum-in-time PSR (Pearson)",
            IMSfile=settings['IMSfile'],
            reactorfile=settings['reactorfile'],
            prefix="pearson_",
            outpath=settings['outpath'],
            show=True
        )

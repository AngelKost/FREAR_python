import numpy as np

from typing import Dict, Any, Optional

from frear.plot_geofield import plot_2D_direct

# Calculate the accumulated SRS associated with each detection for each grid box of the output domain
# for a given srs field and given observations
# srs: array with SRS values [ntimes, nx, ny, nsamples]
# obs: vector of observations

def calc_accFOR(srs: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Calculate accumulated frequency of release (accFOR) for detections.

    Parameters:
        srs (np.ndarray): 4D numpy array with dimensions [ntimes, nx, ny, nsamples].
        obs (np.ndarray): 1D numpy array with length nsamples.
    Returns:
        accFOR (np.ndarray): 2D numpy array with dimensions [nx, ny] containing accFOR values
    """

    detection_indices = np.where(obs > 0)[0]
    srs_det = srs[:, :, :, detection_indices]

    srs_det_tacc = np.sum(srs_det, axis=0)

    zthres = 0.0  # 1 / (Qfact * settings['Qmax'])
    srs_det_tacc_bin = np.where(srs_det_tacc > zthres, 1, 0)

    accFOR = np.sum(srs_det_tacc_bin, axis=2)

    return accFOR

def plot_accFOR(accFOR: np.ndarray, ndet: int, domain: Dict[str, Any],
                title: str = "", 
                IMSfile: Optional[str] = None, reactorfile: Optional[str] = None,
                outpath: Optional[str] = None, show: bool = True):
    """Plot accFOR results.
    
    Parameters:
        accFOR (np.ndarray): 2D numpy array with dimensions [nx, ny] containing accFOR values.
        ndet (int): Number of detections.
        domain (Dict[str, Any]): Dictionary containing domain information.
        title (str): Title for the plot.
        IMSfile (Optional[str]): Path to IMS file.
        reactorfile (Optional[str]): Path to reactor file.
        outpath (Optional[str]): Output path for saving the plot.
        show (bool): Whether to display the plot.
    """
    if ndet == 0:
        raise ValueError("ndet cannot be zero.")

    fracFOR = accFOR / ndet

    colors = ["white", "royalblue", "yellow", "red"]

    levels = np.arange(0, 1.1, 0.1)

    plot_2D_direct(
        data=fracFOR.T,
        domain=domain,
        title=title,
        colors=colors,
        levels=levels,
        IMSfile=IMSfile,
        reactorfile=reactorfile,
        labels=["Longitude", "Latitude", "Fraction of overlapping SRS"],
        show=show,
        outpath=outpath,
        filename="accFOR_plot.pdf"
    )

def write_accFOR(accFOR: np.ndarray, ndet: int, settings: Dict[str, Any], flag_save: bool = True):
    """Write accFOR results to file and plot data.

    Parameters:
        accFOR (np.ndarray): 2D numpy array with dimensions [nx, ny]
        ndet (int): number of detections (sum of obs > 0)
        settings (Dict[str, Any]): dictionary of settings
        flag_save (bool): boolean flag indicating whether to save the input data.
    """
    if flag_save:
        np.save(settings['outpath'] + '/data/accFOR.npy', accFOR)

    title = "Fraction of overlapping SRS of detections"
    plot_accFOR(
        accFOR=accFOR,
        ndet=ndet,
        domain=settings['domain'],
        title=title,
        IMSfile=settings['IMSfile'],
        reactorfile=settings['reactorfile'],
        outpath=settings['outpath'],
        show=True
    )
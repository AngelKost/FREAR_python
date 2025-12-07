import numpy as np
import os

from typing import Dict, Any

from frear.read_samples import read_CRToolSamples
from frear.paths import create_paths, create_processed_paths
from frear.read_srs_SRM import read_all_srm, read_all_srm as readAllSRM_Flexpart

def maincomp_readdata(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Read SRS and observation data based on provided settings.
    
    Parameters:
        settings (Dict[str, Any]): Configuration settings for reading data.
            - inputfile (str): Path to the input file containing samples.
            - expdir (str): Experiment directory.
            - subexp (str): Sub-experiment identifier.
            - datadir (str): Directory containing SRS data.
            - members (Optional[List[str]]): List of member identifiers.
            - processedSRSdir (Optional[str]): Directory for processed SRS files.
            - times (Optional[List[Any]]): List of times for SRS data.
            - domain (Optional[Dict[str, Any]]): Domain settings for SRS data.
            - srsfact (float): Scaling factor for SRS values.
            - halflife (Optional[float]): Halflife for decay calculations.
            - parallel (bool): Flag to enable parallel processing.
            - nproc (int): Number of processes for parallel reading.
            - FP_zlevels (Optional[List[float]]): Vertical levels for Flexpart data.
    Returns:
        data (Dict[str, Any]): A dictionary containing the read data
            - srs_raw (np.ndarray): Raw source-receptor sensitivity data.
            - srs_spread_raw (Optional[np.ndarray]): Spread of raw SRS data.
            - outputfreq (Any): Output frequency information.
            - samples (Dict[str, Any]): Samples read from the input file.
    """

    samples = read_CRToolSamples(settings['inputfile'])

    srsfilelists_prefix = f"{settings['expdir']}/srsfilelist_{settings['subexp']}"
    if settings.get('members') is not None:
        srsfilelists = [srsfilelists_prefix + '_' + member + '.dat' for member in settings['members']]
    else:
        srsfilelists = [srsfilelists_prefix + '.dat']
    
    srmpaths = create_paths(datadir=settings['datadir'], srsfilelists=srsfilelists, members=settings['members'])
    processed_paths = create_processed_paths(processed_srs_dir=settings['processedSRSdir'], srsfilelists=srsfilelists, members=settings['members'])

    for i, srsfile in enumerate(srsfilelists):
        if len(samples) != len(srmpaths[i]):
            raise ValueError(
                f"ERROR: # observations ({len(samples)}) in '{settings['inputfile']}' "
                f"is inconsistent with # SRS files ({len(srmpaths[i])}) in '{srsfile}'"
            )
        
    first_path = srmpaths[0][0][0]

    if os.path.isdir(first_path):  # Flexpart directory case
    
        if settings["members"] is None:
            raise ValueError("Members cannot be NULL if Flexpart is used")

        data = readAllSRM_Flexpart( #FIXME: not implemented yet
            fpdirs=srmpaths,
            times=settings["times"],
            domain=settings["domain"],
            srsfact=settings["srsfact"],
            halflife=settings["halflife"],
            zlevels=settings["FP_zlevels"],
            fp_processed=processed_paths
        )

    else:
        if settings["parallel"]:
            data = read_all_srm(
                srmpaths=srmpaths,
                times=settings["times"],
                domain=settings["domain"],
                srsfact=settings["srsfact"],
                halflife=settings["halflife"],
                nproc=settings["nproc"],
                srm_processed=processed_paths
            )
        else:
            data = read_all_srm(
                srmpaths=srmpaths,
                times=settings["times"],
                domain=settings["domain"],
                srsfact=settings["srsfact"],
                halflife=settings["halflife"],
                nproc=1,
                srm_processed=processed_paths
            )

        if settings["processedSRSdir"] is not None:
            srs0 = data["srs"][0]
            ntime, nx, ny = srs0.shape

            if ntime != len(settings["times"]) - 1:
                raise ValueError(
                    "Requested settings[times] incompatible with SRS files in processedSRSdir"
                )
            if nx != settings["domain"]["nx"]:
                raise ValueError(
                    "Requested settings[domain][nx] incompatible with SRS files in processedSRSdir"
                )
            if ny != settings["domain"]["ny"]:
                raise ValueError(
                    "Requested settings[domain][ny] incompatible with SRS files in processedSRSdir"
                )

    srs_raw = data["srs"]
    srs_spread_raw = data["srs_spread"]
    outputfreq = data["outputfreq"]
    data['samples'] = samples

    return data
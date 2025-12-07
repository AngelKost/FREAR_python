import os
import numpy as np

from typing import List, Optional, Union, Dict

def read_paths(srsfilelist: str, datadir: str, member: Optional[str] = None) -> np.ndarray:
    """Read SRS file paths from a file list and construct full paths
    
    Parameters:
        srsfilelist (str): Path to the file containing SRS file names
        datadir (str): Base directory for data files
        member (Optional[str]): Member identifier to append to the path
    Returns:
        paths (np.ndarray): Array of full SRS file paths
    """
    with open(srsfilelist, 'r+') as file:
        paths = [line.strip().split() for line in file]
    nsamples = len(paths)
    paths = [item for sublist in paths for item in sublist]
    member_path = '' if member is None else f'/{member}'
    paths = [os.path.join(datadir + member_path, p) for p in paths]
    ncols = len(paths) // nsamples
    paths = np.array(paths)
    return paths.reshape((nsamples, ncols))

def check_paths(paths: List[str]) -> None:
    """Check if all provided file paths exist
    
    Parameters:
        paths (List[str]): List of file paths to check
    """
    missing_files = [path for path in paths if not os.path.exists(path)]
    if missing_files:
        print('The following srs files do not exist:')
        print(missing_files)
        raise ValueError('There are missing srs files!')

def create_paths(datadir: str, srsfilelists: List[str], 
                 members: Optional[List[str]] = None) -> Union[Dict[str, List[np.ndarray]], List[np.ndarray]]:
    """
    Create paths to SRS files from file lists.

    Parameters:
        datadir (str): Base directory for data files
        srsfilelists (List[str]): List of file paths containing SRS file names
        members (Optional[List[str]]): Optional list of member identifiers
    Returns:
        paths_list (Union[Dict[str, List[np.ndarray]], List[np.ndarray]]): dictionary mapping member identifiers to lists of SRS file paths or a list of SRS file paths
    """
    if members is None:
        members = [None] * len(srsfilelists)

    paths_list = [read_paths(srsfilelist, datadir, member) for srsfilelist, member in zip(srsfilelists, members)]

    for p in paths_list:
        file_list = list(p.flatten())
        check_paths(file_list)

    if members is not None and all(members):
        return dict(zip(members, paths_list))
    return paths_list #Either dict or list depending on members

def create_processed_paths(processed_srs_dir: Optional[str], 
                           srsfilelists: List[str], 
                           members: Optional[List[str]] = None) -> Union[Dict[str, List[str]], List[str]]:
    """
    Create paths for processed SRS files.

    Parameters:
        processed_srs_dir (Optional[str]): Directory for processed SRS files
        srsfilelists (List[str]): List of file paths containing SRS file names
        members (Optional[List[str]]): Optional list of member identifiers
    Returns:
        processed_paths_list (Union[Dict[str, List[str]], List[str]]): dictionary mapping member identifiers to lists of processed SRS file paths or a list of processed SRS file paths
    """
    if processed_srs_dir is None:
        return None
    else:
        def process_file_list(srsfilelist, processed_srs_dir, member=None):
            """
            Inner function to generate file paths for a single SRS file list.
            """
            with open(srsfilelist, 'r+') as file:
                filenames = [line.strip() for line in file]
            member_path = '' if member is None else f'/{member}'
            processed_paths = [os.path.join(processed_srs_dir + member_path, fname) for fname in filenames]
            return processed_paths

        if members is None:
            members = [None] * len(srsfilelists)

        processed_paths_list = [
            process_file_list(srsfilelist, processed_srs_dir, member)
            for srsfilelist, member in zip(srsfilelists, members)
        ]

        if members is not None and all(members):
            return dict(zip(members, processed_paths_list))
        return processed_paths_list
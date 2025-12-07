import numpy as np
import os

from typing import Dict, Any, Optional
from datetime import datetime

from frear.domain import domain_add_nxny

def get_default_settings() -> Dict[str, Any]:
    """Return a dictionary of default settings for FREAR analysis

    Returns:
        settings (Dict[str, Any]): A dictionary containing default settings
    """
    settings = {

        #Files/folders settings
        'experiment': None, #Name of the experiment
        'expdir': None,     #Experiment directory
        'datadir': None,    #Data directory with SRS files
        'outbasedir': './output', #Root directory for outputs
        'subexp': None,     #Sub-experiment name (defines part of the output directory name and 
                            #is used to select an observation file and path file in expdir)
        'inputfile': None,  #Inputfile: name of observation files, if None: 
                            #set as "expdir/input_subexp.dat"
        'processedSRSdir': None,    #Name of directory for processed files to save/load
        'srcdir': None,   #Source directory with FREAR scripts #FIXME: unused?

        #Input settings
        'members': None,    #List of names of the SRS members; data is stored in datadir/member
                            #None if only one member directly in datadir
        'domain': None,     #Domain settings dictionary (lonmin, lonmax, latmin, latmax, dx, dy)
        'times': None,      #Times corresponding to SRS time dimension (list)
        'srsfact': 1.0 / (1.3 * 10**15),   #Multiplicative factor for SRS values
        'obsunitfact': None,#Multiplicative factor for observation units to Bq/m^3 
                            #e.g. 10^-6 if observations in muBq/m^3
        'obsscalingfact': 1.0, #Additional scaling factor for observations (outside unit conversion)
        'halflife': None,   #Halflife of the released substance in seconds (None if no decay)
        'lnoble': False,    #If noble gas, changes naming

        #SRS model settings
        'fcalcsrs': lambda srs_raw: srs_raw[0],   #Function to calculate aggregated SRS from raw SRS files
                            #If None, srs_raw is assumed to be already aggregated SRS (default = srs_raw[0])
        'srsrelError': 0.5, #Relative error in SRS values (fraction)
        'fcalcsrsspread': np.std, #Function to calculate SRS spread from raw SRS spread files
        'mod_error_mode': 'indep_par',  #AC error calculation,
                                        #'indep_par': srsrelError or srs_spread (if available)
                                        #'fitCtrue': fit distribution of SRS using ensemble members (for ens and ll_Yee only!)
                                        #'fitInvGamma': fit an inverse gamma distribution using ensemble members (for ens and ll_Yee only!)
        'sourcemodelbayes': 'rectRelease',  #Bayes function name (srs2AC_bayes_xxx)
        'sourcemodelcost': 'nsegmentsRel',  #Cost function name (srs2AC_cost_xxx)

        #Bayesian inversion settings
        'Qmin': None,       #Minimum total accumulated source term [Bq] (10**10)
        'Qmax': None,       #Maximum total accumulated source term [Bq] (10**16)
        'trueValues': [None, None, None, None, None],   #List with true source model parameter values (for synthetic tests)
        'parnames': ["lon", "lat", "log10_Q", "rstart", "rstop"],  #Names of source model parameters
        'likelihood': 'Yee2017log', #Likelihood model
        'zalpha': 1. / np.pi,   #Model error parameter for inverse gamma
        'zbeta': 1.,            #Model error parameter for inverse gamma
        'niterations': 20000,   #Number of iterations for MCMC
        'nburnin': 0.25,    #Fraction of burn-in iterations to discard in MCMC
        'nthin': 1,         #Thinning factor for MCMC (keep every nthin element)
        'lmultipliers': False,  #If True, include per-observation AC multipliers
        'lfixedLocation': False,#If True, fix the location parameter in Bayesian inversion
        'lfixedTime': False,    #If True, fix the release time parameters in Bayesian inversion
        'lllfactors': False,    #If True, apply likelihood factors

        #MCMC settings
        'nchains': 3,       #Number of MCMC chains
        'pSnooker': 0.,     #Probability of Snooker update in DREAMzs
        'DEpairs': 2,       #Number of pairs for differential evolution in DREAMzs
        'Zupdatefreq': 10,  #Frequency of Z archive updates in DREAMzs
        'adaptation': 0,    #Adaptation period for DREAMzs
        'nMTM': 5,         #Number of multiple-try MCMC trials in DREAMzs
        'zgammaFactor': 1., #Gamma factor which scales the perturbation used to create new proposals for DREAMzs
        'zb': 0.05,         #Bound of uniform draw in generation of new proposals for DREAMzs
        'zbstar': 1e-6,     #Sigma of uniform draw in generation of new proposals for DREAMzs
        'nCR': 3,           #Number of crossover values in DREAMzs
        'CRupdatefreq': 10, #Frequency of crossover value updates in DREAMzs
        'checkfreq': 100,   #Frequency of convergence checks in DREAMzs

        #Other settings
        'version': None,    #FREAR version #FIXME: unused?
        'reactorfile': None,#Reactor locations file
        'IMSfile': None,    #IMS locations file
        'ellipsefile': None,#Seismic data file #FIXME: unused?
        'FP_zlevels': 1,    #FLEXPART model levels #FIXME: unused?
        'nProbLocBoundary': 5, #Boundary in degrees for probable location calculation

        'nproc': 1,        #Number of processors to use
        'parallel': False, #If True, use parallel processing #FIXME: unused?
    }
    return settings

def write_settings(settings: Dict[str, Any]) -> None:
    """Write settings dictionary to a txt file
    
    Parameters:
        settings (Dict[str, Any]): Configuration settings dictionary
    """
    logfile_path = settings['logfile']
    with open(logfile_path, 'w+') as f:
        for key, value in settings.items():
            f.write(f"{key}: {value}\n") #FIXME: better formatting for dict/function/etc

def check_settings(settings: Dict[str, Any]) -> None:
    """Check that required settings are provided and valid
    
    Parameters:
        settings (Dict[str, Any]): Configuration settings dictionary
    """
    required_keys = ['expdir', 'datadir', 'domain']
    for key in required_keys:
        if key not in settings or settings[key] is None:
            raise ValueError(f"Missing required setting: {key}")

    domain = settings['domain']
    for dim in ['lonmin', 'lonmax', 'latmin', 'latmax', 'dx', 'dy']:
        if dim not in domain:
            raise ValueError(f"Missing domain setting: {dim}")
    if not ((domain['lonmax'] - domain['lonmin']) / domain['dx']).is_integer():
        raise ValueError("Can't divide domain into cells")
    if not ((domain['latmax'] - domain['latmin']) / domain['dy']).is_integer():
        raise ValueError("Can't divide domain into cells")

    if settings['inputfile'] is None:
        settings['inputfile'] = f"{settings['expdir']}/input_{settings['subexp']}.dat"
    settings['domain'] = domain_add_nxny(settings['domain'])

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings['outpath'] = f"{settings['outbasedir']}/{settings['experiment']}_{settings['subexp']}_{current_time}"
    settings['logfile'] = f"{settings['outpath']}/settings_log.txt"


    if settings['nburnin'] < 1:
        settings['nburnin'] = int(settings['niterations'] * settings['nburnin'])
    if settings['adaptation'] < 1:
        settings['adaptation'] = int(settings['niterations'] * settings['adaptation'])


    os.makedirs(settings['outpath'], exist_ok=True)
    write_settings(settings)
    return
    
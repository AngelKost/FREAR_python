import numpy as np

from typing import Dict, Any, Optional

from frear.srs2AC_bayes_contRel import srs2AC_bayes_contRel_setup
from frear.srs2AC_bayes_nsegmentsRel import srs2AC_bayes_nsegmentsRel_setup
from frear.srs2AC_bayes_rectRelease import srs2AC_bayes_rectRelease_setup
from frear.srs2AC_cost_contRel import srs2AC_cost_contRel_setup
from frear.srs2AC_cost_nsegmentsRel import srs2AC_cost_nsegmentsRel_setup

def maincomp_model_setup(settings: Dict[str, Any], Qfact: float, 
                         nobs: int, ntimes: int,
                         flag_cost: bool, flag_bayes: bool) -> Dict[str, Any]:
    """
    Setup bayes/cost model specific parameters and functions

    Parameters:
        settings (Dict[str, Any]): Configuration settings
            - sourcemodelcost (str): Type of source model cost to use
            - sourcemodelbayes (str): Type of source model Bayesian prior to use
        Qfact (float): Scaling factor for SRR
        nobs (int): Number of observations (srs.shape[3])
        ntimes (int): Number of time steps (srs.shape[0])
        flag_cost (bool): Flag to enable cost model computation
        flag_bayes (bool): Flag to enable Bayesian model computation

    Returns:
        data (Dict[str, Any]): A dictionary containing model setup data
            - cost_data (Optional[Dict[str, Any]]): Updated cost model data if flag_cost is True
            - bayes_data (Optional[Dict[str, Any]]): Updated bayesian model data if flag_bayes is True
    """
    if flag_cost: #lower_cost, upper_cost, par_init, settings['sourcemodelcost_exec']
        if settings['sourcemodelcost'] == 'contRel':
            cost_data = srs2AC_cost_contRel_setup(settings, Qfact, ntimes)
        elif settings['sourcemodelcost'] == 'nsegmentsRel':
            cost_data = srs2AC_cost_nsegmentsRel_setup(settings, Qfact, ntimes)
        else:
            raise ValueError(f"Unknown sourcemodelcost: {settings['sourcemodelcost']}")

    if flag_bayes: #lower_bayes, upper_bayes, settings['sourcemodelbayes_exec']
        if settings['sourcemodelbayes'] == 'contRel':
            bayes_data = srs2AC_bayes_contRel_setup(settings, nobs)
        elif settings['sourcemodelbayes'] == 'nsegmentsRel':
            bayes_data = srs2AC_bayes_nsegmentsRel_setup(settings, nobs)
        elif settings['sourcemodelbayes'] == 'rectRelease':
            bayes_data = srs2AC_bayes_rectRelease_setup(settings, nobs)
        else:
            raise ValueError(f"Unknown sourcemodelbayes: {settings['sourcemodelbayes']}")
        
    return {
        'cost_data': cost_data if flag_cost else None,
        'bayes_data': bayes_data if flag_bayes else None
    }
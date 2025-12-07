import numpy as np

from typing import Any, Dict
from scipy.optimize import minimize
from scipy.optimize import Bounds
from multiprocessing import Pool

def h(x: np.ndarray, MDC: np.ndarray) -> np.ndarray:
    """Apply transformation on input data x based on MDC values.
    
    Parameters:
        x (np.ndarray): Input data array
        MDC (np.ndarray): Minimum Detectable Concentration array

    Returns:
        out (np.ndarray): Transformed data array
    """
    transformed = MDC + (x * x) / (4.0 * MDC)
    out = np.where(x < 2.0 * MDC, transformed, x)
    return out

def f_cost(par: np.ndarray, obs: np.ndarray, M: np.ndarray, 
           sigma: np.ndarray, upper_cost: np.ndarray, 
           srs2AC_cost: Any) -> float:
    """Calculate cost function value for given parameters.
    
    Parameters:
        par (np.ndarray): Parameter array
        obs (np.ndarray): Observed data array
        M (np.ndarray): Slice of SRS data
        sigma (np.ndarray): Standard deviation or error array
        upper_cost (np.ndarray): Upper bounds for cost calculation
        srs2AC_cost (Any): Function to convert source parameters to modeled concentrations

    Returns:
        cost (float): Computed cost value
    """
    par = np.asarray(par)
    obs = np.asarray(obs)
    sigma = np.asarray(sigma)

    mod = srs2AC_cost(par=par, M=M, Qfact=1)

    h_obs = h(obs, sigma)
    h_mod = h(mod, sigma)

    log_diff = np.log(h_obs) - np.log(h_mod)
    mean_logdiff = np.mean(log_diff ** 2)

    add = np.log10(1.0 + np.sum(par / upper_cost))

    return np.exp(mean_logdiff) + add

def optimizer(srs: np.ndarray, par_init: np.ndarray, 
                  obs: np.ndarray, sigma: np.ndarray,
                  lower_cost: np.ndarray, upper_cost: np.ndarray,
                  f_cost_func: Any,
                  ix: int, iy: int,
                  srs2AC_cost: Any) -> Dict[str, Any]:
    """Optimize parameters for fixed (ix, iy) location
    
    Parameters:
        srs (np.ndarray): Source-receptor sensitivity array
        par_init (np.ndarray): Initial guess for parameters
        obs (np.ndarray): Observed data array
        sigma (np.ndarray): Standard deviation or error array
        lower_cost (np.ndarray): Lower bounds for cost calculation
        upper_cost (np.ndarray): Upper bounds for cost calculation
        f_cost_func (Any): Cost function to minimize
        ix (int): X-coordinate (longitude) index
        iy (int): Y-coordinate (latitude) index
        srs2AC_cost (Any): Function to convert source parameters to modeled concentrations
    Returns:
        result (Dict[str, Any]): Dictionary containing optimization results
            - par (np.ndarray): Optimized parameters
            - cost (float): Final cost value
            - ntries (int): Number of tries taken to converge
    """
    rel_tols = [1e-10, 1e-5, 1e-4, 1e-3]
    step_mins = [0.1, 100, 100, 100]
    step_maxs = [1, 100, 100, 1000]
    eval_maxs = [400, 4000, 4000, 10000]
    max_tries = len(rel_tols)
    itry = 0
    flag_converged = False

    M = srs[:, ix, iy, :]

    while itry < max_tries and not flag_converged:
        # Use trust-constr to better mirror R's nlminb (PORT) behavior with box constraints
        bnds = Bounds(lower_cost, upper_cost, keep_feasible=True)
        costmod = minimize(
            fun=f_cost_func,
            x0=par_init,
            args=(obs, M, sigma, upper_cost, srs2AC_cost),
            method='L-BFGS-B',
            bounds=bnds,
            options={
                'ftol': rel_tols[itry],
                'maxfun': eval_maxs[itry],
                'maxiter': 1000,
            }
        )

        # Accept only if optimizer reports success AND objective is finite and below the 1e6 penalty cap
        if costmod.success and np.isfinite(costmod.fun) and costmod.fun < 1e6:
            flag_converged = True
            par = costmod.x
            cost = float(costmod.fun)
        else:
            if itry == max_tries - 1:
                # Penalize non-convergence to mirror R's 1e6 sentinel
                par = np.zeros_like(par_init)
                cost = 1e6
            itry += 1

    return {
        "par": par,
        "cost": cost,
        "ntries": itry
    }

def worker(ix: int, nx: int, ny: int, srs: np.ndarray, 
           par_init: np.ndarray, obs: np.ndarray, sigma: np.ndarray,
           lower_cost: np.ndarray, upper_cost: np.ndarray,
           f_cost: Any,
           Qfact: float,
           srs2AC_cost: Any) -> Dict[str, Any]:
    """Worker function for parallel cost calculation over ix index

    Parameters:
        ix (int): X-coordinate (longitude) index
        nx (int): Total number of x-coordinates
        ny (int): Total number of y-coordinates
        srs (np.ndarray): Source-receptor sensitivity array
        par_init (np.ndarray): Initial guess for parameters
        obs (np.ndarray): Observed data array
        sigma (np.ndarray): Standard deviation or error array
        lower_cost (np.ndarray): Lower bounds for cost calculation
        upper_cost (np.ndarray): Upper bounds for cost calculation
        f_cost (Any): Cost function to minimize
        Qfact (float): Scaling factor for parameters
        srs2AC_cost (Any): Function to convert source parameters to modeled concentrations
    Returns:
        optsed_ix (Dict[str, Any]): Dictionary containing cost calculation results for the given ix index
            - cost (np.ndarray): Array of cost values for each iy
            - accQ (np.ndarray): Array of accumulated Q values for each iy
            - Qs (np.ndarray): Array of optimized parameters for each iy
    """
    print(f'Processing x={ix + 1} of {nx}')
    cost = np.zeros(ny, dtype=float)
    accQ = np.zeros(ny, dtype=float)
    Qs = np.zeros((len(par_init), ny), dtype=float)
    for iy in range(ny):
        result = optimizer(
            srs=srs,
            par_init=par_init,
            obs=obs,
            sigma=sigma,
            lower_cost=lower_cost,
            upper_cost=upper_cost,
            f_cost_func=f_cost,
            ix=ix,
            iy=iy,
            srs2AC_cost=srs2AC_cost
        )
        cost[iy] = result['cost']
        Qs[:, iy] = result['par'] / Qfact
        accQ[iy] = np.sum(Qs[:, iy])
    return {"cost": cost, 
            "accQ": accQ, 
            "Qs": Qs}

def calc_cost(srs: np.ndarray, obs: np.ndarray, 
              Qfact: float, sigma: np.ndarray,
              lower_cost: np.ndarray, upper_cost: np.ndarray,
              par_init: np.ndarray,
              nproc: int,
              settings: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate costs over spatial grid
    
    Parameters:
        srs (np.ndarray): Source-receptor sensitivity array
        obs (np.ndarray): Observed data array
        Qfact (float): Scaling factor for parameters
        sigma (np.ndarray): Standard deviation or error array
        lower_cost (np.ndarray): Lower bounds for cost calculation
        upper_cost (np.ndarray): Upper bounds for cost calculation
        par_init (np.ndarray): Initial guess for parameters
        nproc (int): Number of processes for parallel computation
        settings (Dict[str, Any]): Additional settings for cost calculation
            - 'sourcemodelcost_exec': Function to convert source parameters to modeled concentrations
    
    Returns:
        cost_data (Dict[str, Any]): Dictionary containing overall cost calculation results
            - cost (np.ndarray): 2D array of cost values [nx, ny]
            - Qs (np.ndarray): 3D array of optimized parameters [ntimes, nx, ny]
            - accQ (np.ndarray): 2D array of accumulated Q values [nx, ny]
    """
    if srs.ndim != 4:
        raise ValueError(f"object srs does not have four dimensions: {srs.shape}")
    if np.all(obs < sigma):
        print("WARNING: The high input error might result in an artificially low cost everywhere; consider lowering the input error if problems arise.")

    ntimes, nx, ny, _ = srs.shape

    print("Starting cost calculation")
    srs2AC_cost = settings['sourcemodelcost_exec']

    if nproc == 1:
        out = np.array([worker(ix, nx, ny, srs, par_init, obs, sigma, lower_cost, upper_cost, f_cost, Qfact, srs2AC_cost) for ix in range(nx)])
    else:
        with Pool(processes=nproc) as pool:
            out = pool.starmap(worker, [(ix, nx, ny, srs, par_init, obs, sigma, lower_cost, upper_cost, f_cost, Qfact, srs2AC_cost) for ix in range(nx)])
    
    cost_list = np.vstack([x['cost'] for x in out])
    accQ_list = np.vstack([x['accQ'] for x in out])
    Qs_list = np.vstack([x['Qs'] for x in out])

    return {
        "cost": cost_list,
        "Qs": Qs_list.reshape(len(par_init), nx, ny, order='F'),
        "accQ": accQ_list
    }
import numpy as np
import scipy.special as sc

from typing import Any, Dict, Optional
from scipy.optimize import minimize

def calc_srs_error(srs_raw: np.ndarray, settings: Dict[str, Any], 
                   srs_spread_raw: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate source-receptor sensitivity (SRS) error
    
    Parameters:
        srs_raw (np.ndarray): Raw source-receptor relationship data
        settings (Dict[str, Any]): Configuration settings
            - fcalcsrsspread (Callable): Function to calculate SRS spread
            - srsrelError (float): Relative error to use if no spread is provided
        srs_spread_raw (Optional[np.ndarray]): Spread of raw source-receptor relationship
    Returns:
        srs_error (np.ndarray): Calculated SRS error
    """
    if srs_spread_raw is not None:
        srs_error = srs_spread_raw[0]
    elif isinstance(srs_raw, (list, tuple)) and len(srs_raw) == 1:
        srs_error = settings['srsrelError'] * np.maximum(srs_raw[0], 1e-7)
    else:
        if not isinstance(srs_raw, (list, tuple)):
            raise ValueError('srs_raw must be a list of ensemble arrays when srs_spread_raw is None')
        stacked = np.stack(srs_raw, axis=-1)
        try:
            srs_error = np.apply_along_axis(settings['fcalcsrsspread'], -1, stacked)
        except Exception:
            f = np.vectorize(settings['fcalcsrsspread'])
            srs_error = f(stacked)

    if np.any(srs_error == 0):
        print('srs_error contains zero(s), resetting to 10^-7')
        srs_error = np.maximum(srs_error, 1e-7)
    return srs_error

def calc_mod_error(srs: np.ndarray, srs_error: np.ndarray, defaultRelErr: float = 0.5) -> np.ndarray:
    """Calculate model error from SRS and SRS error
    
    Parameters:
        srs (np.ndarray): Source-receptor relationship data
        srs_error (np.ndarray): Error in SRS data
        defaultRelErr (float): Default relative error to use if sum(srs) == 0
    Returns:
        mod_error (np.ndarray): Calculated model error
    """
    ntimes = srs.shape[0]
    nsamples = srs.shape[3]
    mod_error = np.zeros((ntimes, nsamples))
    for isample in range(nsamples):
        for itime in range(ntimes):
            srs_error_sum = np.sum(srs_error[itime, :, :, isample])
            srs_sum = np.sum(np.maximum(srs[itime, :, :, isample], 1e-7))
            if np.sum(srs[itime, :, :, isample]) == 0:
                print(f'modelError: sum(srs) = 0 for time {itime + 1} and sample {isample + 1} - using settings[srsrelError]')
                mod_error[itime, isample] = defaultRelErr
            else:
                mod_error[itime, isample] = srs_error_sum / srs_sum
    return mod_error

def fitInvGamma(srs: np.ndarray, srs_error: np.ndarray) -> Dict[str, Any]:
    """Fit inverse gamma distribution to SRS relative errors

    Parameters:
        srs (np.ndarray): Source-receptor relationship data
        srs_error (np.ndarray): Error in SRS data
    Returns:
        fit_params (Dict[str, Any]): Fitted parameters of the inverse gamma distribution
            - s (np.ndarray): s parameters
            - a (np.ndarray): a parameters
            - b (np.ndarray): b parameters
    """
    nsamples = srs.shape[3]

    s = np.zeros(nsamples)
    a = np.zeros(nsamples)
    b = np.zeros(nsamples)

    def fInvGamma(x, s, a, b):
        return 2 * a**b / sc.gamma(b) * (s / x)**(2 * b) * np.exp(-a * s**2 / x**2) * 1 / x

    def fcost(par, x, y):
        return np.sum((y - fInvGamma(x, s=par[0], a=par[1], b=par[2]))**2)

    for isample in range(nsamples):
        srsaux = srs[:, :, :, isample]
        srs_erroraux = srs_error[:, :, :, isample]
        srs_sel = np.where(srsaux > 1e-7)
        srsrelerror = srs_erroraux[srs_sel] / srsaux[srs_sel]
        srsrelerror = srsrelerror[srsrelerror < 10]
        if len(srsrelerror) > 1000:
            nbreaks = np.concatenate((np.arange(0., 1.05, 0.05), np.arange(1.2, 10.2, 0.2)))
        elif len(srsrelerror) > 500:
            nbreaks = np.concatenate((np.arange(0., 1.1, 0.1), np.arange(1.5, 10.5, 0.5)))
        else:
            nbreaks = np.concatenate((np.arange(0., 1.2, 0.2), np.arange(2, 11, 1)))

        hist, edges = np.histogram(srsrelerror, bins=nbreaks, density=True)
        mids = 0.5 * (edges[:-1] + edges[1:])

        par0 = np.array([0.5, 1.0 / np.pi, 1.0])
        res = minimize(
            fcost,
            par0,
            args=(mids, hist),
            method='Nelder-Mead',
            options={
                'maxiter': 200000,
                'xatol': 1e-12,
                'fatol': 1e-12,
                'disp': False
            }
        )
        s[isample] = float(res.x[0])
        a[isample] = float(res.x[1])
        b[isample] = float(res.x[2])
    return {'s': s, 'a': a, 'b': b}

def fitCtrue(srs_raw: np.ndarray) -> Dict[str, Any]:
    """Fit Ctrue distribution to SRS data
    
    Parameters:
        srs_raw (np.ndarray): Raw source-receptor relationship data
    Returns:
        fit_params (Dict[str, Any]): Fitted parameters of the Ctrue distribution
            - s (np.ndarray): s parameters
            - a (np.ndarray): a parameters
            - b (np.ndarray): b parameters
    """
    first = srs_raw[0]
    shape = (*first.shape, len(srs_raw))
    srs = np.zeros(shape, dtype=float)
    nsamples = srs.shape[3]

    for i, member in enumerate(srs_raw):
        srs[..., i] = member

    def srs2matrix(subsrs: np.ndarray, srs_threshold: float = np.exp(-20)) -> np.ndarray:
        subsrs_reshaped = subsrs.reshape(-1, subsrs.shape[-1])
        subsrs_median = np.median(subsrs_reshaped, axis=1)
        mask = subsrs_median > srs_threshold
        subsrs_filtered = subsrs_reshaped[mask]
        subsrs_normalized = subsrs_filtered / subsrs_median[mask][:, None]
        subsrs_log = np.log(subsrs_normalized[subsrs_normalized != 0])
        return subsrs_log

    def dctrue3(c_true: np.ndarray, c_mod: float, sigma: float, alpha: float = 1.0 / np.pi, beta: float = 1.0, eps: float = 1e-6) -> np.ndarray:
        Y = alpha**beta * sc.gamma(beta + 0.5) / (np.sqrt(2 * np.pi) * sc.gamma(beta))
        return Y / (sigma * (alpha + (c_true - c_mod)**2 / (2 * sigma**2))**(beta + 0.5))

    def fcost(par: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        return np.sum((y - dctrue3(x, 0.0, np.log(1 + par[0]), par[1], par[2]))**2)

    def fit_c_true_inner(subsrs: np.ndarray, fcost: Any, nbreaks: int = 50) -> Dict[str, Any]:
        density, edges = np.histogram(subsrs, bins=nbreaks, density=True)
        mids = 0.5 * (edges[:-1] + edges[1:])

        res = minimize(
            fcost,
            x0=np.array([0.5, 1/np.pi, 1.0]),
            args=(mids, density),
            method="Nelder-Mead"
        )

        return {
            "s": res.x[0],
            "a": res.x[1],
            "b": res.x[2],
            "cost": res.fun,
            "hist_mids": mids,
            "hist_density": density
        }

    s = np.zeros(nsamples)
    a = np.zeros(nsamples)
    b = np.zeros(nsamples)

    for isample in range(nsamples):
        fit = fit_c_true_inner(srs2matrix(srs[:, :, :, isample]), fcost=fcost)
        s[isample] = fit['s']
        a[isample] = fit['a']
        b[isample] = fit['b']

    return {'s': s, 'a': a, 'b': b}

def calcModErr_invGamma(settings: Dict[str, Any], srs: np.ndarray, 
                        srs_raw: np.ndarray, srs_spread_raw: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Calculate model error using inverse gamma fitting
    
    Parameters:
        settings (Dict[str, Any]): Configuration settings
            - mod_error_mode (str): Mode for model error calculation
            - srsrelError (float): Default relative error
            - zalpha (float): Alpha parameter for inverse gamma
            - zbeta (float): Beta parameter for inverse gamma
            - members (List[Any]): Ensemble members
        srs (np.ndarray): Source-receptor relationship data
        srs_raw (np.ndarray): Raw source-receptor relationship data
        srs_spread_raw (Optional[np.ndarray]): Spread of raw source-receptor relationship data
    Returns:
        result (Dict[str, Any]): Calculated model error and parameters
            - sigmas (np.ndarray): S parameters for model error
            - alphas (np.ndarray): Alpha parameters for inverse gamma
            - betas (np.ndarray): Beta parameters for inverse gamma
    """
    ntimes = srs.shape[0]
    nsamples = srs.shape[3]

    mode = settings['mod_error_mode']
    if mode == 'indep_par':
        srs_error = calc_srs_error(srs_raw, settings, srs_spread_raw=srs_spread_raw)
        sigmas = calc_mod_error(srs, srs_error, defaultRelErr=settings['srsrelError'])
        alphas = np.full(nsamples, settings['zalpha'])
        betas = np.full(nsamples, settings['zbeta'])
    elif mode in ('fitInvGamma', 'fitCtrue'):
        members = settings['members']
        if len(members) <= 1:
            raise ValueError(f"Cannot use mod_error_mode={mode} when no ensemble available")

        if mode == 'fitInvGamma':
            srs_error = calc_srs_error(srs_raw, settings, srs_spread_raw=srs_spread_raw)
            fit = fitInvGamma(srs, srs_error)
        else:
            fit = fitCtrue(srs_raw)

        sigmas = np.zeros((ntimes, nsamples))
        for i in range(ntimes):
            sigmas[i, :] = np.asarray(fit['s'])
        alphas = np.asarray(fit['a'])
        betas = np.asarray(fit['b'])
    else:
        raise ValueError(f"Invalid mod_error_mode='{mode}'")

    return {'sigmas': sigmas, 'alphas': alphas, 'betas': betas}
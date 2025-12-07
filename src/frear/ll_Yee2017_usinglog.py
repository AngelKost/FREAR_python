#FIXME: remove unnesessary duplicates and create separate Yee2017_tools.py for joint functions
import numpy as np
import scipy.special as sc

from typing import Dict, Any, Optional
from scipy.stats import norm
from scipy import integrate

from frear.srs2AC_aux import getACsubset

def dctrue(ctrue: np.ndarray, logmod: np.ndarray, sigma_tot: np.ndarray, 
           eps: float, Y: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Probability density function for true concentration given modelled concentration
    
    Parameters:
        ctrue (np.ndarray): True concentration values
        logmod (np.ndarray): Logarithm of modelled concentration values
        sigma_tot (np.ndarray): Total standard deviation values
        eps (float): Small constant to avoid log(0)
        Y (np.ndarray): Parameter Y
        alpha (np.ndarray): Parameter alpha
        beta (np.ndarray): Parameter beta

    Returns:
        c_true (np.ndarray): Probability density values
    """
    log_ctrue = np.log(ctrue + eps)
    logs = np.log(ctrue + eps + sigma_tot) - log_ctrue
    coeff = Y / logs
    power = beta + 0.5
    dc = (log_ctrue - logmod) ** 2
    denominator = (alpha + dc / (2 * logs ** 2)) ** power
    return coeff / denominator

def p_ndet_ctrue(ctrue: np.ndarray, L_C: np.ndarray, k_alpha: float = 1.645) -> np.ndarray:
    """Probability of non-detection given true concentration
    
    Parameters:
        ctrue (np.ndarray): True concentration values
        L_C (np.ndarray): Critical level values
        k_alpha (float): Constant for detection limit calculation
    Returns:
        p_ndet (np.ndarray): Probability of non-detection values
    """
    return norm.cdf(L_C, loc=ctrue, scale=L_C / k_alpha)

def p_det_ctrue(ctrue: np.ndarray, L_C: np.ndarray, k_alpha: float = 1.645) -> np.ndarray:
    """Probability of detection given true concentration
    
    Parameters:
        ctrue (np.ndarray): True concentration values
        L_C (np.ndarray): Critical level values
        k_alpha (float): Constant for detection limit calculation
    Returns:
        p_det (np.ndarray): Probability of detection values
    """
    return 1.0 - p_ndet_ctrue(ctrue=ctrue, L_C=L_C, k_alpha=k_alpha)

def p_true_det_cmod(L_C: np.ndarray, logmod: np.ndarray, sigma_tot: np.ndarray, eps: float, Y: np.ndarray, 
                    alpha: np.ndarray, beta: np.ndarray, zdctruenorm: np.ndarray, upper: Optional[float] = None) -> np.ndarray:
    """Probability of a true detection given modelled concentration
    
    Parameters:
        L_C (np.ndarray): Critical level values
        logmod (np.ndarray): Logarithm of modelled concentration values
        sigma_tot (np.ndarray): Total standard deviation values
        eps (float): Small constant to avoid log(0)
        Y (np.ndarray): Parameter Y
        alpha (np.ndarray): Parameter alpha
        beta (np.ndarray): Parameter beta
        zdctruenorm (np.ndarray): Normalization factor for true concentration density
        upper (Optional[float]): Upper limit for integration
    Returns:
        p_true_det (np.ndarray): Probability of true detection values
    """
    if upper is None:
        upper = 20 * L_C
    integral, _ = integrate.quad(
        lambda ctrue: dctrue(ctrue, logmod, sigma_tot, eps, Y, alpha, beta),
        L_C, upper
    )
    return integral / zdctruenorm

def p_true_ndet_cmod(L_C: np.ndarray, logmod: np.ndarray, sigma_tot: np.ndarray, eps: float, Y: np.ndarray, 
                     alpha: np.ndarray, beta: np.ndarray, zdctruenorm: np.ndarray, upper: Optional[float] = None) -> np.ndarray:
    """Probability of a true non-detection given modelled concentration
    
    Parameters:
        L_C (np.ndarray): Critical level values
        logmod (np.ndarray): Logarithm of modelled concentration values
        sigma_tot (np.ndarray): Total standard deviation values
        eps (float): Small constant to avoid log(0)
        Y (np.ndarray): Parameter Y
        alpha (np.ndarray): Parameter alpha
        beta (np.ndarray): Parameter beta
        zdctruenorm (np.ndarray): Normalization factor for true concentration density
        upper (Optional[float]): Upper limit for integration
    Returns:
        p_true_ndet (np.ndarray): Probability of true non-detection values
    """
    if upper is None:
        upper = 20 * L_C
    integral, _ = integrate.quad(
        lambda ctrue: dctrue(ctrue, logmod, sigma_tot, eps, Y, alpha, beta),
        0, L_C
    )
    return integral / zdctruenorm

def p_cdet_true_ndet(cdet: np.ndarray, L_C: np.ndarray, k_alpha: float = 1.645) -> np.ndarray:
    """Probability of a detecting cdet given a true non-detection (false alarm)
    
    Parameters:
        cdet (np.ndarray): Detected concentration values
        L_C (np.ndarray): Critical level values
        k_alpha (float): Constant for detection limit calculation
    Returns:
        p_cdet_true_ndet (np.ndarray): Probability of detecting cdet given a true non-detection
    """
    aux = norm.cdf(0, loc=cdet, scale=L_C / k_alpha)
    numerator = norm.cdf(L_C, loc=cdet, scale=L_C / k_alpha) - aux
    denominator = 1.0 - aux
    return numerator / denominator

def p_ndet_cmod(L_C: np.ndarray, logmod: np.ndarray, sigma_mod: np.ndarray, eps: float, Y: np.ndarray, 
                   alpha: np.ndarray, beta: np.ndarray, zdctruenorm: np.ndarray) -> np.ndarray:
    """Probability of a non-detection (true non-detection or miss) given c_mod
    
    Parameters:
        L_C (np.ndarray): Critical level values
        logmod (np.ndarray): Logarithm of modelled concentration values
        sigma_mod (np.ndarray): Standard deviation of modelled concentration values
        eps (float): Small constant to avoid log(0)
        Y (np.ndarray): Parameter Y
        alpha (np.ndarray): Parameter alpha
        beta (np.ndarray): Parameter beta
        zdctruenorm (np.ndarray): Normalization factor for true concentration density
    Returns:
        p_ndet (np.ndarray): Probability of non-detection values
    """
    MDC = 2 * L_C
    integral, _ = integrate.quad(
        lambda ctrue: p_ndet_ctrue(ctrue, L_C) * dctrue(ctrue, logmod, sigma_mod, eps, Y, alpha, beta) / zdctruenorm,
        0, 10 * MDC
    )
    return integral

def ll_Yee2017log_det(ac: np.ndarray, 
                      alphas: np.ndarray = np.array(1.0 / np.pi), betas: np.ndarray = np.array(1.0), 
                      Ys: Optional[np.ndarray] = None, 
                      ll_noRelease: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate log-likelihoods for detections according to Yee et al.(2017)
    
    Parameters:
        ac (Dict[str, Any]): Dictionary of data
            - obs (np.ndarray): Observed concentration values
            - mod (np.ndarray): Modeled concentration values
            - sigma_obs (np.ndarray): Observation standard deviation values
            - sigma_mod (np.ndarray): Model standard deviation values
            - MDC (np.ndarray): Method detection limit values
        alphas (np.ndarray): Parameter alpha values
        betas (np.ndarray): Parameter beta values
        Ys (Optional[np.ndarray]): Parameter Y values
        ll_noRelease (Optional[np.ndarray]): Precalculated log-likelihoods for no release cases
    Returns:
        likes (np.ndarray): Log-likelihood values
    """
    nsamples = len(ac['obs'])
    if Ys is None:
        if np.isscalar(alphas):
            alphas = np.full(nsamples, alphas)
        if np.isscalar(betas):
            betas = np.full(nsamples, betas)
        Ys = alphas ** betas * sc.gamma(betas + 0.5) / (np.sqrt(2 * np.pi) * sc.gamma(betas))

    eps = 1e-6
    sigma_tot = np.sqrt(ac['sigma_obs'] ** 2 + ac['sigma_mod'] ** 2)
    MDC = ac['MDC']
    L_C = MDC / 2.0
    likes = np.zeros(nsamples)

    # If precalculated values of ll exist (for ac['mod'] == 0), then make use of it
    if ll_noRelease is not None:
        iprecalc = np.where(ac['mod'] == 0)[0]
        icalc = np.where(ac['mod'] != 0)[0]
        likes[iprecalc] = ll_noRelease[iprecalc]
    else:
        icalc = np.arange(nsamples)

    if len(icalc) > 0:
        cond_icalcfast = (ac['obs'] > L_C * 5) & (ac['mod'] > L_C * 5)
        icalcfast = np.where(cond_icalcfast)[0]
        icalc = np.setdiff1d(icalc, icalcfast)
        logmod = np.log(ac['mod'] + eps)

        if len(icalcfast) > 0:
            # Assuming a detection, the likelihood of c_det given c_mod
            llcdet_cmod = dctrue(
                ctrue=ac['obs'][icalcfast],
                logmod=logmod[icalcfast],
                sigma_tot=sigma_tot[icalcfast],
                eps=eps,
                Y=Ys[icalcfast],
                alpha=alphas[icalcfast],
                beta=betas[icalcfast]
            )
            likes[icalcfast] = np.log(llcdet_cmod)
        if len(icalc) > 0:
            # Slow calculation including possibility of false alarm
            lowLim = np.where(ac['mod'] < MDC,
                              0,
                              np.maximum(0, ac['mod'] - 100 * ac['sigma_mod']))
            uppLim = np.where(ac['mod'] < MDC,
                              MDC * 10,
                              np.maximum(MDC * 10,
                                         ac['mod'] + np.minimum(10 * ac['mod'], 100 * ac['sigma_mod'])))
            zdctruenorm = np.ones(nsamples)
            for i in icalc:
                integral, _ = integrate.quad(
                    lambda ctrue: dctrue(ctrue, logmod[i], sigma_tot[i], eps, Ys[i], alphas[i], betas[i]),
                    lowLim[i],
                    uppLim[i]
                )
                zdctruenorm[i] = integral
            lltruendet_cmod = np.array([
                p_true_ndet_cmod(
                    L_C=L_C[i],
                    logmod=logmod[i],
                    sigma_tot=sigma_tot[i],
                    eps=eps,
                    Y=Ys[i],
                    alpha=alphas[i],
                    beta=betas[i],
                    zdctruenorm=zdctruenorm[i]
                ) 
                for i in icalc
            ])
            lltruedet_cmod = 1.0 - lltruendet_cmod

            lltruendet_cmod = np.maximum(lltruendet_cmod, 0)
            lltruedet_cmod = np.maximum(lltruedet_cmod, 0)

            llcdet_truendet = p_cdet_true_ndet(
                cdet=ac['obs'][icalc],
                L_C=L_C[icalc]
            )
            
            llcdet_cmod = dctrue(
                ctrue=ac['obs'][icalc],
                logmod=logmod[icalc],
                sigma_tot=sigma_tot[icalc],
                eps=eps,
                Y=Ys[icalc],
                alpha=alphas[icalc],
                beta=betas[icalc]
            )
            likes[icalc] = np.log(llcdet_cmod * lltruedet_cmod + llcdet_truendet * lltruendet_cmod)
    return likes

def ll_Yee2017log_nondet(ac: Dict[str, Any], 
                         alphas: np.ndarray = np.array(1.0 / np.pi), betas: np.ndarray = np.array(1.0),
                         Ys: Optional[np.ndarray] = None, 
                         ll_noRelease: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate log-likelihoods for non-detections according to Yee et al.(2017)
    
    Parameters:
        ac (Dict[str, Any]): Dictionary of data
            - obs (np.ndarray): Observed concentration values
            - mod (np.ndarray): Modeled concentration values
            - sigma_obs (np.ndarray): Observation standard deviation values
            - sigma_mod (np.ndarray): Model standard deviation values
            - MDC (float): Minimum detection concentration
        alphas (np.ndarray): Parameter alpha values
        betas (np.ndarray): Parameter beta values
        Ys (Optional[np.ndarray]): Optional precomputed Y values
        ll_noRelease (Optional[np.ndarray]): Optional precomputed log-likelihood values for no release
    Returns:
        likes (np.ndarray): Log-likelihood non-detection values
    """
    nsamples = len(ac['obs'])
    if Ys is None:
        if np.isscalar(alphas):
            alphas = np.full(nsamples, alphas)
        if np.isscalar(betas):
            betas = np.full(nsamples, betas)
        Ys = alphas ** betas * sc.gamma(betas + 0.5) / (np.sqrt(2 * np.pi) * sc.gamma(betas))

    eps = 1e-6
    sigma_tot = np.sqrt(ac['sigma_obs'] ** 2 + ac['sigma_mod'] ** 2)
    MDC = ac['MDC']
    L_C = MDC / 2.0
    likes = np.zeros(nsamples)

    # If precalculated values of ll exist (for ac['mod'] == 0), then make use of it
    if ll_noRelease is not None:
        iprecalc = np.where(ac['mod'] == 0)[0]
        icalc = np.where(ac['mod'] != 0)[0]
        likes[iprecalc] = ll_noRelease[iprecalc]
    else:
        icalc = np.arange(nsamples)

    if len(icalc) > 0:
        ac['mod'] = np.maximum(ac['mod'], 0.01 * MDC)  # Some problems for very small ac['mod']
        logmod = np.log(ac['mod'] + eps)
        lowLim = np.where(ac['mod'] < MDC,
                          0,
                          np.maximum(0, ac['mod'] - 100 * ac['sigma_mod']))
        uppLim = np.where(ac['mod'] < MDC,
                          10 * MDC,
                          np.maximum(10 * MDC, ac['mod'] + np.minimum(10 * ac['mod'], 100 * ac['sigma_mod'])))
        zdctruenorm = np.ones(len(nsamples))
        for i in icalc:
            integral, _ = integrate.quad(
                lambda ctrue: dctrue(ctrue, logmod[i], sigma_tot[i], eps, Ys[i], alphas[i], betas[i]),
                lowLim[i],
                uppLim[i]
            )
            zdctruenorm[i] = integral
        llndet_cmod = np.array([p_ndet_cmod(L_C=L_C[i], 
                                            logmod=logmod[i], 
                                            sigma_mod=ac['sigma_mod'][i], 
                                            eps=eps,
                                            Y=Ys[i], 
                                            alpha=alphas[i], 
                                            beta=betas[i], 
                                            zdctruenorm=zdctruenorm[i]) 
                               for i in icalc])
        likes = np.log(llndet_cmod)
    return likes

def getLikes_Yee2017log(ac: Dict[str, Any], ll_out: Optional[Dict[str, Any]] = {
    "alphas": np.array(1/np.pi),
    "betas": np.array(1),
    "ll_noRelease": None,
    "Ys": None,
    "ll_factors": None
}) -> Dict[str, Any]:
    """Calculate log-likelihoods according to Yee et al.(2017) for both detections and non-detections
    
    Parameters:
        ac (Dict[str, Any]): Dictionary of data
            - obs (np.ndarray): Observed concentration values
            - mod (np.ndarray): Modeled concentration values
            - sigma_obs (np.ndarray): Observation standard deviation values
            - sigma_mod (np.ndarray): Model standard deviation values
            - MDC (float): Minimum detection concentration
        ll_out (Optional[Dict[str, Any]]): Dictionary of precomputed parameters
            - alphas (np.ndarray): Parameter alpha values
            - betas (np.ndarray): Parameter beta values
            - ll_noRelease (np.ndarray): Log-likelihood values
            - Ys (np.ndarray): Precomputed Ys values
            - ll_factors (np.ndarray): Log-likelihood factors
    Returns:
        Dict[str, Any]: Dictionary containing log-likelihoods
            - likes (np.ndarray): Log-likelihood values
    """
    if np.isscalar(ll_out['alphas']) and ll_out['alphas'] == 1:
        alphas = np.full(len(ac['obs']), ll_out['alphas'])
    else:
        alphas = ll_out['alphas']

    if np.isscalar(ll_out['betas']) and ll_out['betas'] == 1:
        betas = np.full(len(ac['obs']), ll_out['betas'])
    else:
        betas = ll_out['betas']

    ll_noRelease = ll_out['ll_noRelease']

    if ll_out['Ys'] is None:
        Ys = alphas ** betas * sc.gamma(betas + 0.5) / (np.sqrt(2 * np.pi) * sc.gamma(betas))
    else:
        Ys = ll_out['Ys']

    if ll_out['ll_factors'] is None:
        ll_factors = np.ones(len(ac['obs']))
    else:
        ll_factors = ll_out['ll_factors']

    # Calculate likelihood for detections and non-detections separately
    idet = np.where(ac['obs'] != 0)[0]
    inondet = np.where(ac['obs'] == 0)[0]

    ac_det = getACsubset(ac, idet)
    ac_nondet = getACsubset(ac, inondet)

    likes = np.zeros(len(ac['obs']))
    if len(idet) != 0:
        likes[idet] = ll_Yee2017log_det(ac=ac_det,
                                     alphas=alphas[idet],
                                     betas=betas[idet],
                                     Ys=Ys[idet],
                                     ll_noRelease=None if ll_noRelease is None else ll_noRelease[idet])
    if len(inondet) != 0:
        likes[inondet] = ll_Yee2017log_nondet(ac=ac_nondet,
                                           alphas=alphas[inondet],
                                           betas=betas[inondet],
                                           Ys=Ys[inondet],
                                           ll_noRelease=None if ll_noRelease is None else ll_noRelease[inondet])
    likes = likes * ll_factors
    return {"likes": likes}

def preCalc_ll_noRelease(settings: Dict[str, Any], lower_bayes: np.ndarray, upper_bayes: np.ndarray,
                           ACfun_bayes: Any, ll_out: Optional[Dict[str, Any]] = {
                               "alphas": np.array(1/np.pi),
                               "betas": np.array(1),
                               "ll_noRelease": None,
                               "Ys": None
                           }) -> np.ndarray:
    """Precalculate log-likelihoods for no release cases according to Yee et al.(2017)
    
    Parameters:
        settings (Dict[str, Any]): Settings dictionary
            - parnames (List[str]): List of parameter names
        lower_bayes (np.ndarray): Lower bounds for Bayesian parameters
        upper_bayes (np.ndarray): Upper bounds for Bayesian parameters
        ACfun_bayes (Any): Function to compute AC values
        ll_out (Optional[Dict[str, Any]]): Dictionary of precomputed parameters
            - alphas (np.ndarray): Parameter alpha values
            - betas (np.ndarray): Parameter beta values
            - ll_noRelease (np.ndarray): Log-likelihood values
            - Ys (np.ndarray): Parameter Y values
    Returns:
        ll_noRelease (np.ndarray): Log-likelihood values for no release cases
    """
    param_noRelease = np.array(np.array([(u + l) / 2.0 for u, l in zip(upper_bayes, lower_bayes)]))
    q_indices = [i for i, name in enumerate(settings['parnames']) if 'log10_Q' in name]
    for qi in q_indices:
        param_noRelease[qi] = -np.inf  # since Q <- 10^Q
    ac_noRelease = ACfun_bayes(param_noRelease)
    ll_noRelease = getLikes_Yee2017log(ac_noRelease, ll_out)["likes"]
    return ll_noRelease

def preCalc_ll_factors(settings: Dict[str, Any], lower_bayes: np.ndarray, upper_bayes: np.ndarray,
                        ACfun_bayes: Any, ll_out: Dict[str, Any] = {'alphas': 1. / np.pi, 
                                                                    'betas': 1,
                                                                    'll_noRelease': None,
                                                                    'Ys': None}) -> np.ndarray:
    """Precalculate log-likelihood factors according to Yee et al.(2017)
    
    Parameters:
        settings (Dict[str, Any]): Settings dictionary
            - parnames (List[str]): List of parameter names
        lower_bayes (np.ndarray): Lower bounds for Bayesian parameters
        upper_bayes (np.ndarray): Upper bounds for Bayesian parameters
        ACfun_bayes (Any): Function to compute AC values
        ll_out (Dict[str, Any]): Dictionary of precomputed parameters
            - alphas (np.ndarray): Parameter alpha values
            - betas (np.ndarray): Parameter beta values
            - ll_noRelease (np.ndarray): Log-likelihood values
            - Ys (np.ndarray): Parameter Y values
    Returns:
        ll_factors (np.ndarray): Log-likelihood factors
    """
    ac_dummy = ACfun_bayes(np.array([(u + l) / 2.0 for u, l in zip(upper_bayes, lower_bayes)]))
    ll_factors = np.ones(len(ac_dummy['obs']))
    for iobs in range(len(ac_dummy['obs'])):
        if ac_dummy['obs'][iobs] == 0:
            sigma_obs = ac_dummy['sigma_obs'][iobs]
            MDC = ac_dummy['MDC'][iobs]
            sigma_mod = ac_dummy['sigma_mod'][iobs]
            ac1 = {'obs': 0, 'sigma_obs': sigma_obs, 'MDC': MDC, 'mod': MDC / 2, 'sigma_mod': sigma_mod}
            ac2 = {'obs': 0, 'sigma_obs': sigma_obs, 'MDC': MDC, 'mod': 0, 'sigma_mod': sigma_mod}
            ac3 = {'obs': MDC, 'sigma_obs': sigma_obs, 'MDC': MDC, 'mod': MDC / 2, 'sigma_mod': sigma_mod}
            ac4 = {'obs': MDC, 'sigma_obs': sigma_obs, 'MDC': MDC, 'mod': MDC, 'sigma_mod': sigma_mod}
            deltall1 = getLikes_Yee2017log(ac1, ll_out)["likes"] - getLikes_Yee2017log(ac2, ll_out)["likes"]
            deltall2 = getLikes_Yee2017log(ac3, ll_out)["likes"] - getLikes_Yee2017log(ac4, ll_out)["likes"]
            ll_factors[iobs] = deltall2 / deltall1
    return ll_factors
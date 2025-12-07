import numpy as np

from typing import Dict, Any, Callable, Optional

def density_ffact(upper_bayes: np.ndarray, lower_bayes: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return a density function for uniform prior between lower_bayes and upper_bayes
    
    Parameters:
        upper_bayes (np.ndarray): Upper bounds of the parameters
        lower_bayes (np.ndarray): Lower bounds of the parameters

    Returns:
        density (Callable[[np.ndarray], np.ndarray]): Density function
    """
    if len(upper_bayes) != len(lower_bayes):
        raise ValueError("Length of 'upper_bayes' and 'lower_bayes' must be the same")
    def density(par: np.ndarray) -> np.ndarray:
        if len(upper_bayes) != len(par):
            raise ValueError("Length of 'par' must be the same as 'upper_bayes' and 'lower_bayes'")
        par = np.asarray(par)
        dens = 0.0

        for i in range(len(par)):
            if lower_bayes[i] == upper_bayes[i]:
                out = 0.0
            elif lower_bayes[i] > upper_bayes[i]:
                if par[i] < lower_bayes[i]:
                    if par[i] < -180:
                        out = -np.inf
                    else:
                        out = -np.log(lower_bayes[i] + 180)
                else:
                    if par[i] < upper_bayes[i] or par[i] > 180:
                        out = -np.inf
                    else:
                        out = -np.log(180 - upper_bayes[i])
            else:
                if par[i] < lower_bayes[i] or par[i] > upper_bayes[i]:
                    out = -np.inf
                else:
                    out = -np.log(upper_bayes[i] - lower_bayes[i])

            dens += out
        return dens
    return density

def sampler_ffact(upper_bayes: np.ndarray, lower_bayes: np.ndarray) -> Callable[[int], np.ndarray]:
    """Return a sampler function for uniform prior between lower_bayes and upper_bayes
    
    Parameters:
        upper_bayes (np.ndarray): Upper bounds of the parameters
        lower_bayes (np.ndarray): Lower bounds of the parameters

    Returns:
        sampler (Callable[[int], np.ndarray]): Sampler function
    """
    if len(upper_bayes) != len(lower_bayes):
        raise ValueError("Length of 'upper_bayes' and 'lower_bayes' must be the same")
    def sampler(n: int = 1) -> np.ndarray:
        N = len(upper_bayes)
        samples = np.zeros((n, N))
        for i in range(N):
            if lower_bayes[i] == upper_bayes[i]:
                samples[:, i] = lower_bayes[i]
            elif lower_bayes[i] > upper_bayes[i]:
                p_bound = (180 - lower_bayes[i])/(360 + upper_bayes[i] - lower_bayes[i])
                sample = np.random.rand(n)
                idx_lower = sample < p_bound
                samples[idx_lower, i] = np.random.uniform(lower_bayes[i], 180, size=np.sum(idx_lower))
                samples[~idx_lower, i] = np.random.uniform(-180, upper_bayes[i], size=np.sum(~idx_lower))
            else:
                samples[:, i] = np.random.uniform(lower_bayes[i], upper_bayes[i], size=n)
        return samples
    return sampler

def make_prior(density: Callable[[np.ndarray], float],
               sampler:  Callable[[int], np.ndarray],
               lower_bayes: np.ndarray,
               upper_bayes: np.ndarray,
               best: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Create prior object with density and sampler functions and bounds
    
    Parameters:
        density (Callable[[np.ndarray], float]): Density function
        sampler (Callable[[int], np.ndarray]): Sampler function
        lower_bayes (np.ndarray): Lower bounds of the parameters
        upper_bayes (np.ndarray): Upper bounds of the parameters
        best (Optional[np.ndarray]): Best guess for the parameters.

    Returns:
        prior_data (Dict[str, Any]): dictionary containing information about the prior distribution
            - density (Callable[[np.ndarray], float]): density function
            - sampler (Callable[[int], np.ndarray]): sampler function
            - lower_bayes (np.ndarray): lower bounds
            - upper_bayes (np.ndarray): upper bounds
            - best (np.ndarray): best guess for the parameters
    """
    if np.any(lower_bayes > upper_bayes):
        raise ValueError("Prior has lower_bayes > upper_bayes!")
    if best is None:
        best = np.array([(lower_bayes[i] + upper_bayes[i]) / 2 for i in range(len(lower_bayes))])

    def prior_wrapper(x: np.ndarray) -> float:
        if lower_bayes is not None:
            if np.any(x < lower_bayes):
                return -np.inf
        if upper_bayes is not None:
            if np.any(x > upper_bayes):
                return -np.inf
        try:
            out = density(x)
        except Exception as e:
            print("Problem in the prior:", e)
            return -np.inf
        if out == np.inf:
            raise ValueError("Inf encountered in prior")
        return out

    def parallel_density(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return prior_wrapper(x)
        elif x.ndim == 2:
            return np.apply_along_axis(prior_wrapper, 1, x)
        else:
            raise ValueError("Parameter must be a vector or a matrix")

    out = {
        'density': parallel_density,
        'sampler': sampler,
        'lower_bayes': lower_bayes,
        'upper_bayes': upper_bayes,
        'best': best
    }
    return out

def make_likelihood(likelihood: Callable[[np.ndarray], float]) -> Dict[str, Any]:
    """Create likelihood function wrapper for multidimensional evaluation
    
    Parameters:
        likelihood (Callable[[np.ndarray], float]): Likelihood function
    Returns:
        likelihood_data (Dict[str, Any]): dictionary containing the likelihood density function
            - density (Callable[[np.ndarray], np.ndarray]): density function
    """
    def likelihood_wrapper(x: np.ndarray) -> float:
        out = None
        try:
            y = likelihood(x)
            if np.any(y == np.inf) or np.any(np.isnan(y)) or np.any(np.isnan(y)) or not np.issubdtype(type(y), np.number):
                print(f"BayesianTools warning: positive Inf or NA / nan values, or non-numeric values occured in the likelihood. Setting likelihood to -Inf.\n Original value was {y} for parameters {x}\n\n ")
                y = -np.inf
            out = y
        except Exception as e:
            print(f"*** Problem encountered in the calculation of the likelihood ***\n* Error message was: {e}\n* Parameter values were: {x}\n* Set result of the parameter evaluation to -Inf \n***************************************************************")
            out = -np.inf
        return out

    def parallel_density(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return likelihood_wrapper(x)
        elif x.ndim == 2:
            return np.apply_along_axis(likelihood_wrapper, 1, x)
        else:
            raise ValueError("Parameter must be a vector or a matrix")
    return {'density': parallel_density}

def make_posterior(prior: Dict[str, Any],
                   likelihood: Dict[str, Any],
                   beta: float = 1.0) -> Dict[str, Any]:
    """Create posterior distribution function from prior and likelihood
    
    Parameters:
        prior (Dict[str, Any]): Prior distribution dictionary containing density function
            - density (Callable[[np.ndarray], float]): density function
        likelihood (Dict[str, Any]): Likelihood distribution dictionary containing density function
            - density (Callable[[np.ndarray], float]): density function
        beta (float): Scaling factor for the likelihood
    
    Returns:
        posterior_data (Dict[str, Any]): dictionary containing the posterior density function
            - density (Callable[[np.ndarray], Dict[str, Any]]): density function returning prior, likelihood, and posterior values
    """
    def posterior(x: np.ndarray) -> Dict[str, Any]:
        if x.ndim == 1:
            pr = po = prior['density'](x)
            if pr != -np.inf:
                ll = likelihood['density'](x) * beta
                po = po + ll
            else:
                ll = -np.inf
            return {'prior': pr, 'likelihood': ll, 'posterior': po}
        elif x.ndim == 2:
            pr = prior['density'](x)
            iaccept = (pr != -np.inf)
            if x.shape[1] == 1:
                ll = likelihood['density'](x[iaccept, :]) * beta  # why needed, why not just x?
            else:
                if True in iaccept:
                    ll = likelihood['density'](x[iaccept, :]) * beta
                else:
                    ll = -np.inf
            # ll_out should contain just likelihood values (or NaN for rejected)
            ll_out = np.full_like(pr, np.nan)
            ll_out[iaccept] = ll
            # post_out = prior + likelihood
            post_out = pr.copy()
            post_out[~iaccept] = -np.inf
            post_out[iaccept] = post_out[iaccept] + ll
            return {'prior': pr, 'likelihood': ll_out, 'posterior': post_out}
        else:
            raise ValueError("Parameter must be a vector or a matrix")
    return {'density': posterior}


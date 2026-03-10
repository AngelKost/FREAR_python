import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime, timezone
from scipy.stats import uniform, gaussian_kde

from frear.domain import lon2ix, lat2iy
from frear.MCMC_aux import check_convergence
from frear.tools import rstart2tstart, rstop2tstop, to_timestamp, to_timestamps
from frear.tools import _signif

# -----------------------------------------------------------------------------
# Probabilistic source location
# -----------------------------------------------------------------------------

def chain2probloc(chainburned: np.ndarray, domain: Dict[str, Any]) -> np.ndarray:
    """Creates source location probability grid
    
    Parameters:
        chainburned (np.ndarray): Burned and thinned MCMC chain array over lon and lat
        domain (Dict[str, Any]): Domain dictionary with 'lonmin', 'latmin', 'dx', 'dy', 'nx', 'ny'

    Returns:
        probloc (np.ndarray): 2D array of source location probabilities
    """
    lonlats = np.asarray(chainburned)
    ix = lon2ix(lonlats[:, 0], domain['lonmin'], domain['dx']).astype(int)
    iy = lat2iy(lonlats[:, 1], domain['latmin'], domain['dy']).astype(int)
    ixiys = np.column_stack((ix, iy))
    
    line_indices = ix * domain['ny'] + iy
    probloc = np.bincount(line_indices, minlength=domain['nx'] * domain['ny'])
    probloc = probloc.reshape((domain['nx'], domain['ny'])).astype(float)

    probloc /= float(lonlats.shape[0])

    if not np.isclose(np.sum(probloc), 1.0):
        print(f'Sum of probloc is {np.sum(probloc):1.4f}')
    
    return probloc

# Apply burnin and thinning on each chain, then concatinate into a single chain

def burnThinConcat(zchains: np.ndarray, nburnin = 0, nthin = 1) -> Tuple[np.ndarray]:
    """Apply burn-in and thinning to MCMC chains and concatenate them.

    Parameters:
        zchains (np.ndarray): MCMC chains with shape (niter, npar, nchains)
        nburnin (int): Number of iterations to discard as burn-in
        nthin (int): Thinning interval

    Returns:
        chainsburned (Tuple[np.ndarray]): a tuple containing
            - chainburned (np.ndarray): burned and thinned chains with shape (niter_burned, npar, nchains)
            - chainburned_all (np.ndarray): concatenated burned and thinned chains with shape (niter_burned * nchains, npar)
    """
    chainburned = zchains
    if nburnin > 0:
        chainburned = chainburned[nburnin:, :, :]
    if nthin > 1:
        chainburned = chainburned[::nthin, :, :]

    chainburned_all = [chainburned[:, :, i] for i in range(chainburned.shape[2])]
    return chainburned, np.vstack(chainburned_all)

# Calculate mean, median and 95% uncertainty intervals
# If "times" is provided, then rstart and rstop (fraction between 0 and 1)
# are converted to tstart and tstop (time in UTC)

def chain_summary(chainburned: np.ndarray, chainburned_parnames: List[str], 
                  probs: np.ndarray = np.array([0.025, 0.5, 0.975]), 
                  times: np.ndarray = None) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate summary statistics from MCMC chain.

    Parameters:
        chainburned (np.ndarray): Burned and thinned MCMC chain with shape (niter, npar)
        chainburned_parnames (List[str]): Names of parameters in the chain
        probs (np.ndarray): Probabilities for quantiles to compute
        times (np.ndarray): Array of datetime objects for time conversion

    Returns:
        summary_data (Tuple[pd.DataFrame, List[str]]): a tuple containing
            - chainsum (pd.DataFrame): DataFrame with summary statistics for each parameter
            - prob_labels (List[str]): Labels for the probabilities used in the summary
    """
    parnames = chainburned_parnames
    npar = len(parnames)

    prob_labels = [f'{p:.3f}' for p in probs] + ['mean']
    chainsum = pd.DataFrame(index=prob_labels, columns=parnames, dtype=float)

    for ipar, parname in enumerate(parnames):
        if 'rstart' in parname and times is not None:
            tstart = times[0] + chainburned[:, ipar] * (times[-1] - times[0])
            aux = np.concatenate(([np.quantile(tstart, p) for p in probs], [datetime.fromtimestamp(np.mean([t.timestamp() for t in tstart]), timezone.utc)]))
        elif 'rstop' in parname and times is not None:
            col_rstop = parname
            col_rstart = 'rstart' + col_rstop[5:] # same suffix as rstop
            tstart = times[0] + chainburned[:, parnames.index(col_rstart)] * (times[-1] - times[0])
            tstop = tstart + chainburned[:, ipar] * (times[-1] - tstart)
            aux = np.concatenate(([np.quantile(tstop, p) for p in probs], [datetime.fromtimestamp(np.mean([t.timestamp() for t in tstop]), timezone.utc)]))
        else:
            aux = np.concatenate(([np.quantile(chainburned[:, ipar], p) for p in probs], [np.mean(chainburned[:, ipar])]))
        chainsum[parname] = aux

    return chainsum, prob_labels

def get_posterior_mode(chain: np.ndarray) -> np.ndarray:
    """Calculate the posterior mode from MCMC chain
    
    Parameters:
        chain (np.ndarray): Burned and thinned MCMC chain with shape (niter, npar)
    
    Returns:
        post_mode (np.ndarray): Posterior mode for each parameter
    """
    npar = chain.shape[1]
    post_mode = np.zeros(npar, dtype=float)
    
    colAllEqual = np.abs(chain.max(axis=0) - chain.min(axis=0)) < 1e-18
    post_mode[colAllEqual] = chain[0, colAllEqual]

    #reduce precision if too many unique rows
    digits = 6
    rounded_chain = chain.copy()
    unique_rows = np.unique(rounded_chain, axis=0)
    n_samples = chain.shape[0]

    while len(unique_rows) > 0.2 * n_samples and digits >= 0:
        rounded_chain = _signif(chain, digits)
        unique_rows = np.unique(rounded_chain, axis=0)
        digits -= 1

    unique_rows = np.unique(rounded_chain, axis=0)
    occurrences = np.array([np.sum(np.all(rounded_chain == unique_row, axis=1)) for unique_row in unique_rows])

    if sum(occurrences == max(occurrences)) > 1:
        print('There is more than one posterior mode; printing all, using the first one')
        print(np.where(occurrences == max(occurrences))[0])
    
    post_mode[~colAllEqual] = unique_rows[np.argmax(occurrences), :][~colAllEqual]
    return post_mode

# -----------------------------------------------------------------------------
# Selection of MCMC analysis
# -----------------------------------------------------------------------------

def analyse_mcmc(out: Dict[str, Any], settings: Dict[str, Any], 
                 probs: np.ndarray = np.array([0.025, 0.5, 0.975]), 
                 get_post_mode: bool = False) -> Dict[str, Any]:
    """Analyze MCMC output to compute summary statistics and diagnostics.

    Parameters:
        out (Dict[str, Any]): MCMC output dictionary
            - chains (np.ndarray): MCMC chains with shape (niter, npar, nchains)
        settings (Dict[str, Any]): Settings dictionary with MCMC configuration
            - parnames (List[str]): Names of parameters
            - nburnin (int): Number of burn-in iterations
            - nthin (int): Thinning interval
            - nchains (int): Number of MCMC chains
            - times (np.ndarray): Array of datetime objects for time conversion
            - domain (Dict[str, Any]): Domain dictionary for source location probability
        probs (np.ndarray): Probabilities for quantiles to compute.
        get_post_mode (bool): Whether to compute the posterior mode.

    Returns:
        data (Dict[str, Any]): Dictionary containing analysis results
            - chainburned (np.ndarray): Burned and thinned MCMC chains
            - zchainsummary (pd.DataFrame): Summary statistics of the chains
            - Rs (np.ndarray): Rubin-Gelman convergence diagnostics
            - probloc (np.ndarray): Probabilistic source location grid
            - post_median (np.ndarray): Posterior median for each parameter
            - post_mode (np.ndarray): Posterior mode for each parameter
    """
    npar = len(settings['parnames'])
    zchains = out['chains']
    chainburned, chainburned_all = burnThinConcat(zchains = zchains[:, 0:npar, :], nburnin = round(settings['nburnin'] / settings['nchains']),
                                                  nthin = settings['nthin']) # omitting prior, ll, post #FIXME: check if correct
    zchainsummary, prob_labels = chain_summary(chainburned_all, settings['parnames'], probs = probs, times = settings.get('times'))

    # Calculate the posterior median (for each variable separately)
    post_median = np.median(chainburned_all, axis=0)

    # Calculate the posterior mode (for all variables at once)
    if get_post_mode:
        post_mode = get_posterior_mode(chain = chainburned_all)
    else:
        post_mode = np.full(npar, np.nan)

    # Source location probability
    if 'lon' in settings['parnames'] and 'lat' in settings['parnames']:
        lon_idx = settings['parnames'].index('lon')
        lat_idx = settings['parnames'].index('lat')
        probloc = chain2probloc(chainburned = chainburned_all[:, [lon_idx, lat_idx]], domain = settings['domain'])
    else:
        probloc = None

    # Rubin-Gelman diagnostic
    Rs = check_convergence(zchains, ipars = list(range(npar)) + [npar + 2]) # posterior is at npar + 2 index

    return {"chainburned": chainburned_all,
            "zchainsummary": zchainsummary,
            "Rs": Rs,
            "probloc": probloc,
            "post_median": post_median,
            "post_mode": post_mode}

# -----------------------------------------------------------------------------
# Traceplot of MCMC
# -----------------------------------------------------------------------------

def plot_trace(zchains: np.ndarray, ipars: Optional[List[int]] = None, 
               parnames: Optional[List[str]] = None,
               outpath: Optional[str] = None, show: bool = True):
    """
    Plot MCMC trace for selected parameters

    Parameters:
        zchains (np.ndarray): MCMC chains with shape (niter, npar, nchains)
        ipars (Optional[List[int]]): Indices of parameters to plot
        parnames (Optional[List[str]]): Names of parameters
        outpath (Optional[str]): Path to save the plot
        show (bool): Whether to display the plot
    """
    niter, npar, nchains = zchains.shape
    if ipars is None:
        ipars = list(range(npar))
    if parnames is None:
        parnames = [f"Param {i+1}" for i in ipars]

    nplots = len(ipars)
    if nplots < 8:
        ncol = 1
    elif nplots < 12:
        ncol = 2
    elif nplots < 16:
        ncol = 3
    else:
        ncol = 4
    nrow = int(np.ceil(nplots // ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2 * nrow), squeeze=False)
    axes = axes.flatten()

    chain_colors = ["#F8766D", "#B79F00", "#00BA38", "#00BFC4", "#619CFF", "#F564E3"]

    for i, ipar in enumerate(ipars):
        ax = axes[i]
        ylim = [zchains[:, ipar, :].min(), zchains[:, ipar, :].max()]
        for ichain in range(nchains):
            color = chain_colors[ichain % len(chain_colors)]
            ax.plot(range(niter), zchains[:, ipar, ichain], color=color)
        ax.set_title(parnames[i], fontsize=10)
        ax.set_xlim(0, niter-1)
        ax.set_ylim(ylim)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.grid(True, color='lightgray', linestyle='--', alpha=0.5)

    for j in range(nplots, nrow*ncol):
        axes[j].axis('off')

    plt.tight_layout()
    if outpath is not None:
        plt.savefig(f"{outpath}/mcmc_trace.pdf")
    if show:
        plt.show()

# -----------------------------------------------------------------------------
# Monovariate marginal posterior plot
# - density instead of frequencies
# - added prior pdf
# lpriorrange: does the x-axis span the full prior parameter range? Else zoom where pdf > 0
# -----------------------------------------------------------------------------

def aux_marginal_post(chainburned: np.ndarray, times: np.ndarray, true_values: np.ndarray, 
                      post_median: np.ndarray, post_mode: np.ndarray,
                      lower_bayes: np.ndarray, upper_bayes: np.ndarray,
                      settings: Dict[str, Any],
                      parnames: Optional[List[str]] = None, nbreaks = 50) -> Dict[str, Any]:
    """
    Process MCMC chain for marginal posterior plots

    Parameters:
        chainburned (np.ndarray): Burned and thinned MCMC chain with shape (npoints, npar)
        times (np.ndarray): Array of times corresponding to the chain
        true_values (np.ndarray): True values of the parameters
        post_median (np.ndarray): Posterior median values of the parameters
        post_mode (np.ndarray): Posterior mode values of the parameters
        lower_bayes (np.ndarray): Lower bounds of the parameters
        upper_bayes (np.ndarray): Upper bounds of the parameters
        settings (Dict[str, Any]): Settings dictionary
            - domain (Dict[str, Any]): Domain information
        parnames (Optional[List[str]]): Names of the parameters
        nbreaks (int): Number of bins for evaluation

    Returns:
        data (Dict[str, Any]): Dictionary containing processed data for marginal posterior plots
            - chaindf (pd.DataFrame): DataFrame of the chain with parameter names
            - true_values (np.ndarray): True values of the parameters
            - post_median (np.ndarray): Posterior median values of the parameters
            - post_mode (np.ndarray): Posterior mode values of the parameters
            - breaks (List[np.ndarray]): List of arrays defining the breaks for each parameter
            - zprior (np.ndarray): Prior probability density values for each parameter
    """

    true_values = true_values.copy()
    true_values = [tv if tv is not None else np.nan for tv in true_values]
    post_median = post_median.copy()
    post_mode = post_mode.copy()

    npoints, npar = chainburned.shape
    if parnames is None:
        parnames = [f'par{i+1}' for i in range(npar)]

    chaindf = pd.DataFrame(chainburned, columns=parnames)
    times_numeric = np.array([t.timestamp() if isinstance(t, datetime) else t for t in times])

    rstart_cols = [i for i, name in enumerate(parnames) if "rstart" in name]
    for i_rstart in rstart_cols:
        s_rstart = parnames[i_rstart]
        s_tstart = s_rstart.replace("rstart", "tstart")
        chaindf[s_tstart] = times_numeric[0] + chaindf[s_rstart] * (times_numeric[-1] - times_numeric[0])

        s_rstop = s_rstart.replace("start", "stop")
        s_rstop = next((c for c in chaindf.columns if c.startswith(s_rstop)), None)
        if s_rstop:
            s_tstop = s_rstop.replace("rstop", "tstop")
            chaindf[s_tstop] = chaindf[s_tstart] + chaindf[s_rstop] * (times_numeric[-1] - chaindf[s_tstart])
        chaindf[s_tstart] = [datetime.fromtimestamp(x, timezone.utc) for x in chaindf[s_tstart].values]
        if s_rstop:
            chaindf[s_tstop] = [datetime.fromtimestamp(x, timezone.utc) for x in chaindf[s_tstop].values]

    chaindf = chaindf.drop(columns=[c for c in chaindf.columns if 'rstart' in c or 'rstop' in c], errors='ignore')

    breaks = [np.linspace(lower_bayes[i], upper_bayes[i], nbreaks) for i in range(npar)]

    for ipar, name in enumerate(parnames):
        if "rstart" in name:
            breaks[ipar] = np.linspace(times_numeric[0], times_numeric[-1], nbreaks)
            true_values[ipar] = times_numeric[0] + true_values[ipar] * (times_numeric[-1] - times_numeric[0])
            post_median[ipar] = times_numeric[0] + post_median[ipar] * (times_numeric[-1] - times_numeric[0])
            post_mode[ipar] = times_numeric[0] + post_mode[ipar] * (times_numeric[-1] - times_numeric[0])
        if "rstop" in name:
            breaks[ipar] = np.linspace(times_numeric[0], times_numeric[-1], nbreaks)
            rstart_idx = None if name.replace("rstop", "rstart") not in parnames else parnames.index(name.replace("rstop", "rstart"))
            if rstart_idx is not None:
                true_values[ipar] = true_values[rstart_idx] + true_values[ipar] * (times_numeric[-1] - true_values[rstart_idx])
                post_median[ipar] = post_median[rstart_idx] + post_median[ipar] * (times_numeric[-1] - post_median[rstart_idx])
                post_mode[ipar] = post_mode[rstart_idx] + post_mode[ipar] * (times_numeric[-1] - post_mode[rstart_idx])

    if "lon" in chaindf.columns:
        ilon = chaindf.columns.get_loc("lon")
        if lower_bayes[ilon] == upper_bayes[ilon]:
            breaks[ilon] = np.linspace(settings["domain"]["lonmin"], settings["domain"]["lonmax"], nbreaks)
    if "lat" in chaindf.columns:
        ilat = chaindf.columns.get_loc("lat")
        if lower_bayes[ilat] == upper_bayes[ilat]:
            breaks[ilat] = np.linspace(settings["domain"]["latmin"], settings["domain"]["latmax"], nbreaks)

    zprior = np.full((nbreaks, npar), -999.0)
    for ipar in range(npar):
        if lower_bayes[ipar] != upper_bayes[ipar]:
            pname = parnames[ipar]
            if "rstart" in pname:
                zprior[:, ipar] = uniform.pdf(breaks[ipar], loc=times_numeric[0], scale=times_numeric[-1]-times_numeric[0])
            elif "rstop" in pname:
                zprior[:, ipar] = 0.0
                for ib in range(nbreaks):
                    zprior[:, ipar] += np.array([uniform.pdf(x, loc=ib, scale=nbreaks-ib) for x in np.arange(1, nbreaks+1)])  # dunif(1:nbreaks, min=ibreak, max=nbreaks)
                zprior[-1, ipar] += 1.0
                zprior[:, ipar] /= (times_numeric[-1] - times_numeric[0])
            else:
                zprior[:, ipar] = uniform.pdf(breaks[ipar], loc=lower_bayes[ipar], scale=upper_bayes[ipar]-lower_bayes[ipar])


    return {
        "chaindf": chaindf,
        "true_values": true_values,
        "post_median": post_median,
        "post_mode": post_mode,
        "breaks": breaks,
        "zprior": zprior
    }


def monovar_marginal_post(chainburned: np.ndarray,
                          times: np.ndarray,
                          lower_bayes: np.ndarray,
                          upper_bayes: np.ndarray,
                          settings: Dict[str, Any],
                          true_values: Optional[np.ndarray] = None,
                          post_median: Optional[np.ndarray] = None,
                          post_mode: Optional[np.ndarray] = None,
                          parnames: Optional[List[str]] = None,
                          nbreaks: int = 50,
                          lpriorrange: bool = True,
                          outpath: Optional[str] = None,
                          show: bool = True) -> None:
    """
    Plot marginal posterior for each parameter

    Parameters:
        chainburned (np.ndarray): Burned and thinned MCMC chain with shape (npoints, npar)
        times (np.ndarray): Array of times corresponding to the chain
        lower_bayes (np.ndarray): Lower bounds of the parameters
        upper_bayes (np.ndarray): Upper bounds of the parameters
        settings (Dict[str, Any]): Settings dictionary
            - domain (Dict[str, Any]): Domain information
        true_values (Optional[np.ndarray]): True values of the parameters
        post_median (Optional[np.ndarray]): Posterior median values of the parameters
        post_mode (Optional[np.ndarray]): Posterior mode values of the parameters
        parnames (Optional[List[str]]): Names of the parameters
        nbreaks (int): Number of bins for evaluation
        lpriorrange (bool): Whether to set x-axis limits to prior range
        outpath (Optional[str]): Path to save the plot
        show (bool): Whether to display the plot
    """
    aux_output = aux_marginal_post(
        chainburned=chainburned,
        times=times,
        true_values=true_values if true_values is not None else np.full(5, np.nan),
        post_median=post_median if post_median is not None else np.full(5, np.nan),
        post_mode=post_mode if post_mode is not None else np.full(5, np.nan),
        lower_bayes=lower_bayes,
        upper_bayes=upper_bayes,
        settings=settings,
        parnames=parnames,
        nbreaks=nbreaks
    )

    chaindf = aux_output["chaindf"]
    true_values = aux_output["true_values"]
    post_median = aux_output["post_median"]
    post_mode = aux_output["post_mode"]
    breaks = aux_output["breaks"]
    zprior = aux_output["zprior"]

    npar = chaindf.shape[1]
    if npar < 8:
        nrows, ncols = 1, npar
    elif npar < 12:
        nrows = 2
        ncols = int(np.ceil(npar/2))
    elif npar < 16:
        nrows = 3
        ncols = int(np.ceil(npar/3))
    else:
        nrows = 4
        ncols = int(np.ceil(npar/4))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    if npar == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    lower_bayes = lower_bayes.copy()
    upper_bayes = upper_bayes.copy()

    for ipar, parname in enumerate(parnames):
        if "rstop" in parname:
            lower_rstart = lower_bayes[parnames.index(parname.replace("rstop", "rstart"))]
            upper_rstart = upper_bayes[parnames.index(parname.replace("rstop", "rstart"))]
            lower_bayes[ipar] = rstop2tstop(times, rstart=lower_rstart, rstop=lower_bayes[ipar]).timestamp()
            upper_bayes[ipar] = rstop2tstop(times, rstart=upper_rstart, rstop=upper_bayes[ipar]).timestamp()
    for ipar, parname in enumerate(parnames):
        if "rstart" in parname:
            lower_bayes[ipar] = rstart2tstart(times, lower_bayes[ipar]).timestamp()
            upper_bayes[ipar] = rstart2tstart(times, upper_bayes[ipar]).timestamp()

    for ipar in range(npar):
        ax = axes[ipar]
        data = chaindf[parnames[ipar].replace("rstart", "tstart").replace("rstop", "tstop")].values
        is_datetime_param = "start" in parnames[ipar] or "stop" in parnames[ipar]

        # Convert datetime-like objects to numeric (timestamp) values for KDE evaluation
        data = to_timestamps(data)

        try:
            kde = gaussian_kde(data)
            x_eval = np.linspace(data.min(), data.max(), 200)
            
            # If it's a datetime parameter, convert to datetime for matplotlib
            if is_datetime_param:
                x_eval_plot = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in x_eval]
            else:
                x_eval_plot = x_eval
                
            y_eval = kde(x_eval)
            ax.fill_between(x_eval_plot, y_eval, color="#22A884FF", alpha=0.6, label="Posterior")
            ax.plot(x_eval_plot, y_eval, color="slateblue", lw=1)
        except Exception as e:
            print(f"Error in KDE plot for parameter {parnames[ipar]}: {e}")

        prior_x = np.linspace(lower_bayes[ipar], upper_bayes[ipar], nbreaks)
        if is_datetime_param:
            prior_x_plot = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in prior_x]
        else:
            prior_x_plot = prior_x
        prior_y = np.ones_like(prior_x) / (upper_bayes[ipar] - lower_bayes[ipar])
        ax.plot(prior_x_plot, prior_y, color="green", lw=1, label="Prior")

        if isinstance(post_median[ipar], datetime) or not np.isnan(post_median[ipar]):
            if is_datetime_param:
                if not isinstance(post_median[ipar], datetime):
                    post_median_plot = datetime.fromtimestamp(post_median[ipar], tz=timezone.utc)
                else:
                    post_median_plot = post_median[ipar]
            else:
                post_median_plot = post_median[ipar]
            ax.axvline(post_median_plot, color="blue", ls='--', lw=1)
        if isinstance(post_mode[ipar], datetime) or not np.isnan(post_mode[ipar]):
            if is_datetime_param:
                if not isinstance(post_mode[ipar], datetime):
                    post_mode_plot = datetime.fromtimestamp(post_mode[ipar], tz=timezone.utc)
                else:
                    post_mode_plot = post_mode[ipar]
            else:
                post_mode_plot = post_mode[ipar]
            ax.axvline(post_mode_plot, color="green", ls='--', lw=1)
        if isinstance(true_values[ipar], datetime) or not np.isnan(true_values[ipar]):
            if is_datetime_param:
                if not isinstance(true_values[ipar], datetime):
                    true_val_plot = datetime.fromtimestamp(true_values[ipar], tz=timezone.utc)
                else:
                    true_val_plot = true_values[ipar]
            else:
                true_val_plot = true_values[ipar]
            ax.axvline(true_val_plot, color="red", lw=1)

        if lpriorrange:
            if "rstart" in parnames[ipar] or "rstop" in parnames[ipar]:
                lb = lower_bayes[ipar]
                lb = to_timestamp(lb)

                ub = upper_bayes[ipar]
                ub = to_timestamp(ub)

                if is_datetime_param:
                    lb_datetime = datetime.fromtimestamp(lb, tz=timezone.utc)
                    ub_datetime = datetime.fromtimestamp(ub, tz=timezone.utc)
                    ax.set_xlim(lb_datetime, ub_datetime)
                else:
                    ax.set_xlim(lb, ub)
            else: 
                ax.set_xlim(lower_bayes[ipar], upper_bayes[ipar])
        else:
            # Zoomed posterior plot: set x-axis limits to min and max of (data + true value)
            zoom_lb = float(np.min(data))
            zoom_ub = float(np.max(data))
    
            true_val_num = None if pd.isna(true_values[ipar]) else to_timestamp(true_values[ipar])

            if true_val_num is not None:
                zoom_lb = min(zoom_lb, true_val_num)
                zoom_ub = max(zoom_ub, true_val_num)

            if zoom_lb == zoom_ub:
                pad = max(abs(zoom_lb) * 0.01, 1e-6)
                zoom_lb -= pad
                zoom_ub += pad

            if is_datetime_param:
                ax.set_xlim(
                    datetime.fromtimestamp(zoom_lb, tz=timezone.utc),
                    datetime.fromtimestamp(zoom_ub, tz=timezone.utc)
                )
            else:
                ax.set_xlim(zoom_lb, zoom_ub)
        ax.set_title(parnames[ipar] if "start" not in parnames[ipar] and "stop" not in parnames[ipar] else parnames[ipar].replace("rstart", "tstart").replace("rstop", "tstop"))
        ax.set_ylabel("Density")
        ax.grid(True, ls='--', alpha=0.3)
        
        # Format x-axis as readable date+time for tstart and tstop parameters
        if is_datetime_param:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    for j in range(npar, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()

    if outpath is not None:
        if not lpriorrange:
            fig.savefig(f"{outpath}/marginal_posteriors_zoomed.pdf")
        else:
            fig.savefig(f"{outpath}/marginal_posteriors.pdf")

    if show:
        plt.show()
    else:
        plt.close(fig)

# -----------------------------------------------------------------------------
# Bivariate marginal posterior plot
# differences:
# - density instead of frequencies
# - added prior pdf
# -----------------------------------------------------------------------------

def bivariate_marginal_post(chainburned: np.ndarray,
                            times: np.ndarray,
                            lower_bayes: np.ndarray,
                            upper_bayes: np.ndarray,
                            settings: Optional[Dict[str, Any]] = None,
                            true_values: Optional[np.ndarray] = None,
                            post_median: Optional[np.ndarray] = None,
                            post_mode: Optional[np.ndarray] = None,
                            parnames: Optional[List[str]] = None,
                            outpath: Optional[str] = None,
                            show: bool = True) -> None:
    """Plot bivariate marginal posterior for each parameter pair
    
    Parameters:
        chainburned (np.ndarray): Burned and thinned MCMC chain with shape (npoints, npar)
        times (np.ndarray): Array of times corresponding to the chain
        lower_bayes (np.ndarray): Lower bounds of the parameters
        upper_bayes (np.ndarray): Upper bounds of the parameters
        settings (Optional[Dict[str, Any]]): Settings dictionary
        true_values (Optional[np.ndarray]): True values of the parameters
        post_median (Optional[np.ndarray]): Posterior median values of the parameters
        post_mode (Optional[np.ndarray]): Posterior mode values of the parameters
        parnames (Optional[List[str]]): Names of the parameters
        outpath (Optional[str]): Path to save the plot
        show (bool): Whether to display the plot
    """
    npoints, npar = chainburned.shape
    if parnames is None:
        parnames = [f'par{i+1}' for i in range(npar)]
    
    aux_output = aux_marginal_post(
        chainburned=chainburned,
        times=times,
        true_values=true_values if true_values is not None else np.full(5, np.nan),
        post_median=post_median if post_median is not None else np.full(5, np.nan),
        post_mode=post_mode if post_mode is not None else np.full(5, np.nan),
        lower_bayes=lower_bayes if lower_bayes is not None else np.full(npar, np.nan),
        upper_bayes=upper_bayes if upper_bayes is not None else np.full(npar, np.nan),
        settings=settings if settings is not None else {},
        parnames=parnames,
        nbreaks=50
    )

    chaindf = aux_output["chaindf"]
    true_values = aux_output["true_values"]
    post_median = aux_output["post_median"]
    post_mode = aux_output["post_mode"]
    breaks = aux_output["breaks"]
    zprior = aux_output["zprior"]

    nplots = (npar**2 + npar) // 2
    nplot = 1

    fig, axes = plt.subplots(npar, npar, figsize=(4*npar, 4*npar))
    axes = np.array(axes)

    for i in range(npar):
        for j in range(npar):
            ax = axes[i, j]
            
            # Diagonal
            if i == j:
                data = chaindf.iloc[:, i].values
                is_datetime_param = "start" in parnames[i] or "stop" in parnames[i]

                # Convert datetime-like objects to numeric (timestamp) values for KDE evaluation
                data = to_timestamps(data)

                try:
                    kde = gaussian_kde(data)
                    x_eval = np.linspace(data.min(), data.max(), 200)
            
                    # If it's a datetime parameter, convert to datetime for matplotlib
                    if is_datetime_param:
                        x_eval_plot = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in x_eval]
                    else:
                        x_eval_plot = x_eval
                
                    y_eval = kde(x_eval)
                    ax.fill_between(x_eval_plot, y_eval, color="#22A884FF", alpha=0.6, label="Posterior")
                    ax.plot(x_eval_plot, y_eval, color="slateblue", lw=1)
                except Exception as e:
                    print(f"Error in KDE plot for parameter {parnames[i]}: {e}")

                if isinstance(post_median[i], datetime) or not np.isnan(post_median[i]):
                    if is_datetime_param:
                        if not isinstance(post_median[i], datetime):
                            post_median_plot = datetime.fromtimestamp(post_median[i], tz=timezone.utc)
                        else:
                            post_median_plot = post_median[i]
                    else:
                        post_median_plot = post_median[i]
                    ax.axvline(post_median_plot, color="blue", ls='--', lw=1)
                if isinstance(post_mode[i], datetime) or not np.isnan(post_mode[i]):
                    if is_datetime_param:
                        if not isinstance(post_mode[i], datetime):
                            post_mode_plot = datetime.fromtimestamp(post_mode[i], tz=timezone.utc)
                        else:
                            post_mode_plot = post_mode[i]
                    else:
                        post_mode_plot = post_mode[i]
                    ax.axvline(post_mode_plot, color="green", ls='--', lw=1)
                if isinstance(true_values[i], datetime) or not np.isnan(true_values[i]):
                    if is_datetime_param:
                        if not isinstance(true_values[i], datetime):
                            true_val_plot = datetime.fromtimestamp(true_values[i], tz=timezone.utc)
                        else:
                            true_val_plot = true_values[i]
                    else:
                        true_val_plot = true_values[i]
                    ax.axvline(true_val_plot, color="red", lw=1)
                
                ax.set_xlabel(parnames[i])
                ax.set_ylabel("Density")
                ax.grid(True, ls='--', alpha=0.3)

                # Format x-axis as readable date+time for tstart and tstop parameters
                if is_datetime_param:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            elif j > i:
                # Skip bivariate KDE if either parameter is fixed (lower_bayes == upper_bayes)
                if lower_bayes[i] == upper_bayes[i] or lower_bayes[j] == upper_bayes[j]:
                    ax.axis('off')
                    continue
                
                xdata = chaindf.iloc[:, i].values
                ydata = chaindf.iloc[:, j].values

                is_x_datetime_param = "start" in parnames[i] or "stop" in parnames[i]
                is_y_datetime_param = "start" in parnames[j] or "stop" in parnames[j]

                xdata = to_timestamps(xdata)
                ydata = to_timestamps(ydata)
                
                nx = ny = 50
                x_min, x_max = xdata.min(), xdata.max()
                y_min, y_max = ydata.min(), ydata.max()
                xi = np.linspace(x_min, x_max, nx)
                yi = np.linspace(y_min, y_max, ny)
                xi_grid, yi_grid = np.meshgrid(xi, yi)
                zi = np.zeros_like(xi_grid)

                kde2 = gaussian_kde(np.vstack([xdata, ydata]))
                zi = kde2(np.vstack([xi_grid.ravel(), yi_grid.ravel()])).reshape(xi_grid.shape)

                if is_x_datetime_param:
                    xi_grid_plot = np.array([[datetime.fromtimestamp(ts, tz=timezone.utc) for ts in row] for row in xi_grid])
                else:
                    xi_grid_plot = xi_grid
                if is_y_datetime_param:
                    yi_grid_plot = np.array([[datetime.fromtimestamp(ts, tz=timezone.utc) for ts in row] for row in yi_grid])
                else:
                    yi_grid_plot = yi_grid

                im = ax.contourf(xi_grid_plot, yi_grid_plot, zi, levels=20, cmap='viridis')
                fig.colorbar(im, ax=ax, label="Density")
                
                if not (np.isnan(true_values[i]) or np.isnan(true_values[j])):
                    if is_x_datetime_param:
                        if not isinstance(true_values[i], datetime):
                            true_val_x_plot = datetime.fromtimestamp(true_values[i], tz=timezone.utc)
                        else:
                            true_val_x_plot = true_values[i]
                    else:
                        true_val_x_plot = true_values[i]
                if isinstance(true_values[j], datetime) or not np.isnan(true_values[j]):
                    if is_y_datetime_param:
                        if not isinstance(true_values[j], datetime):
                            true_val_y_plot = datetime.fromtimestamp(true_values[j], tz=timezone.utc)
                        else:
                            true_val_y_plot = true_values[j]
                    else:
                        true_val_y_plot = true_values[j]
                    ax.scatter(true_val_x_plot, true_val_y_plot, color='red', s=50)
                
                if not (np.isnan(post_median[i]) or np.isnan(post_median[j])):
                    if is_x_datetime_param:
                        if not isinstance(post_median[i], datetime):
                            post_median_x_plot = datetime.fromtimestamp(post_median[i], tz=timezone.utc)
                        else:
                            post_median_x_plot = post_median[i]
                    else:
                        post_median_x_plot = post_median[i]
                    if is_y_datetime_param:
                        if not isinstance(post_median[j], datetime):
                            post_median_y_plot = datetime.fromtimestamp(post_median[j], tz=timezone.utc)
                        else:
                            post_median_y_plot = post_median[j]                    
                    else:
                        post_median_y_plot = post_median[j]
                    ax.scatter(post_median_x_plot, post_median_y_plot, color='blue', s=50)
                
                if not (np.isnan(post_mode[i]) or np.isnan(post_mode[j])):
                    if is_x_datetime_param:
                        if not isinstance(post_mode[i], datetime):
                            post_mode_x_plot = datetime.fromtimestamp(post_mode[i], tz=timezone.utc)
                        else:
                            post_mode_x_plot = post_mode[i]
                    else:
                        post_mode_x_plot = post_mode[i]
                    if is_y_datetime_param:
                        if not isinstance(post_mode[j], datetime):
                            post_mode_y_plot = datetime.fromtimestamp(post_mode[j], tz=timezone.utc)
                        else:
                            post_mode_y_plot = post_mode[j]
                    else:
                        post_mode_y_plot = post_mode[j]
                    ax.scatter(post_mode_x_plot, post_mode_y_plot, color='green', s=50)
                ax.set_xlabel(parnames[i])
                ax.set_ylabel(parnames[j])
                ax.grid(True, ls='--', alpha=0.3)
                # Format x-axis as readable date+time for tstart and tstop parameters
                if is_x_datetime_param:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                # Lower triangle: empty
                ax.axis('off')

    fig.tight_layout()
    if outpath is not None:
        fig.savefig(f"{outpath}/bivariate_marginal_posteriors.pdf")
    if show:
        plt.show()
    else:
        plt.close(fig)
    

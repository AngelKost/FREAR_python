import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Any, Optional, Tuple
from matplotlib.colors import LinearSegmentedColormap

from frear.plot_geofield import plot_2D_kde, plot_2D_direct
from frear.plot_timeseries import plot_ac_timeseries
from frear.domain import lon2ix, lat2iy, ix2lon, iy2lat
from frear.tools import _signif

def plot_cost(cost: np.ndarray, domain: Dict[str, Any], 
              IMSfile: Optional[str] = None, reactorfile: Optional[str] = None, 
              title: str = "",
              outpath: Optional[str] = None, show: bool = True) -> None:
    """Plot cost over domain
    
    Parameters:
        cost (np.ndarray): 2D numpy array with dimensions [nx, ny] containing cost values.
        domain (Dict[str, Any]): Dictionary containing domain information.
        IMSfile (Optional[str]): Path to IMS file.
        reactorfile (Optional[str]): Path to reactor file.
        title (str): Title for the plot.
        outpath (Optional[str]): Output path for saving the plot.
        show (bool): Whether to display the plot.
    """
    legend_colors = ["red", "yellow", "royalblue", "white"]

    cmin = np.nanmin(cost)
    cmax = np.nanmax(cost)
    levels = None
    if cmin < 2:
        levels = np.arange(1, 2.0001, 0.1)
        if cmax < levels.max():
            levels = np.arange(1, 1.8001, 0.08)
        if cmax < levels.max():
            levels = np.arange(1, 1.6001, 0.06)
        if cmax < levels.max():
            levels = np.arange(1, 1.5001, 0.05)
        if cmax < levels.max():
            levels = np.arange(1, 1.4001, 0.04)
        if cmax < levels.max():
            levels = np.arange(1, 1.3001, 0.03)
        if cmax < levels.max():
            levels = np.arange(1, 1.2001, 0.02)
        if cmax > levels.max():
            levels = np.concatenate([levels, [np.round(cmax, 1)]])
        levels = np.unique(levels)

    else:
        q = np.quantile(
            cost,
            [0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
        )
        levels = np.round(q, 1)

    plot_2D_direct(
        data=cost.T,
        domain=domain,
        title=title,
        colors=legend_colors,
        IMSfile=IMSfile,
        reactorfile=reactorfile,
        levels=levels,
        labels=["Longitude", "Latitude", "Cost"],
        outpath=outpath,
        filename="cost_plot.pdf",
        show=show
    )

def plot_accQ(
    accQ: np.ndarray,
    domain: Dict[str, Any],
    IMSfile: Optional[str] = None,
    reactorfile: Optional[str] = None,
    title: str = "",
    outpath: Optional[str] = None, show: bool = True) -> None:
    """Plot accumulated Q over domain
    
    Parameters:
        accQ (np.ndarray): 2D numpy array with accumulated Q values.
        domain (Dict[str, Any]): Dictionary containing domain information.
        IMSfile (Optional[str]): Path to IMS file.
        reactorfile (Optional[str]): Path to reactor file.
        title (str): Title for the plot.
        outpath (Optional[str]): Output path for saving the plot.
        show (bool): Whether to display the plot.
    """
    colors = ["white", "darkgreen", "yellow", "royalblue"]

    nonzero = accQ[accQ != 0]
    if len(nonzero) == 0:
        raise ValueError("accQ contains no non-zero values.")

    levels = np.concatenate((
        [0],
        _signif(10 ** np.linspace(np.log10(nonzero.min()),
                                   np.log10(accQ.max()), 10), 2)
    ))

    plot_2D_direct(
        data=accQ.T,
        domain=domain,
        title=title,
        colors=colors,
        IMSfile=IMSfile,
        reactorfile=reactorfile,
        levels=levels,
        labels=["Longitude", "Latitude", "Accumulated Q [Bq]"],
        outpath=outpath,
        filename="accQ_plot.pdf",
        show=show
    )

def plot_release_term(times: np.ndarray, Qs: np.ndarray, title: str = '', 
                      log: bool = False, ylim: Optional[Tuple[float, float]] = None,
                      outpath: Optional[str] = None, show: bool = True,
                      ax: Optional[plt.Axes] = None) -> None:
    """
    Plot release term over time using step function

    Parameters:
        times (np.ndarray): Array of datetime objects representing time intervals
        Qs (np.ndarray): Array of release values corresponding to time intervals (length(times) - 1) x nisotopes
        title (str): Title for the plot
        log (bool): Whether to use logarithmic scale for y-axis
        ylim (Optional[Tuple[float, float]]): Y-axis limits as (ymin, ymax)
        outpath (Optional[str]): Output path for saving the plot
        show (bool): Whether to display the plot
    """
    times = np.array(times)
    ntimes = len(times) - 1  # Number of release periods (intervals between times)
    nisotopes = len(Qs) // ntimes if ntimes > 0 else 1

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = None
    colors = plt.cm.tab10.colors

    for i in range(nisotopes):
        Qs_i = Qs[i*ntimes:(i+1)*ntimes]
        # Plot as bar chart showing release in each time interval
        # times has n+1 elements (interval boundaries): [t0, t1, t2, ..., tn]
        # Qs_i has n elements (releases in each interval)
        # Center bars between the time boundaries
        x_positions = np.arange(ntimes) + 0.5
        ax.bar(x_positions, Qs_i, label=f'Isotope {i+1}', color=colors[i % len(colors)], alpha=0.7, width=0.8)
    
    # Set ticks at time boundaries
    if len(times) > 1:
        ax.set_xticks(np.arange(len(times)))
        ax.set_xticklabels([t.strftime('%m-%d') for t in times])

    ax.set_xlabel('Time [UTC]')
    ax.set_ylabel('Release [Bq]')
    ax.set_title(title)
    if log:
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.5, axis='y')

    if fig is not None:
        if outpath is not None:
            fig.savefig(f"{outpath}/release_term.pdf", bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

def plot_AC(
    Qs: np.ndarray,
    ix: int,
    iy: int,
    mod_error_cost: np.ndarray,
    settings: Dict[str, Any],
    data: Dict[str, Any],
    outpath: Optional[str] = None,
    show: bool = True
):
    """Plot AC time series at grid location (ix, iy) for cost-based source model
    
    Parameters:
        Qs (np.ndarray): optimized source parameters
        ix (int): grid index in x (longitude) direction
        iy (int): grid index in y (latitude) direction
        mod_error_cost (np.ndarray): model error for cost function
        settings (Dict[str, Any]): settings dictionary
            - sourcemodelcost_exec (Callable): function to compute modeled concentrations
            - obsunitfact (float): observation unit factor
            - obsscalingfact (float): observation scaling factor
        data (Dict[str, Any]): additional data dictionary
            - srs (np.ndarray): numpy array of shape [ntimes, nx, ny, nsamples] of SRS values
            - Qfact (float): scaling factor for source model
            - samples (pd.DataFrame): dataframe with sample information
            - obs (np.ndarray): observed concentrations
            - obs_error (np.ndarray): observation errors
            - MDC (np.ndarray): minimum detectable concentration
        outpath (Optional[str]): output path for saving plots
        show (bool): whether to display the plot
    """

    title = "Modelled concentrations using cost function"
    srs2AC_cost = settings["sourcemodelcost_exec"]
    srs = data['srs']

    par = Qs[:, ix, iy]
    M = srs[:, ix, iy, :]
    Qfact = data['Qfact']

    mod = srs2AC_cost(par=par, Qfact=Qfact, M=M)

    mod_min = mod - mod_error_cost
    mod_max = mod + mod_error_cost

    labels = data["samples"]["Entity"]
    obs = data['obs']
    obs_error = data['obs_error']
    obs_min = obs - obs_error
    obs_max = obs + obs_error
    MDC = data['MDC']

    plot_ac_timeseries(
        ac=None,
        obs=obs,
        obs_min=obs_min,
        obs_max=obs_max,
        mod=mod,
        mod_min=mod_min,
        mod_max=mod_max,
        MDC=MDC,
        obsunitfact=settings["obsunitfact"],
        obsscalingfact=settings["obsscalingfact"],
        title=title,
        labels=labels,
        prefix="cost_",
        outpath=outpath,
        show=show
    )

def plot_AC_expertPanel(
    Qs: np.ndarray, mod_error_cost: np.ndarray, cost: np.ndarray,
    settings: Dict[str, Any], data: Dict[str, Any],
    lon: Optional[float] = None, lat: Optional[float] = None,
    outpath: Optional[str] = None, show: bool = True
):
    """Plot AC expert panel at specified lon/lat or at minimum cost location (release term, AC time series, relative contribution)
    
    Parameters:
        Qs (np.ndarray): optimized source parameters
        mod_error_cost (np.ndarray): model error for cost function
        cost (np.ndarray): 2D numpy array with cost values over the domain
        settings (Dict[str, Any]): settings dictionary
            - times (np.ndarray): array of datetime objects representing time intervals
            - domain (Dict[str, Any]): domain information including lonmin, latmin, dx, dy
        data (Dict[str, Any]): additional data dictionary
            - srs (np.ndarray): numpy array of shape [ntimes, nx, ny, nsamples] of SRS values
            - Qfact (float): scaling factor for source model
            - samples (pd.DataFrame): dataframe with sample information
            - obs (np.ndarray): observed concentrations
            - obs_error (np.ndarray): observation errors
            - MDC (np.ndarray): minimum detectable concentration
        lon (Optional[float]): longitude of location to plot
        lat (Optional[float]): latitude of location to plot
        outpath (Optional[str]): output path for saving plots
        show (bool): whether to display the plot
    """
    srs = data['srs']
    Qfact = data['Qfact']
    samples = data['samples']
    obs = data['obs']
    obs_error = data['obs_error']
    MDC = data['MDC']

    if lon is None or lat is None:
        ix, iy = np.unravel_index(np.argmin(cost), cost.shape)
        lon = settings['domain']['lonmin'] + ix * settings['domain']['dx']
        lat = settings['domain']['latmin'] + iy * settings['domain']['dy']
    else:
        ix = lon2ix(lon, settings['domain']['lonmin'], settings['domain']['dx'])
        iy = lat2iy(lat, settings['domain']['latmin'], settings['domain']['dy'])

    ntimes = len(settings['times']) - 1  # Number of release periods (intervals between times)
    nisotopes = Qs.shape[0] // ntimes if ntimes > 0 else 1
    nobs = srs.shape[3]

    #Subplots definition
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 3, 3])
    
    ax_release = fig.add_subplot(gs[0, :])
    ax_ac = fig.add_subplot(gs[1, :])
    ax_rel = fig.add_subplot(gs[2, :])

    #release bar plot - use same approach as R (plot all Qs values with colors by segment)
    release_values = Qs[:, ix, iy]  # Shape: ntimes*nisotopes
    
    # Generate color palette with ntimes colors (matching R behavior)
    color_list = ["darkorange", "lemonchiffon", "darkgreen", "lightpink", "royalblue"]
    if ntimes <= len(color_list):
        barcols = color_list[:ntimes]
    else:
        cmap = LinearSegmentedColormap.from_list('release_colors', color_list)
        barcols = [cmap(i / (ntimes - 1)) for i in range(ntimes)]
    
    # For multiple isotopes, create side-by-side bars (like R does)
    # Bar width and spacing for side-by-side bars
    bar_width = 0.8 / nisotopes if nisotopes > 1 else 0.8
    x_base = np.arange(ntimes)
    
    # Plot bars for each isotope
    for iso in range(nisotopes):
        x_positions = x_base + (iso - nisotopes/2 + 0.5) * bar_width
        Qs_i = release_values[iso*ntimes:(iso+1)*ntimes]
        # Use time segment colors (bar k gets barcols[k], same for all isotopes)
        colors = [barcols[j % len(barcols)] for j in range(ntimes)]
        ax_release.bar(x_positions, Qs_i, label=f'Isotope {iso+1}' if nisotopes > 1 else None,
                      color=colors, alpha=0.7, width=bar_width)
    
    # Set ticks at time segment positions
    if len(settings['times']) > 1:
        ax_release.set_xticks(np.arange(ntimes))
        ax_release.set_xticklabels([f"{settings['times'][i].strftime('%m-%d')}" for i in range(ntimes)])
    
    ax_release.set_ylabel('Release [Bq]')
    ax_release.set_title('Release Term at Selected Location')
    if nisotopes > 1:
        ax_release.legend()
    ax_release.grid(True, which='both', linestyle='--', alpha=0.5, axis='y')

    # AC time series plot
    mod_each_release = np.zeros((ntimes, nobs))
    for i in range(nisotopes):
        idx_release = slice(i*ntimes, (i+1)*ntimes)
        idx_obs = slice(int(i*nobs/nisotopes), int((i+1)*nobs/nisotopes))
        mod_each_release[:, idx_obs] = (Qs[idx_release, ix, iy][:, None] * srs[:, ix, iy, idx_obs]) * Qfact
    mod = np.sum(mod_each_release, axis=0)
    plot_ac_timeseries(
        ac=None,
        obs=obs,
        obs_min=obs - obs_error,
        obs_max=obs + obs_error,
        mod=mod,
        mod_min=mod - mod_error_cost,
        mod_max=mod + mod_error_cost,
        MDC=MDC,
        obsunitfact=settings['obsunitfact'],
        obsscalingfact=settings['obsscalingfact'],
        labels=samples['Entity'],
        title=f"lon = {lon:.2f}, lat = {lat:.2f}",
        outpath=outpath,
        show=show,
        ax=ax_ac
    )

    # Relative contribution bar plot - use same colors as release bars
    mod_rel = mod_each_release / (mod + 1e-12)
    for i in range(ntimes):
        ax_rel.bar(np.arange(nobs), mod_rel[i, :]*100, bottom=np.sum(mod_rel[:i, :], axis=0)*100,
                   color=barcols[i % len(barcols)], label=f"Segment {i+1}")
    ax_rel.set_ylabel('Relative contribution [%]')
    ax_rel.set_xlabel('Sample')
    ax_rel.set_xticks(np.arange(nobs))
    ax_rel.set_xticklabels(samples['Entity'], rotation=45, ha='right')
    ax_rel.legend(loc='upper right', fontsize=9)

    fig.suptitle(f"AC Expert Panel - lon={lon:.2f}, lat={lat:.2f}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if outpath is not None:
        fig.savefig(f"{outpath}/ac_expertPanel.pdf")
    if show:
        plt.show()
    else:
        plt.close(fig)

def write_cost(optsed: np.ndarray, mod_error_cost: np.ndarray, 
               settings: Dict[str, Any], 
               data: Dict[str, Any],
               flag_save: bool = True):
    """Write objects to file for diagnostics

    Parameters:
        optsed (np.ndarray): dictionary containing optimization results
            - 'cost': 2D numpy array with cost values
            - 'accQ': 2D numpy array with accumulated Q values
            - 'Qs': 2D numpy array with optimized source parameters
        mod_error_cost (np.ndarray): model error for cost function
        settings (Dict[str, Any]): dictionary of settings
            - outpath (str): output path for saving files
            - lower_cost (np.ndarray): lower cost threshold
            - upper_cost (np.ndarray): upper cost threshold
            - domain (Dict[str, Any]): domain information
            - IMSfile (str): path to IMS file
            - reactorfile (str): path to reactor file
            - trueValues (np.ndarray): true values of parameters
            - times (np.ndarray): array of datetime objects representing time intervals
            - Qmin (float): minimum Q value for plotting
            - Qmax (float): maximum Q value for plotting
        data (Dict[str, Any]): additional data dictionary
            - srs (np.ndarray): numpy array of shape [ntimes, nx, ny, nsamples] of SRS values
            - Qfact (float): scaling factor for source model
            - samples (pd.DataFrame): dataframe with sample information
            - obs (np.ndarray): observed concentrations
            - obs_error (np.ndarray): observation errors
            - MDC (np.ndarray): minimum detectable concentration
        flag_save (bool): whether to save the data to files
    """
    if flag_save:
        np.savez(settings['outpath'] + '/optsed.npz', **optsed)
        np.save(settings['outpath'] + '/lower_cost.npy', settings['lower_cost'])
        np.save(settings['outpath'] + '/upper_cost.npy', settings['upper_cost'])
        np.save(settings['outpath'] + '/mod_error_cost.npy', mod_error_cost)

    cost = optsed['cost']
    accQ = optsed['accQ']

    title = "Residual cost after optimisation"
    plot_cost(
        cost=cost,
        domain=settings['domain'],
        IMSfile=settings['IMSfile'],
        reactorfile=settings['reactorfile'],
        title=title,
        outpath=settings['outpath'],
        show=True
    )

    if np.quantile(cost, 0.1) < 2:
        accQ = accQ.copy()
        accQ[cost > 2] = 0
    title = "Accumulated release [Bq]"
    plot_accQ(
        accQ=accQ,
        domain=settings['domain'],
        IMSfile=settings.get("IMSfile"),
        reactorfile=settings.get("reactorfile"),
        title=title,
        outpath=settings['outpath'],
        show=True
    )

    lon = lat = None
    # Use true values if available
    if not np.isnan(settings['trueValues'][0]) and not np.isnan(settings['trueValues'][1]):
        lon = settings['trueValues'][0]
        lat = settings['trueValues'][1]
        ix = lon2ix(lon=lon, lon0=settings['domain']['lonmin'], dx=settings['domain']['dx'])
        iy = lat2iy(lat=lat, lat0=settings['domain']['latmin'], dy=settings['domain']['dy'])
    else:
        aux = np.unravel_index(np.argmin(optsed['cost'], axis=None), optsed['cost'].shape)
        ix, iy = aux
        lon = ix2lon(ix=ix, lon0=settings['domain']['lonmin'], dx=settings['domain']['dx'])
        lat = iy2lat(iy=iy, lat0=settings['domain']['latmin'], dy=settings['domain']['dy'])

    title = f"Source term (lon = {lon:.2f}, lat = {lat:.2f})"
    Qs_loc = optsed['Qs'][:, ix, iy]
    ymin, ymax = settings['Qmin'], settings['Qmax']
    plot_release_term(
        times=settings['times'],
        Qs=Qs_loc,
        title=title,
        outpath=settings['outpath'],
        show=True
    )

    nobs = len(data['obs'])
    width = min(12 + np.log(nobs), (0.8 + 2 / nobs**2) * nobs)
    plot_AC(
        Qs=optsed['Qs'],
        ix=ix,
        iy=iy,
        mod_error_cost=mod_error_cost,
        settings=settings,
        data=data,
        outpath=settings['outpath'],
        show=True
    )

    if optsed['Qs'].shape[0] > 1:
        plot_AC_expertPanel(
            Qs=optsed['Qs'],
            mod_error_cost=mod_error_cost,
            cost=optsed['cost'],
            settings=settings,
            data=data,
            lon=lon,
            lat=lat,
            outpath=settings['outpath'],
            show=True
        )
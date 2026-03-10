import numpy as np
import pandas as pd
import pickle
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from typing import Dict, Any, Optional, Tuple
from scipy.stats import gaussian_kde

from frear.analyse_chain import plot_trace
from frear.MCMC_aux import plot_monitorMCMC
from frear.analyse_chain import monovar_marginal_post
from frear.analyse_chain import bivariate_marginal_post
from frear.plot_timeseries import plot_ac_timeseries
from frear.plot_geofield import add_IMS, add_reactors

def get_zoomed_lon_range(lons: np.ndarray, bound: int) -> Tuple[float, float, float]:
    """
    Get zoomed longitude range handling crossing of 180 meridian

    Parameters:
        lons (np.ndarray): Array of longitudes
        bound (int): Boundary to add around the min/max longitude

    Returns:
        Tuple[float, float, float]: Zoomed longitude range (lonmin, lonmax, central_lon)
    """
    # Algorithm finding largest gap in sorted unique longitudes to determine crossing of 180 meridian

    unique_lons = np.sort(np.unique(lons))

    if len(unique_lons) == 1:
        lonmin = unique_lons[0] - bound
        lonmax = unique_lons[0] + bound
        return lonmin, lonmax, unique_lons[0]

    gaps = np.diff(unique_lons)
    gaps = np.append(gaps, 360 - (unique_lons[-1] - unique_lons[0]))
    
    max_gap_idx = np.argmax(gaps)
    
    if max_gap_idx == len(unique_lons) - 1:
        # Normal case (data does not cross the 180/-180 seam) -> min/max of unique lons
        lon_start, lon_end = unique_lons[0], unique_lons[-1]
    else:
        # Crossing the meridian: choose opposite of gap as start/end
        lon_start = unique_lons[max_gap_idx + 1]
        lon_end = unique_lons[max_gap_idx] + 360
        
    lonmin = lon_start - bound
    lonmax = lon_end + bound

    central_lon = ((lon_start + lon_end) / 2.)

    return lonmin, lonmax, central_lon


def plot_probloc_plain(probloc: np.ndarray, domain: Dict[str, Any], title: str = "", IMSfile: Optional[str] = None, reactorfile: Optional[str] = None, outpath: Optional[str] = None, show: bool = True):
    """
    Plot probability over lat/lon grid

    Parameters:
        probloc (np.ndarray): 2D array of probabilities over the domain grid
        domain (Dict[str, Any]): Domain information with keys 'lonmin', 'lonmax', 'latmin', 'latmax', 'dx', 'dy'
        title (str): Title of the plot
        IMSfile (Optional[str]): Path to IMS stations file for plotting
        reactorfile (Optional[str]): Path to reactor locations file for plotting
        outpath (Optional[str]): Directory to save the plot
        show (bool): Whether to display the plot
    """
    lon = np.arange(domain['lonmin'], domain['lonmax'] + domain['dx'], domain['dx'])
    lat = np.arange(domain['latmin'], domain['latmax'] + domain['dy'], domain['dy'])
    lons, lats = np.meshgrid(lon, lat, indexing='ij')

    # Color levels
    levels = np.array([0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1.0])
    colors = ['red', 'yellow', 'royalblue', 'white'][::-1]
    cmap = mcolors.LinearSegmentedColormap.from_list("plume_cmap", colors, N=len(levels)-1)

    scale_fig = (domain['lonmax'] - domain['lonmin']) / (domain['latmax'] - domain['latmin'])
    fig = plt.figure(figsize=(10, 10 * (1/scale_fig)**(2/3)))
    ax = plt.axes(projection=ccrs.PlateCarree())
    cf = ax.pcolormesh(lons, lats, probloc, cmap=cmap, shading='auto', norm=mcolors.BoundaryNorm(levels, ncolors=cmap.N))

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.gridlines(draw_labels=True)

    # IMS stations
    if IMSfile is not None:
        add_IMS(ax, IMSfile, domain)
    
    if reactorfile is not None:
        add_reactors(ax, reactorfile, domain)

    plt.title(title)
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="Probability")

    if outpath is not None:
        plt.savefig(os.path.join(outpath, 'probloc_plain.pdf'))
    if show:
        plt.show()

def plot_probloc_kde(chainburned: np.ndarray, domain: Dict[str, Any], 
                     IMSfile: Optional[str] = None, reactorfile: Optional[str] = None, 
                     breaks = np.arange(0.1, 1.1, 0.1), title='Source density',
                     outpath: Optional[str] = None, filename: str = 'probloc_kde.pdf',
                     show: bool = True):
    """Plot probability over location using kernel density estimation
    
    Parameters:
        chainburned (np.ndarray): Burned MCMC chain with source locations
        domain (Dict[str, Any]): Domain information with keys 'lonmin', 'lonmax', 'latmin', 'latmax', 'dx', 'dy'
        IMSfile (Optional[str]): Path to IMS stations file for plotting
        reactorfile (Optional[str]): Path to reactor locations file for plotting
        breaks (np.ndarray): Breakpoints for contour levels
        title (str): Title of the plot
        outpath (Optional[str]): Directory to save the plot
        filename (str): Filename for the saved plot
        show (bool): Whether to display the plot
    """
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=domain.get('central_lon', 0)))

    x = chainburned[:,0]
    y = chainburned[:,1]
    
    #Kernel density estimation
    xy = np.vstack([x,y])
    kde = gaussian_kde(xy)
    xi, yi = np.mgrid[domain['lonmin']:domain['lonmax']:200j, domain['latmin']:domain['latmax']:200j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    
    zi_norm = zi / zi.max()
    
    cf = ax.contourf(xi, yi, zi_norm, levels=breaks, cmap='Reds', alpha=0.6,
                     transform=ccrs.PlateCarree())
    
    ax.set_extent([domain['lonmin'], domain['lonmax'], 
                   domain['latmin'], domain['latmax']], 
                  crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.gridlines(draw_labels=True)

    fig.colorbar(cf, ax=ax, label='Normalized density')
    
    if reactorfile is not None:
        add_reactors(ax, reactorfile, domain)
    if IMSfile is not None:
        add_IMS(ax, IMSfile, domain)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    
    if outpath is not None:
        plt.savefig(os.path.join(outpath, filename))
    if show:
        plt.show()

def plot_probloc_kde_zoom(chainburned: np.ndarray, domain: Dict[str, Any], 
                          bound: int = 2, breaks: np.ndarray =np.arange(0.1, 1.1, 0.1), 
                          IMSfile: Optional[str] = None, reactorfile: Optional[str] = None,
                          outpath: Optional[str] = None, filename: str = 'probloc_kde_zoom.pdf',
                          show: bool = True):
    """Plot zoomed probability over location using kernel density estimation
    
    Parameters:
        chainburned (np.ndarray): Burned MCMC chain with source locations
        domain (Dict[str, Any]): Domain information with keys 'lonmin', 'lonmax', 'latmin', 'latmax', 'dx', 'dy'
        IMSfile (Optional[str]): Path to IMS stations file for plotting
        reactorfile (Optional[str]): Path to reactor locations file for plotting
        breaks (np.ndarray): Breakpoints for contour levels
        title (str): Title of the plot
        outpath (Optional[str]): Directory to save the plot
        filename (str): Filename for the saved plot
        show (bool): Whether to display the plot
    """
    zoomed_domain = domain.copy()
    
    lons = chainburned[:,0].copy()
    lonmin, lonmax, central_lon = get_zoomed_lon_range(lons, bound)
    zoomed_domain['lonmin'] = lonmin
    zoomed_domain['lonmax'] = lonmax
    zoomed_domain['central_lon'] = central_lon
    
    # Latitude has no issues (probably no measurements near poles)
    zoomed_domain['latmin'] = np.floor(np.min(chainburned[:,1])) - bound
    zoomed_domain['latmax'] = np.ceil(np.max(chainburned[:,1])) + bound
    
    plot_probloc_kde(chainburned, zoomed_domain, breaks=breaks, IMSfile=IMSfile, 
                     reactorfile=reactorfile, outpath=outpath, filename=filename, show=show)


def write_bayes(out: Dict[str, Any], mcmc: Dict[str, Any], 
                settings: Dict[str, Any], misc: Dict[str, Any],
                obs: np.ndarray, obs_error: np.ndarray, 
                data: Dict[str, Any], lsave=True):
    """Write objects to file and plot
    
    Parameters:
        out (Dict[str, Any]): MCMC output data
        mcmc (Dict[str, Any]): MCMC chain and statistics
        settings (Dict[str, Any]): Configuration settings
        misc (Dict[str, Any]): Miscellaneous data
        obs (np.ndarray): Observations
        obs_error (np.ndarray): Observation errors
        data (Dict[str, Any]): Additional data including AC function and samples
        lsave (bool): Whether to save output files
    """
    if lsave:
        upper_bayes = data['upper_bayes']
        lower_bayes = data['lower_bayes']
        mod_error_bayes = settings['mod_error_bayes']
        zchainSummary = mcmc['zchainsummary']
        post_median = mcmc['post_median']
        probloc = mcmc['probloc']
        np.save(settings['outpath'] + '/upper_bayes.npy', upper_bayes)
        np.save(settings['outpath'] + '/lower_bayes.npy', lower_bayes)
        pickle.dump(out, settings['outpath'] + '/out.pkl')
        np.save(settings['outpath'] + '/mod_error_bayes.npy', mod_error_bayes)
        np.save(settings['outpath'] + '/zchainSummary.npy', zchainSummary)
        np.save(settings['outpath'] + '/post_median.npy', post_median)
        np.save(settings['outpath'] + '/probloc.npy', probloc)

    outpath = settings['outpath']
    npar = len(settings['parnames'])

    ARs = out['ARs']
    Rhats = out['Rhats']
    checkfreq = out['checkfreq']
    chainburned = mcmc['chainburned']
    domain = settings['domain']

    ACfun_bayes = data['ACfun_bayes']
    MDC = data['MDC']
    samples = data['samples']

    if lsave:
        np.savetxt(f"{outpath}/bayes_summary.txt", mcmc['zchainsummary'], delimiter="\t")

    plot_trace(zchains=out['chains'], ipars=list(range(npar)) + [npar + 2], parnames=settings['parnames'] + ['posterior'], outpath=outpath, show=True)

    plot_monitorMCMC(ARs=ARs, Rhats=Rhats, checkfreq=checkfreq, parnames=settings['parnames'] + ['posterior'], outpath=outpath, show=True, separate=False)

    monovar_marginal_post(chainburned=chainburned, times = settings['times'], 
                          lower_bayes=data['lower_bayes'], upper_bayes=data['upper_bayes'], 
                          settings = settings,
                          true_values = settings['trueValues'],
                          post_median = mcmc['post_median'],
                          post_mode = mcmc['post_mode'],
                          parnames = settings['parnames'],
                          lpriorrange=True,
                          outpath=outpath, show=True)


    monovar_marginal_post(chainburned=chainburned, times = settings['times'], 
                          lower_bayes=data['lower_bayes'], upper_bayes=data['upper_bayes'], 
                          settings = settings,
                          true_values = settings['trueValues'],
                          post_median = mcmc['post_median'],
                          post_mode = mcmc['post_mode'],
                          parnames = settings['parnames'],
                          lpriorrange=False,
                          outpath=outpath, show=True)

    bivariate_marginal_post(chainburned=chainburned, times = settings['times'],
                            lower_bayes=data['lower_bayes'], upper_bayes=data['upper_bayes'], 
                            settings=settings,
                            true_values = settings['trueValues'],
                            post_median = mcmc['post_median'],
                            post_mode = mcmc['post_mode'],
                            parnames = settings['parnames'],
                            outpath=outpath, show=True)

    nsamples = len(obs)
    width = min(12 + np.log(nsamples),
                (0.8 + 2/nsamples**2) * nsamples)
    height = 0.7 * 7

    #Posterior median
    title = "Modelled concentrations using posterior median"
    plot_ac_timeseries(
        ac=ACfun_bayes(mcmc["post_median"]),
        MDC=MDC,
        obsunitfact=settings["obsunitfact"],
        obsscalingfact=settings["obsscalingfact"],
        title=title,
        labels=samples["Entity"],
        prefix = "bayes_postmedian_",
        show=True,
        outpath=outpath
    )

    #Posterior mode
    if not np.any(np.isnan(mcmc["post_mode"])):
        title = "Modelled concentrations using posterior mode"

        plot_ac_timeseries(
            ac=ACfun_bayes(mcmc["post_mode"]),
            MDC=MDC,
            obsunitfact=settings["obsunitfact"],
            obsscalingfact=settings["obsscalingfact"],
            title=title,
            labels=samples["Entity"],
            prefix = "bayes_postmode_",
            show=True,
            outpath=outpath
        )


    #Posterior chain
    mods = np.array([ACfun_bayes(sample)["mod"]
                    for sample in mcmc["chainburned"]])
    mod = np.median(mods, axis=0)
    mod_min = np.quantile(mods, 0.025, axis=0)
    mod_max = np.quantile(mods, 0.975, axis=0)
    title = "Modelled concentrations using full chain"

    plot_ac_timeseries(
        obs=obs,
        obs_min=obs - obs_error,
        obs_max=obs + obs_error,
        mod=mod,
        mod_min=mod_min,
        mod_max=mod_max,
        MDC=MDC,
        obsunitfact=settings["obsunitfact"],
        obsscalingfact=settings["obsscalingfact"],
        title=title,
        labels=samples["Entity"],
        prefix = "bayes_fullchain_",
        show=True,
        outpath=outpath
    )

    probloc = mcmc['probloc']
    if probloc is not None:
        #plain plot
        plot_probloc_plain(probloc, domain, title="Bayesian source location probability",
                        IMSfile=settings['IMSfile'], reactorfile=settings['reactorfile'], outpath=outpath)

        if np.sum(np.array(probloc) != 0) > 1:
            # KDE plot
            plot_probloc_kde(chainburned, domain, IMSfile=settings['IMSfile'], reactorfile=settings['reactorfile'],
                            breaks=np.arange(0.1,1.1,0.1), outpath=outpath)
            
            # KDE zoomed
            plot_probloc_kde_zoom(chainburned, domain, bound=settings['nProbLocBoundary'],
                                IMSfile=settings['IMSfile'], reactorfile=settings['reactorfile'], outpath=outpath)
    

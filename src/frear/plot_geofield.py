import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from typing import Dict, Any, List, Optional
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm


def add_reactors(ax: plt.Axes, reactorfile: str, domain: Dict[str, Any], 
                 marker='^', color='darkblue', size=50):
    """Add reactors to plot
    
    Parameters:
        ax (plt.Axes): Matplotlib Axes to plot on
        reactorfile (str): Path to reactor data file
        domain (Dict[str, Any]): Domain dictionary with 'lonmin', 'lonmax', 'latmin', 'latmax'
        marker (str): Marker style for reactors
        color (str): Color for reactor markers
        size (int): Size of reactor markers
    """
    if os.path.exists(reactorfile):
        cols = ["Reactor", "Country", "Latitude", "Longitude", "NA1", "NA2"]
        reactors = pd.read_csv(reactorfile, delim_whitespace=True, names=cols)
        reactors = reactors[(reactors['Longitude'] >= domain['lonmin']) & (reactors['Longitude'] <= domain['lonmax']) &
                            (reactors['Latitude'] >= domain['latmin']) & (reactors['Latitude'] <= domain['latmax'])]
        ax.scatter(reactors['Longitude'], reactors['Latitude'], marker=marker, color=color, s=size, zorder=10)
    else:
        print(f"Warning: Reactor file '{reactorfile}' not found")

def add_IMS(ax: plt.Axes, IMSfile: str, domain: Dict[str, Any], 
            stat_number: Optional[List[int]] = None, 
            use_labels: bool = True, label_type: str = 'code', 
            marker: str = 'o', color: str = 'black', size: int = 40):
    """Add IMS stations to plot
    
    Parameters:
        ax (plt.Axes): Matplotlib Axes to plot on
        IMSfile (str): Path to IMS data file
        domain (Dict[str, Any]): Domain dictionary with 'lonmin', 'lonmax', 'latmin', 'latmax'
        stat_number (Optional[List[int]]): List of station numbers to filter
        use_labels (bool): Whether to use labels for stations
        label_type (str): Type of label to use ('code' or 'number')
        marker (str): Marker style for IMS stations
        color (str): Color for IMS markers
        size (int): Size of IMS markers
    """
    if os.path.exists(IMSfile):
        cols = ["stat_code", "stat_number", "status", "lat", "lon"]
        IMS = pd.read_csv(IMSfile, delim_whitespace=True, names=cols)
        IMS = IMS[(IMS['lon'] >= domain['lonmin']) & (IMS['lon'] <= domain['lonmax']) &
                  (IMS['lat'] >= domain['latmin']) & (IMS['lat'] <= domain['latmax'])]
        if stat_number is not None:
            IMS = IMS[IMS['stat_number'].isin(stat_number)]
        if len(IMS) > 0:
            ax.scatter(IMS['lon'], IMS['lat'], marker=marker, color=color, s=size, zorder=10)
            if use_labels:
                labels = IMS['stat_code'] if label_type=='code' else IMS['stat_number']
                for lon, lat, label in zip(IMS['lon'], IMS['lat'], labels):
                    ax.text(lon, lat + 0.02, label, ha='center', va='bottom', color=color, fontsize=8)
    else:
        print(f"Warning: IMS file '{IMSfile}' not found")

def plot_2D_direct(
    data: np.ndarray,
    domain: Dict[str, Any],
    title: str = "",
    colors: list = ["white", "darkgreen", "yellow", "royalblue"],
    levels: Optional[np.ndarray] = None,
    labels: list = ["Longitude", "Latitude", "Value"],
    IMSfile: Optional[str] = None,
    reactorfile: Optional[str] = None,
    outpath: Optional[str] = None,
    filename: str = "2d_plot.png",
    show: bool = True
):
    """Direct 2D contourf plot (without KDE)
    
    Parameters:
        data (np.ndarray): 2D array (nx, ny) of values to plot
        domain (Dict[str, Any]): Domain dictionary with 'lonmin', 'lonmax', 'latmin', 'latmax', 'nx', 'ny'
        title (str): Title of the plot
        colors (list): List of colors for colormap
        levels (Optional[np.ndarray]): Contour levels
        labels (list): Labels for x, y, and colorbar
        IMSfile (Optional[str]): Path to IMS data file
        reactorfile (Optional[str]): Path to reactor data file
        outpath (Optional[str]): Output path to save the plot
        filename (str): Filename for saving the plot
        show (bool): Whether to display the plot
    """

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.gridlines(draw_labels=True)

    nx, ny = data.shape
    lons = np.linspace(domain["lonmin"], domain["lonmax"], ny)
    lats = np.linspace(domain["latmin"], domain["latmax"], nx)
    
    xi, yi = np.meshgrid(lons, lats)

    if levels is None:
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
        cf = ax.contourf(xi, yi, data, levels=20, cmap=cmap)
    else:
        N = len(levels) - 1    

        levels = np.asarray(levels, dtype=float)
        levels = levels[np.isfinite(levels)]  # Remove NaN/inf
        levels = np.unique(levels)
        
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=N)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        cf = ax.contourf(xi, yi, data, levels=levels, cmap=cmap, norm=norm, extend='max')
    
    fig.colorbar(cf, ax=ax, label=labels[2])
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    
    if IMSfile is not None:
        add_IMS(ax, IMSfile, domain)
    if reactorfile is not None:
        add_reactors(ax, reactorfile, domain)
    
    if outpath is not None:
        fig.savefig(f"{outpath}/{filename}", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_2D_kde(
    data: np.ndarray,
    domain: Dict[str, Any],
    title: str = "",
    colors: List[str] = ["white", "darkgreen", "yellow", "royalblue"],
    IMSfile: Optional[str] = None,
    reactorfile: Optional[str] = None,
    n_levels: int = 20,
    levels: Optional[np.ndarray] = None,
    labels: List[str] = ["Longitude", "Latitude", "Value"],
    show: bool = True,
    outpath: Optional[str] = None,
    filename: Optional[str] = "2D_kde_plot.png",
    ax: Optional[plt.Axes] = None
):
    """
    Plot a 2D KDE/fill contour plot
    
    Parameters:
        data (np.ndarray): 2D array (nx, ny) of values to plot
        domain (Dict[str, Any]): Domain dictionary with 'lonmin', 'lonmax', 'latmin', 'latmax', 'nx', 'ny'
        title (str): Title of the plot
        colors (List[str]): List of colors for colormap
        IMSfile (Optional[str]): Path to IMS data file
        reactorfile (Optional[str]): Path to reactor data file
        n_levels (int): Number of contour levels if levels not provided
        levels (Optional[np.ndarray]): Contour levels
        labels (List[str]): Labels for x, y, and colorbar
        show (bool): Whether to display the plot
        outpath (Optional[str]): Output path to save the plot
        filename (Optional[str]): Filename for saving the plot
        ax (Optional[plt.Axes]): Matplotlib Axes to plot on
    """
    nx, ny = data.shape
    lons = np.linspace(domain["lonmin"], domain["lonmax"], ny)
    lats = np.linspace(domain["latmin"], domain["latmax"], nx)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    x_flat = lon_grid.ravel()
    y_flat = lat_grid.ravel()
    z_flat = data.ravel()

    kde = gaussian_kde(np.vstack([x_flat, y_flat]), weights=z_flat)
    xi, yi = np.meshgrid(lons, lats)
    zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
    
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = None

    if levels is None:
        cf = ax.contourf(xi, yi, zi, levels=n_levels, cmap=cmap)
    else:
        if len(levels) > 1:
            levels = np.sort(levels)
        cf = ax.contourf(xi, yi, zi, levels=levels, cmap=cmap)
    if fig is not None:
        fig.colorbar(cf, ax=ax, label=labels[2])
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    
    if IMSfile is not None:
        add_IMS(ax, IMSfile, domain)
    if reactorfile is not None:
        add_reactors(ax, reactorfile, domain)
    
    if fig is not None:
        if outpath is not None:
            fig.savefig(f"{outpath}/{filename}", dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
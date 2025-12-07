import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Dict, Any, List

def check_parameters(theta: np.ndarray, prior: Dict[str, Any]) -> np.ndarray:
    """Check and adjust parameters to be within prior bounds
    
    Parameters:
        theta (np.ndarray): Parameter array
        prior (Dict[str, Any]): Prior information dictionary
            - upper_bayes (np.ndarray): upper bounds
            - lower_bayes (np.ndarray): lower bounds
            - density (callable): Density function
    Returns:
        theta (np.ndarray): Adjusted parameter array within bounds
    """
    upper_bayes = np.array(prior['upper_bayes'])
    lower_bayes = np.array(prior['lower_bayes'])

    while not np.all(np.isfinite(prior['density'](theta))):
        ind_toobig = np.where(theta > upper_bayes)[0]
        ind_toosmall = np.where(theta < lower_bayes)[0]
        theta[ind_toobig] -= (upper_bayes[ind_toobig] - lower_bayes[ind_toobig])
        theta[ind_toosmall] += (upper_bayes[ind_toosmall] - lower_bayes[ind_toosmall])

    return theta

def check_convergence(zchains: np.ndarray, ipars: List[int] = [0, 1, 2, 3, 4, 7]) -> np.ndarray:
    """Calculate Gelman-Rubin R-hat values for specified parameters
    
    Parameters:
        zchains (np.ndarray): MCMC chains array with shape (niterations, nparameters, nchains)
        ipars (List[int]): List of parameter indices to calculate R-hat for
    Returns:
        Rs (np.ndarray): Array of R-hat values for the specified parameters
    """
    # first, apply burnin by discarding the first half of each chain
    chainburned = zchains[zchains.shape[0] // 2 :, :, :]
    Rs = np.zeros(len(ipars))
    for idx, ipar in enumerate(ipars):
        aux = chainburned[:, ipar, :]
        # Split each chain in half to double the number of chains (matching R's approach)
        # R: aux <- array(aux, dim=c(dim(aux)*c(.5,2)))
        # This splits n iterations × m chains into (n/2) iterations × (2m) chains
        niter_half = aux.shape[0] // 2
        aux_first_half = aux[:niter_half, :]  # First half of all chains
        aux_second_half = aux[niter_half:2*niter_half, :]  # Second half of all chains
        aux = np.hstack([aux_first_half, aux_second_half])  # Concatenate horizontally
        
        n = aux.shape[0]  # length of one sequence (now half the original)
        m = aux.shape[1]  # number of sequences (now double the original)
        B = n / (m - 1) * np.sum((aux.mean(axis=0) - aux.mean()) ** 2)
        W = np.mean(np.var(aux, axis=0, ddof=1))
        varplus = (n - 1) / n * W + 1 / n * B
        Rs[idx] = np.sqrt(varplus / W)
    return Rs

def plot_ARs(ARs: np.ndarray, checkfreq=100, 
             outpath: Optional[str] = None, show: bool = True, 
             ax: Optional[plt.Axes] = None) -> Optional[plt.Axes]:
    """Plot acceptance rates of all chains
    
    Parameters:
        ARs (np.ndarray): Acceptance rates array
        checkfreq (int): Frequency at which acceptance rates were checked
        outpath (Optional[str]): Output path to save the plot
        show (bool): Whether to display the plot
        ax (Optional[plt.Axes]): Matplotlib Axes object to plot on
    """

    nchains = ARs.shape[0]
    x = checkfreq * np.arange(1, ARs.shape[1] + 1)

    if ax is None:
        plt.figure(figsize=(10, 10))
        for i in range(nchains):
            plt.plot(x, ARs[i, :], label=f'Chain {i+1}')
        plt.axhline(y=23.4, color='black', linestyle='--')
        plt.xlabel('Iterations')
        plt.ylabel('Acceptance Rate [%]')
        plt.title('Acceptance Rates of MCMC Chains')
        plt.legend()

        if outpath is not None:
            plt.savefig(outpath + '/acceptance_rates.pdf')
        if show:
            plt.show()
        return None
    else:
        for i in range(nchains):
            ax.plot(x, ARs[i, :], label=f'Chain {i+1}')
        ax.axhline(y=23.4, color='black', linestyle='--')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Acceptance Rate [%]')
        ax.set_title('Acceptance Rates of MCMC Chains')
        ax.legend()
        return ax

def plot_Rhats(Rhats: np.ndarray, checkfreq=100, 
               parnames: Optional[List[str]] = None, 
               outpath: Optional[str] = None, show: bool = True, 
               ax: Optional[plt.Axes] = None) -> Optional[plt.Axes]:
    """Plot Gelman-Rubin R-hat values for all parameters

    Parameters:
        Rhats (np.ndarray): R-hat values array
        checkfreq (int): Frequency at which R-hat values were checked
        parnames (List[str], optional): Names of the parameters for labels
        outpath (Optional[str]): Output path to save the plot
        show (bool): Whether to display the plot
        ax (Optional[plt.Axes]): Matplotlib Axes object to plot on
    """

    x = checkfreq * np.arange(1, Rhats.shape[1] + 1)

    if ax is None:
        plt.figure(figsize=(10, 10))
        for i in range(Rhats.shape[0]):
            label = parnames[i] if parnames is not None else f'Param {i+1}'
            plt.plot(x, Rhats[i, :], label=label)
        plt.axhline(y=1.2, color='black', linestyle='--')
        plt.ylim(1, 2)
        plt.xlabel('Iterations')
        plt.ylabel('Gelman-Rubin R-hat')
        plt.title('Gelman-Rubin R-hat Values of MCMC Chains')
        plt.legend()

        if outpath is not None:
            plt.savefig(outpath + '/gelman_rubin_rhat.pdf')
        if show:
            plt.show()
        return None
    else:
        for i in range(Rhats.shape[0]):
            label = parnames[i] if parnames is not None else f'Param {i+1}'
            ax.plot(x, Rhats[i, :], label=label)
        ax.axhline(y=1.2, color='black', linestyle='--')
        ax.set_ylim(1, 2)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Gelman-Rubin R-hat')
        ax.set_title('Gelman-Rubin R-hat Values of MCMC Chains')
        ax.legend()
        return ax

def plot_monitorMCMC(ARs: np.ndarray, Rhats: np.ndarray, 
                     checkfreq: int, parnames: List[str], 
                     outpath: Optional[str] = None, show: bool = True, 
                     separate: bool = True) -> None:
    """Plot acceptance rates and Gelman-Rubin R-hat values for MCMC chains
    
    Parameters:
        ARs (np.ndarray): Acceptance rates array
        Rhats (np.ndarray): R-hat values array
        checkfreq (int): Frequency at which values were checked
        parnames (List[str]): Names of the parameters for labels
        outpath (Optional[str]): Output path to save the plots
        show (bool): Whether to display the plots
        separate (bool): Whether to plot acceptance rates and R-hat values separately
    """
    if separate:
        plot_ARs(ARs, checkfreq, outpath, show)
        plot_Rhats(Rhats, checkfreq, parnames, outpath, show)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 20))
        plot_ARs(ARs, checkfreq, None, False, ax1)
        plot_Rhats(Rhats, checkfreq, parnames, None, False, ax2)
        if outpath is not None:
            plt.savefig(outpath + '/mcmc_monitor.pdf')
        if show:
            plt.show()
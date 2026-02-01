import numpy as np
import time

from typing import Dict, Any, Optional
from multiprocessing import Pool

from frear.MCMC_aux import check_parameters, check_convergence
from frear.bayes import make_prior, make_posterior, make_likelihood

################################################################################
# Create proposals using the DREAMzs algorithm (also used by MT-DREAMzs)
# Valid for a fixed chain using an archive that can be shared by multiple chains
# X: current chain position
# Z: common archive of states
# DEpairs: number of pairs selected from the archive for the Differential
#          Evolution
# npar: number of parameters
# CR: crossover probabilitiy
# gammaFactor: a scale factor for gamma which allows to control the acceptance 
#              rate
# e: function returning a draw from a uniform distribution with dimensions npar
# eps: function returning a draw from a multinormal distribution with 
#      dimensions npar
################################################################################

def make_proposal(X: np.ndarray, Z: np.ndarray, 
                  M: int, DEpairs: int, npar: int, CR: float,
                  gammaFactor: float, e: Any, eps: Any) -> np.ndarray:
    """Create a proposal using the DREAMzs algorithm
    
    Parameters:
        X (np.ndarray): Current chain position
        Z (np.ndarray): Common archive of states
        M (int): Number of states in the archive
        DEpairs (int): Number of pairs selected from the archive for Differential Evolution
        npar (int): Number of parameters
        CR (float): Crossover probability
        gammaFactor (float): Scale factor for gamma to control acceptance rate
        e (Any): Function returning a draw from a uniform distribution with dimensions npar (noise)
        eps (Any): Function returning a draw from a multinormal distribution with dimensions npar (noise)
    Returns:
        proposal (np.ndarray): Proposed new state
    """

    r = np.random.choice(np.arange(M), size = 2 * DEpairs, replace=False)
    r1 = r[:DEpairs]
    r2 = r[DEpairs:]

    indX = np.where(np.random.rand(npar) > (1 - CR))[0]
    if len(indX) == 0:
        indX = np.random.choice(np.arange(npar), size=1)

    if np.random.rand() > 0.8:
        gamma = 1.0
    else:
        gamma = gammaFactor * 2.38 / np.sqrt(2 * DEpairs * len(indX))

    proposal = np.copy(X)
    delta = np.sum(Z[r1][:, indX], axis=0) - np.sum(Z[r2][:, indX], axis=0)
    proposal[indX] = X[indX] + (1 - e()[indX]) * gamma * delta + eps()[indX]
    return proposal

################################################################################
# Calculate proposal without using a snooker update
################################################################################

def calc_proposal_noSnooker(X: np.ndarray, Z: np.ndarray, M: int, CR: float,
                             prior: Any, posterior: Any, e: Any, eps: Any,
                             DEpairs: int, zgammaFactor: float,
                             npar: int, nMTM: int) -> Dict[str, Any]:
    """Calculate proposals without snooker update using DREAMzs algorithm
    
    Parameters:
        X (np.ndarray): Current chain position
        Z (np.ndarray): Common archive of states
        M (int): Number of states in the archive
        CR (float): Crossover probability
        prior (Any): Prior distribution object
        posterior (Any): Posterior distribution object
        e (Any): Function returning a draw from a uniform distribution with dimensions npar (noise)
        eps (Any): Function returning a draw from a multinormal distribution with dimensions npar (noise)
        DEpairs (int): Number of pairs selected from the archive for Differential Evolution
        zgammaFactor (float): Scale factor for gamma to control acceptance rate
        npar (int): Number of parameters
        nMTM (int): Number of proposals for Multiple Try Metropolis
    Returns:
        proposals (Dict[str, Any]): Dictionary containing proposals and their posterior values
            - zprops_post (np.ndarray): Posterior values of z proposals
            - xprops_post (np.ndarray): Posterior values of x proposals
            - z (np.ndarray): Selected z proposal
    """
    
    zprops = np.tile(X.reshape((npar, 1)), (1, nMTM))
    zprops_post = np.full((nMTM,), -np.inf)

    counter = 0
    while np.any(~np.isfinite(zprops_post)):
        counter += 1
        if counter > 1:
            print("Problem in first while loop - This should not happen - counter:", counter)
            print("zprops_post:", zprops_post)
            print(posterior['density'](zprops.T))

        for iprop in range(nMTM):
            zprops[:, iprop] = make_proposal(X, Z, M, DEpairs, npar, CR, zgammaFactor, e, eps)

        for iprop in range(nMTM):
            zprops[:, iprop] = check_parameters(zprops[:, iprop], prior)

        zprops_post = posterior['density'](zprops.T)["posterior"]
    
    aux = zprops_post.copy()
    while np.sum(np.exp(aux)) == 0:
        aux += 1  # prob get normalised later so one can always add a constant to log(prop)
    aux = np.exp(aux)
    aux /= np.sum(aux)
    z = zprops[:, np.random.choice(np.arange(nMTM), size=1, p=aux)[0]]

    xprops = np.tile(X.reshape((npar, 1)), (1, nMTM))
    xprops_post = np.full((nMTM,), -np.inf)

    counter = 0
    while np.any(~np.isfinite(xprops_post)):
        counter += 1
        if counter > 1:
            print("Problem in second while loop - This should not happen - counter:", counter)
            print("xprops_post:", xprops_post)
            print(posterior['density'](xprops.T))

        for iprop in range(nMTM - 1):
            xprops[:, iprop] = make_proposal(z, Z, M, DEpairs, npar, CR, zgammaFactor, e, eps)

        xprops[:, nMTM - 1] = X
        for iprop in range(nMTM):
            xprops[:, iprop] = check_parameters(xprops[:, iprop], prior)

        xprops_post = posterior['density'](xprops.T)["posterior"]
    return {"zprops_post": zprops_post,
            "xprops_post": xprops_post,
            "z": z}

################################################################################
# Calculate proposal without using a snooker update
################################################################################

def calc_proposal_snooker(X: np.ndarray, Z: np.ndarray, M: int,
                           prior: Any, npar: int) -> Dict[str, Any]:
    """Calculate proposal using snooker update from DREAMzs algorithm
    
    Parameters:
        X (np.ndarray): Current chain position
        Z (np.ndarray): Common archive of states
        M (int): Number of states in the archive
        prior (Any): Prior distribution object
        npar (int): Number of parameters
    Returns:
        Dict[str, Any]: Dictionary containing the proposed state and extra acceptance ratio term
            - xprop (np.ndarray): Proposed new state
            - r_extra (float): Extra term for acceptance ratio
    """
    randomChains = np.random.choice(np.arange(M), size=3, replace=False)

    z = Z[randomChains[0], :].copy()
    x_z = X - z
    D2 = max(np.sum(x_z * x_z), 1.0e-300)
    projdiff = np.dot((Z[randomChains[1], :] - Z[randomChains[2], :]), x_z / D2)
    gamma_snooker = np.random.uniform(1.2, 2.2) 

    xprop = X + gamma_snooker * projdiff * x_z
    xprop = check_parameters(xprop, prior)

    x_z = xprop - z
    D2prop = max(np.sum(x_z * x_z), 1.0e-300)
    npar12 = (npar - 1) / 2
    r_extra = npar12 * (np.log(D2prop) - np.log(D2))
    return {"xprop": xprop,
            "r_extra": r_extra
    }

################################################################################
# Calculate the Crossover probability matrix CR
################################################################################

def calcCR(nchains: int, CRupdatefreq: int, pCR: np.ndarray) -> np.ndarray:
    """Calculate the Crossover probability matrix CR
    
    Parameters:
        nchains (int): Number of chains
        CRupdatefreq (int): Frequency of CR updates
        pCR (np.ndarray): Probability distribution for CR values
    Returns:
        CR (np.ndarray): Crossover probability matrix with shape (nchains, CRupdatefreq)
    """
    nCR = len(pCR)
    randomv = np.concatenate(([0], np.cumsum(np.random.multinomial(nchains * CRupdatefreq, pCR))))
    cand = np.random.permutation(nchains * CRupdatefreq)
    CR = np.full((nchains * CRupdatefreq,), np.nan)
    for i in range(nCR):
        istart = randomv[i]
        istop = randomv[i + 1]
        candx = cand[istart:istop]
        CR[candx] = (i + 1) / nCR
    return CR.reshape((nchains, CRupdatefreq))

################################################################################
# Update chain i one single step 
################################################################################

def update_chain(ichain: int, X: np.ndarray, logpost_X: np.ndarray, Z: np.ndarray, M: int,
                     CR: np.ndarray, prior: Any, posterior: Any, e: Any,
                     eps: Any, DEpairs: int, zgammaFactor: float,
                     npar: int, nMTM: int, iiter: int,
                     CRupdatefreq: int, pSnooker: float) -> Dict[str, Any]:
    """Update chain ichain one single step using DREAMzs algorithm
    
    Parameters:
        ichain (int): Index of the chain to update
        X (np.ndarray): Current states of all chains
        logpost_X (np.ndarray): Log-posterior values of all chains
        Z (np.ndarray): Common archive of states
        M (int): Number of states in the archive
        CR (np.ndarray): Crossover probability matrix
        prior (Any): Prior distribution object
        posterior (Any): Posterior distribution object
        e (Any): Function returning a draw from a uniform distribution with dimensions npar (noise)
        eps (Any): Function returning a draw from a multinormal distribution with dimensions npar (noise)
        DEpairs (int): Number of pairs selected from the archive for Differential Evolution
        zgammaFactor (float): Scale factor for gamma to control acceptance rate
        npar (int): Number of parameters
        nMTM (int): Number of multiple-try Metropolis proposals
        iiter (int): Current iteration number
        CRupdatefreq (int): Frequency of CR updates
        pSnooker (float): Probability of using a snooker update
    Returns:
        update_data (Dict[str, Any]): Dictionary containing updated chain data
            - X_new (np.ndarray): Updated state of the chain
            - logpost_X_new (np.ndarray): Updated log-posterior value of the chain
            - ARs_new (int): Indicator of acceptance
    """
    X_new = X[ichain, :].copy()
    logpost_X_new = logpost_X[ichain, :].copy()
    ARs_new = 0

    if np.random.rand() > pSnooker:
        # Calculate proposal without using a snooker update
        aux = calc_proposal_noSnooker(X=X[ichain, :], Z=Z, M=M,
                                      CR=CR[ichain, iiter % CRupdatefreq],
                                      prior=prior, posterior=posterior,
                                      e=e, eps=eps,
                                      DEpairs=DEpairs,
                                      zgammaFactor=zgammaFactor,
                                      npar=npar, nMTM=nMTM)
        zprops_post = aux["zprops_post"]
        xprops_post = aux["xprops_post"]
        z = aux["z"]

        alpha = -np.max(xprops_post)
        if np.sum(np.exp(zprops_post + alpha)) / np.sum(np.exp(xprops_post + alpha)) > np.random.rand():
            X_new = z
            logpost_X_new = posterior['density'](z)
            ARs_new += 1
    else:
        # Calculate proposal using a snooker update
        aux = calc_proposal_snooker(X=X[ichain, :], Z=Z, M=M,
                                    prior=prior, npar=npar)
        xprop = aux["xprop"]
        r_extra = aux["r_extra"]

        logpost_xprop = posterior['density'](xprop)
        if not np.isnan(logpost_xprop[2] - logpost_X[ichain, 2]):
            if (logpost_xprop[2] - logpost_X[ichain, 2] + r_extra) > np.log(np.random.rand()):
                X_new = xprop
                logpost_X_new = logpost_xprop
                ARs_new += 1
    return {"X_new": X_new,
            "logpost_X_new": logpost_X_new,
            "ARs_new": ARs_new
    }

################################################################################
# density: prior density
# sampler: prior sampler
# ll_logdensity
# upper_bayes
# lower_bayes
# settings: "nchains"
#           "pSnooker"
#           "DEpairs"
#           "Zupdatefreq"
#           "niterations"
#           "nMTM"
#           "parnames"
#           "adaptation"
# beta: [0,1] "coolness" parameter used for simulated annealing
#       if 0, the modified posterior is simply the prior, if 1, the posterior is not modified
# Zinit: initial values for the archive Z
# X: initial values for the current state X of all nchains chains
# lmessage: if TRUE, show information during runtime
################################################################################

def MT_DREAMzs(density: Any, sampler: Any, ll_logdensity: Any,
                 upper_bayes: np.ndarray, lower_bayes: np.ndarray,
                 settings: Dict[str, Any], beta: float = 1,
                 Zinit: Optional[np.ndarray] = None,
                 X: Optional[np.ndarray] = None,
                 lmessage: bool = True, fixed_init: bool = True) -> Dict[str, Any]:
    """MT-DREAMzs algorithm for Bayesian inference
    
    Parameters:
        density (Any): Prior density function
        sampler (Any): Prior sampler function
        ll_logdensity (Any): Log-likelihood density function
        upper_bayes (np.ndarray): Upper bounds of the parameters
        lower_bayes (np.ndarray): Lower bounds of the parameters
        settings (Dict[str, Any]): Settings for the MT-DREAMzs algorithm
            - nchains (int): Number of chains
            - pSnooker (float): Probability of using a snooker update
            - Zupdatefreq (int): Frequency of updating the archive Z
            - nCR (int): Number of crossover probabilities
            - CRupdatefreq (int): Frequency of updating crossover probabilities
            - checkfreq (int): Frequency of checking convergence and acceptance rate
            - niterations (int): Number of iterations
            - zb (float): Parameter b as per Laloy and Vrugt 2012
            - zbstar (float): Parameter b* as per Laloy and Vrugt 2012
            - nMTM (int): Number of proposals for the Multiple Try Metropolis
            - DEpairs (int): Number of pairs selected from the archive used to create proposals
            - zgammaFactor (float): Scale factor for the size of perturbations to create proposals
            - parnames (List[str]): Names of the parameters
            - expdir (str): Experiment directory for data load
            - parallel (bool): Whether to use parallel processing
            - nproc (int): Number of processors to use
        beta (float): Coolness parameter for simulated annealing (0 to 1)
        Zinit (Optional[np.ndarray]): Initial values for the archive Z
        X (Optional[np.ndarray]): Initial values for the current state X of all nchains chains
        lmessage (bool): Whether to show information during runtime
        fixed_init (bool): Whether to use fixed initial values for X
    Returns:
        results (Dict[str, Any]): Results of the MT-DREAMzs algorithm
            - chains (np.ndarray): Chains of sampled states
            - X (np.ndarray): Final states of all chains
            - Z (np.ndarray): Final archive of states
            - zaccepts (np.ndarray): Number of accepted proposals per chain
            - ARs (np.ndarray): Acceptance rates of all chains
            - Rhats (np.ndarray): Gelman-Rubin R-hat values for all parameters
            - checkfreq (int): Frequency of checking convergence and acceptance rate
            - runtime (float): Total runtime of the algorithm in seconds
    """

    nchains = settings["nchains"]
    pSnooker = settings["pSnooker"]

    Zupdatefreq = settings["Zupdatefreq"]  # archive of states Z is updated after Zupdatefreq steps (aka K)
    nCR = settings["nCR"] # 3 # used for cross-over probabilities
    CRupdatefreq = settings["CRupdatefreq"]  # 10 # update CR every CRupdatefreq
    checkfreq = settings["checkfreq"]  # 100 # check convergence and AR every checkfreq
    niter = settings["niterations"] 
    b = settings["zb"]  # 0.05 # p7 laloy and vrugt 2012
    bstar = settings["zbstar"]  # 10^-6 # p7 laloy and vrugt 2012
    nMTM = settings["nMTM"]  # number of proposals for the Multiple Try Metropolis
    DEpairs = settings["DEpairs"]  # number of pairs selected from the archive used to create proposals
    zgammaFactor = settings["zgammaFactor"]  # scale the size of the perturbations to create proposals

    prior = make_prior(density=density, sampler=sampler,
                       lower_bayes=lower_bayes,
                       upper_bayes=upper_bayes,
                       best=None)
    likelihood = make_likelihood(likelihood=ll_logdensity)
    posterior = make_posterior(prior=prior, likelihood=likelihood, beta=beta)  # when interactive, set beta=1!!
    npar = len(settings['parnames'])  # aka 'd'

    zgammaFactorRef = np.sqrt(npar / (npar - np.sum(lower_bayes == upper_bayes)))
    if zgammaFactor != zgammaFactorRef:
        print(f"WARNING - Gamma factor {zgammaFactor:.2f} differs from recommended value sqrt({npar}/{npar - np.sum(lower_bayes == upper_bayes)}) = {zgammaFactorRef:.2f}")

    niter = int(np.ceil(niter / nchains))
    ncols = int(np.floor(niter // checkfreq))
    zchains = np.full((niter, npar + 3, nchains), np.nan)
    # Column names: settings["parnames"] + "logprior", "loglike", "logpost"

    if X is None:
        X = prior['sampler'](nchains)
        if fixed_init: # could be replace by "central/best" start values
            X = prior['best'].reshape((-1, 1)).repeat(nchains, axis=1).T
    logpost_X = posterior['density'](X)
    logpost_X = np.column_stack((logpost_X['prior'], logpost_X['likelihood'], logpost_X['posterior']))

    zchains[0, :, :] = np.column_stack((X, logpost_X)).T #(niter, npar+3, nchains)

    if Zinit is None:
        M = 10 * max(nchains, npar)  # N << M0 in laloy and vrugt 2012 #baytools: 10*npar
        Zinit = prior['sampler'](M)
    else:   
        M = Zinit.shape[0]
    Z = np.full((M + (niter // Zupdatefreq) * nchains, npar), np.nan)
    Z[:M, :] = Zinit

    Delta = np.zeros((nCR,))  # cross-over distances
    pCR = np.full((nCR,), 1 / nCR)  # cross-over probabilities
    lCR = np.zeros((nCR,))  # distance travelled through space for each discrete CR
    CR = calcCR(nchains, CRupdatefreq, pCR)

    ARs = np.ones((nchains, ncols))  # 1 since first sample is accepted
    Rhats = np.zeros((npar + 1, ncols))  # potential scale reduction factor

    def e() -> np.ndarray:
        """Internal function to add small noise to proposals"""
        return np.random.uniform(-b, b, size=npar)
    
    def eps() -> np.ndarray:
        """Internal function to add small noise to proposals"""
        x = np.random.normal(0, bstar, size=npar)
        return x * np.array([upper_bayes[i] - lower_bayes[i] for i in range(npar)])  # seems fairer to scale

    lconverged = False  # Flag so that single message can be printed if chains are converged

    time_start = time.time()
    
    for iiter in range(1, niter):
        Xold = X.copy()
        
        if not settings["parallel"]:
            aux = [update_chain(ichain=i, X=X, logpost_X=logpost_X, Z=Z, M=M,
                                CR=CR, prior=prior, posterior=posterior,
                                e=e, eps=eps, DEpairs=DEpairs,
                                zgammaFactor=zgammaFactor,
                                npar=npar, nMTM=nMTM, iiter=iiter,
                                CRupdatefreq=CRupdatefreq,
                                pSnooker=pSnooker) for i in range(nchains)]
        else:
            args = [(i, X, logpost_X, Z, M, CR, prior, posterior, e, eps,
                     DEpairs, zgammaFactor, npar, nMTM, iiter, CRupdatefreq, pSnooker)
                    for i in range(nchains)]
            with Pool(processes=settings["nproc"]) as pool:
                aux = pool.map(lambda p: update_chain(*p), args)
        X = np.array([t['X_new'] for t in aux])
        
        #make sure to get logpost_X in correct format
        for i in range(len(aux)):
            logpost_X_new = aux[i]['logpost_X_new']
            if isinstance(logpost_X_new, dict):
                logpost_X[i, 0] = logpost_X_new['prior']
                logpost_X[i, 1] = logpost_X_new['likelihood']
                logpost_X[i, 2] = logpost_X_new['posterior']
            else:
                logpost_X[i, 0] = logpost_X_new[0]
                logpost_X[i, 1] = logpost_X_new[1]
                logpost_X[i, 2] = logpost_X_new[2]

        col_idx = int(np.ceil(iiter / checkfreq)) - 1
        if col_idx >= ncols:
            col_idx = ncols - 1
        ARs[:, col_idx] += np.array([t['ARs_new'] for t in aux])

        zchains[iiter, :, :] = np.column_stack([X, logpost_X]).T

        if (iiter + 1) % Zupdatefreq == 0:
            Z[M:M + nchains, :] = X
            M += nchains
        
        if iiter + 1 < settings["adaptation"]:
            # Tuning of the cross-over probabilities
            # Could decrease autocorrelation according to Vrugt et al 2008
            sdX = np.nanstd(X[:, :npar], axis=0)
            if np.any(sdX == 0):
                print(f"During iter {iiter}, sdX contains zeros: {np.round(sdX, 2)}")

            with np.errstate(divide='ignore', invalid='ignore'):
                diff = (Xold[:, :npar] - X[:, :npar]) / sdX
            diff[np.isnan(diff)] = 0.0
            Delta += np.sum(diff**2, axis=0)

            if (iiter + 1) % CRupdatefreq == 0:
                # Update pCR (function AdaptpCR of BayesianTools)
                if np.any(Delta > 0):  # since /sum(Delta)
                    CR = CR.flatten()  # change CR to single vector
                    lCR_old = lCR.copy()
                    lCR = np.full((nCR,), np.nan)
                    for k in range(nCR):
                        lCR[k] = lCR_old[k] + len(np.where(CR == (k + 1) / nCR)[0])
                    pCR = iiter * nchains * (Delta / lCR) / np.sum(Delta)  # p275 in Vrugt et al 2009
                    pCR[np.where(np.isnan(pCR))] = 1 / nCR  # can occur in beginning when lCR is 0?
                    pCR /= np.sum(pCR)  # normalisation

        if (iiter + 1) % CRupdatefreq == 0:
            CR = calcCR(nchains, CRupdatefreq, pCR)
        if lmessage and ((iiter + 1) % 10 == 0 or iiter + 1 == niter):
            print("\r","Running MT-DREAMzs, iteration",iiter*nchains,"of",niter*nchains,". Current logp",logpost_X[:,2])

        if (iiter + 1) % checkfreq == 0:
            Rhats[:, int(np.ceil((iiter + 1) / checkfreq)) - 1] = check_convergence(zchains[:iiter+1, :, :], ipars=list(range(npar)) + [npar+2])
            if not lconverged and (max(Rhats[:, int(np.ceil((iiter + 1) / checkfreq)) - 1]) < 1.2):
                lconverged = True
                print(f"\nConverged after {(iiter + 1) * nchains} iterations \n")

    time_stop = time.time()
    runtime = round(time_stop - time_start, 3)
    if lmessage:
        print(f"\n MT-DREAMzs terminated after {runtime} seconds \n")

    zaccepts = np.nansum(ARs, axis=1) * 100.0 / niter
    ARs = 100.0 * ARs / checkfreq
    if lmessage:
        for i in range(nchains):
            print(f"Acceptance rate for chain {i + 1} is {zaccepts[i]:2.2f}% ")

    ARs = ARs[:, :int(np.ceil(niter / checkfreq))]
    Rhats = Rhats[:, :int(np.ceil(niter / checkfreq))]

    return {
        "chains": zchains,
        "X": X[:, :npar],
        "Z": Z,
        "zaccepts": zaccepts,
        "ARs": ARs,
        "Rhats": Rhats,
        "checkfreq": checkfreq,
        "runtime": runtime
    }



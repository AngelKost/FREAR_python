# -----------------------------------------------------------------------------
# source('test_case/expdir/main.R')
# -----------------------------------------------------------------------------
library(ggmap)
# Note: 
# To run this test, open an R console in the FREAR base directory (~/SomeDirectory/FREAR) and type:
# > source('test_case/expdir/main.R')
#

# Input -----------------------------------------------------------------------
SEED <- 42
set.seed(SEED)

lsave <- F # FALSE or TRUE
lbayes <- TRUE # FALSE or TRUE
lcost <- TRUE # FALSE or TRUE
laccFOR <- TRUE  # FALSE or TRUE
lmaxPSR <- TRUE # FALSE or TRUE
srcdir <- '/home/angelkos/SHAD_project/FREAR-master/R_scripts' # can be an absolute path too

# Load functions
source(paste(srcdir, 'settings.R', sep = '/'))
load_scripts(dir = srcdir)

# Load default settings
settings <- getDefaultSettings(srcdir)


# Define mandatory settings
settings$experiment <- 'FREAR_syntheticTestCase'
settings$expdir <- '/home/angelkos/SHAD_project/FREAR_python/test_case/expdir'
settings$datadir <- '/home/angelkos/SHAD_project/FREAR_python/test_case/data'
settings$subexp <- 'subexp1'
settings$members <- NULL
settings$domain <- list("lonmin"=-150,"latmin"=40,"lonmax"=-50,"latmax"=70,"dx"=.25,"dy"=.25)
settings$obsunitfact <- 1
settings$Qmin <- 10^5 #10^7
settings$Qmax <- 10^15 #10^13
if(!lcost) {
  settings$times <- expression(  seq(as.POSIXct("2019-08-01"),as.POSIXct("2019-08-05"),by=3*3600) )
} else {
  # In the cost function approach, the length of settings$times should be limited as it results 
  # in a new source parameter when using the default cost source parameterisation
  settings$times <- expression(  seq(as.POSIXct("2019-08-01"),as.POSIXct("2019-08-05"),by=24*3600) )
}

# Define optional settings
settings$obsscalingfact <- 10^-10
settings$trueValues <- c(-117, 59.9, 9, tstart2rstart(settings$times, tstart = as.POSIXct("2019-08-02 12:00:00")),
                                        tstop2rstop(settings$times, tstart = as.POSIXct("2019-08-02 12:00:00"), tstop = as.POSIXct("2019-08-02 15:00:00")))
settings$likelihood <- "Yee2017log"
settings$outbasedir <- "~/work/output" 
settings$nproc <- 10
settings$lparallelread <- TRUE
settings$lparallelcalc <- TRUE
settings$niterations <- 10000 # 100000
settings$sourcemodelbayes <- "srs2AC_bayes_rectRelease.R"
settings$sourcemodelcost <- "srs2AC_cost_nsegmentsRel.R" 
settings$srsfact <- 1 / (exp(0.5*(log(settings$Qmin)+log(settings$Qmax))) / (settings$obsunitfact * settings$obsscalingfact))
settings$lusebaseRonly <- TRUE # if FALSE, use new plotting tools
settings$lusemeteogrid <- TRUE



# Initialisation --------------------------------------------------------------

# Check, add and write out settings
settings <- checkSettings(settings)

# Read data --------------------------------------------------------------------

source(paste(srcdir, 'maincomp_readdata.R', sep = '/'))

# Now srs_raw, srs_spread_raw, samples, outputfreq are available

saveRDS(srs_raw, file = file.path(settings$expdir, 'cmp/srs_raw.rds'))
saveRDS(srs_spread_raw, file = file.path(settings$expdir, 'cmp/srs_spread_raw.rds'))
cat("Outputfreq: ")
print(outputfreq)


source(paste(srcdir, 'maincomp_srr.R', sep = '/'))

# Now srr_data is available

saveRDS(srs, file = file.path(settings$expdir, 'cmp/srs.rds'))
saveRDS(obs, file = file.path(settings$expdir, 'cmp/obs.rds'))
saveRDS(obs_error, file = file.path(settings$expdir, 'cmp/obs_error.rds'))
saveRDS(MDC, file = file.path(settings$expdir, 'cmp/MDC.rds'))
cat("Qfact: ")
print(Qfact)
cat("lower_bayes: ")
print(lower_bayes)
cat("upper_bayes: ")
print(upper_bayes)
cat("lower_cost: ")
print(lower_cost)
cat("upper_cost: ")
print(upper_cost)
cat("par_init: ") 
print(par_init)

if(lbayes) {
  # Bayesian inverse modelling -------------------------------------------------
  source(paste(srcdir, 'maincomp_prepBayes.R', sep = '/'))
  out <- MT_DREAMzs(density, sampler, ll_logdensity, upper_bayes, lower_bayes, settings, beta = 1)

  saveRDS(out$chains, file = file.path(settings$expdir, 'cmp/chains.rds'))
  saveRDS(out$X, file = file.path(settings$expdir, 'cmp/X.rds'))
  saveRDS(out$Z, file = file.path(settings$expdir, 'cmp/Z.rds'))
  cat("zaccepts: ")
  print(out$zaccepts)
  saveRDS(out$ARs, file = file.path(settings$expdir, 'cmp/ARs.rds'))
  saveRDS(out$Rhats, file = file.path(settings$expdir, 'cmp/Rhats.rds'))
  cat("checkfreq: ")
  print(out$checkfreq)
  cat("runtime: ")
  print(out$runtime)

  mcmc <- analyseMCMC(out = out, settings = settings, lpost_mode = TRUE)
  saveRDS(mcmc$chainburned, file = file.path(settings$expdir, 'cmp/chainburned.rds'))
  saveRDS(mcmc$zchainSummary, file = file.path(settings$expdir, 'cmp/zchainSummary.rds'))
  cat("Rs: ")
  print(mcmc$Rs)
  saveRDS(mcmc$probloc, file = file.path(settings$expdir, 'cmp/probloc.rds'))
  cat("post_median: ")
  print(mcmc$post_median)
  cat("post_mode: ")
  print(mcmc$post_mode)

  write_bayes(out = out, mcmc = mcmc, settings = settings, misc = misc, obs = obs,
              obs_error = obs_error, lsave = lsave)

}

if(lcost) {
  # Cost inverse modelling -----------------------------------------------------
  mod_error_cost <- MDC + obs_error
  optsed <- calc_costs(srs = srs, obs = obs, Qfact = Qfact, sigma = mod_error_cost, lower_cost = lower_cost,
                       upper_cost = upper_cost, par_init = par_init, nproc = ifelse(settings$lparallelcalc, settings$nproc, 1))
  saveRDS(optsed$cost, file = file.path(settings$expdir, 'cmp/optsed_costs.rds'))
  saveRDS(optsed$Qs, file = file.path(settings$expdir, 'cmp/optsed_Qs.rds'))
  saveRDS(optsed$accQ, file = file.path(settings$expdir, 'cmp/optsed_accQ.rds'))
  write_cost(optsed = optsed, mod_error_cost = mod_error_cost, settings = settings, lsave = lsave)
}

if(laccFOR) {
  source(paste(srcdir, "accFOR.R", sep = "/"))
  accFOR <- calc_accFOR(srs = srs, obs = obs)
  saveRDS(accFOR, file = file.path(settings$expdir, 'cmp/accFOR.rds'))
  write_accFOR(accFOR = accFOR, ndet = sum(obs>0), settings = settings, lsave = lsave)
}

if(lmaxPSR) {
  source(paste(srcdir, "maxPSR.R", sep = "/"))
  maxPSR <- calc_maxPSR(srs = srs, obs = obs)
  saveRDS(maxPSR, file = file.path(settings$expdir, 'cmp/maxPSR.rds'))
  write_maxPSR(maxPSR = maxPSR, settings = settings, lsave = lsave)
}
**Old pipeline:**
    - maincomp_readdata: loads inital data and settings
    - maincomp_srr: source-receptor relationship calculations
        - maincomp_prepbayes: calculates all necessary bayesian model parameters
        - MT_DREAMzs: MT-DREAMzs algorithm
        - analyze_mcmc: plots results of bayesian model

        - calc_cost
        
        - calc_accFOR

        - calc_maxPSR

**Updated pipeline:**
    - maincomp_readdata: loads inital data and settings
    - maincomp_srr_upd: source-receptor relationship calculations
    - maincomp_calcErr: calculates likelihood parameters, calculates error
    - *Last possible point of change for true values and fixed time/location in bayes model setup*
    - maincomp_model_setup: cost/bayes model setup

        - maincomp_prepbayes_upd: prepares bayes model setup
        - MT_DREAMzs_parallel: MT-DREAMzs algorithm with parallel computation
        - analyze_mcmc: plots results of bayesian model

        - calc_cost
        
        - calc_accFOR

        - calc_maxPSR
    - *Has checkpoints_upd to use for storing/loading checkpoints in pipeline*
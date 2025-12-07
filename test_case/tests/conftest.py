import sys
sys.path.append('/home/angelkos/SHAD_project/')

import pytest
import numpy as np
import pandas as pd
import subprocess
import tempfile
import os

from datetime import datetime, timedelta, timezone

from frear.settings import get_default_settings, check_settings
from frear.domain import domain_add_nxny
from frear.tools import tstart2rstart, tstop2rstop
from frear.srs2AC_bayes_rectRelease import srs2AC_bayes_rectRelease
from frear.srs2AC_cost_nsegmentsRel import srs2AC_cost_nsegmentsRel


def load_rds_dataframe_via_r(rds_file):
    """Load RDS file containing a data.frame using R and return as pandas DataFrame"""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Use R to read RDS and save as CSV
        r_code = f"""
data <- readRDS('{rds_file}')
write.csv(data, file='{tmp_path}', row.names=FALSE)
"""
        subprocess.run(['R', '--vanilla', '--quiet', '--slave', '-e', r_code], 
                      check=True, capture_output=True)
        
        # Read CSV as pandas DataFrame
        df = pd.read_csv(tmp_path)
        return df
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def load_rds_via_r(rds_file):
    """Load RDS file using R and return as numpy array"""
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Use R to read RDS and save as binary
        r_code = f"""
data <- readRDS('{rds_file}')
if (is.list(data) && length(data) == 1) {{ 
    arr <- data[[1]] 
}} else {{ 
    arr <- data 
}}
dim_info <- dim(arr)
if (is.null(dim_info)) {{
    dim_info <- c(length(arr))
}}
writeBin(as.numeric(arr), con='{tmp_path}')
writeBin(as.integer(dim_info), con='{tmp_path}', size=4)
"""
        subprocess.run(['R', '--vanilla', '--quiet', '--slave', '-e', r_code], 
                      check=True, capture_output=True)
        
        # Read back the binary file
        with open(tmp_path, 'rb') as f:
            # Read data as float64
            # Note: this is a simplified approach; for production use structured format
            data_bytes = f.read()
        
        # For now, use a simpler approach: use R to save as .csv temporarily
        subprocess.run(['R', '--vanilla', '--quiet', '--slave', '-e', f"""
data <- readRDS('{rds_file}')
if (is.list(data) && length(data) == 1) {{ 
    arr <- data[[1]] 
}} else {{ 
    arr <- data 
}}
write.csv(as.numeric(arr), file='{tmp_path}.csv', row.names=FALSE)
"""], check=True, capture_output=True)
        
        arr = np.loadtxt(f'{tmp_path}.csv', delimiter=',', skiprows=1)
        
        # Get original shape
        result = subprocess.run(['R', '--vanilla', '--quiet', '--slave', '-e', f"""
data <- readRDS('{rds_file}')
if (is.list(data) && length(data) == 1) {{ 
    arr <- data[[1]] 
}} else {{ 
    arr <- data 
}}
cat(paste(dim(arr), collapse=','))
"""], capture_output=True, text=True, check=True)
        
        dims_str = result.stdout.strip()
        if dims_str:
            dims = tuple(int(x) for x in dims_str.split(','))
            # R flattens in column-major order, NumPy reshapes in row-major
            # So we need to reverse dims, reshape, then transpose back
            arr = arr.reshape(dims[::-1]).T
        
        return arr
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(f'{tmp_path}.csv'):
            os.remove(f'{tmp_path}.csv')

@pytest.fixture
def fake_settings():
    """Fake settings for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    settings = get_default_settings()
    settings['experiment'] = 'FREAR_syntheticTestCase'
    settings['expdir'] = root_dir + 'test_case/expdir'
    settings['datadir'] = root_dir + 'test_case/data'
    settings['subexp'] = 'subexp1'
    settings['members'] = None
    settings['domain'] = {"lonmin": -150, "latmin": 40, "lonmax": -50, "latmax": 70, "dx": 0.25, "dy": 0.25}
    settings['domain'] = domain_add_nxny(settings['domain'])
    settings['obsunitfact'] = 1
    settings['Qmin'] = 10**5  # 10**7
    settings['Qmax'] = 10**15  # 10**13

    settings['times'] = [datetime(2019, 8, 1, tzinfo=timezone.utc)]
    time = datetime(2019, 8, 1, tzinfo=timezone.utc)
    while time < datetime(2019, 8, 5, tzinfo=timezone.utc):
        time += timedelta(hours=24)
        settings['times'].append(time)

    # Define optional settings
    settings['obsscalingfact'] = 10**-10
    settings['trueValues'] = [-117, 59.9, 9, tstart2rstart(settings['times'], tstart=datetime(2019, 8, 2, 12, 0, 0, tzinfo=timezone.utc)),
                              tstop2rstop(settings['times'], tstart=datetime(2019, 8, 2, 12, 0, 0, tzinfo=timezone.utc), tstop=datetime(2019, 8, 2, 15, 0, 0, tzinfo=timezone.utc))]
    settings['likelihood'] = "Yee2017log"
    settings['outbasedir'] = root_dir + "outputs"
    settings['nproc'] = 10
    settings['niterations'] = 10000  # 100000
    settings['sourcemodelbayes'] = "rectRelease"
    settings['sourcemodel_bayes_exec'] = srs2AC_bayes_rectRelease
    settings['sourcemodelcost'] = "nsegmentsRel"
    settings['sourcemodelcost_exec'] = srs2AC_cost_nsegmentsRel
    settings['srsfact'] = 1 / (np.exp(0.5 * (np.log(settings['Qmin']) + np.log(settings['Qmax']))) / (settings['obsunitfact'] * settings['obsscalingfact']))
    
    check_settings(settings)
    
    return settings

@pytest.fixture
def fake_samples():
    """Fake samples for testing"""
    expdir = '/home/angelkos/SHAD_project/FREAR_python/test_case/expdir'
    return pd.read_csv(expdir + '/cmp/samples.csv')

@pytest.fixture
def fake_srs_raw():
    """Fake srs_raw for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    srs_raw = np.load(root_dir + 'test_case/expdir/cmp/srs_raw.npz')
    return srs_raw

@pytest.fixture
def fakeR_srs_raw():
    """Fake R srs_raw for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    return load_rds_via_r(root_dir + 'test_case/expdir/cmp/srs_raw.rds')

@pytest.fixture
def fake_srs_spread_raw():
    """Fake srs_spread_raw for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    srs_spread_raw = np.load(root_dir + 'test_case/expdir/cmp/srs_spread_raw.npz')
    return srs_spread_raw

@pytest.fixture
def fakeR_srs_spread_raw():
    """Fake R srs_spread_raw for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    return load_rds_via_r(root_dir + 'test_case/expdir/cmp/srs_spread_raw.rds')

@pytest.fixture
def fake_outputfreq():
    """Fake outputfreq for testing"""
    return 10800

@pytest.fixture
def fake_srs():
    """Fake srs for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    srs = np.load(root_dir + 'test_case/expdir/cmp/srs.npy')
    return srs

@pytest.fixture
def fakeR_srs():
    """Fake R srs for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    return load_rds_via_r(root_dir + 'test_case/expdir/cmp/srs.rds')

@pytest.fixture
def fake_obs():
    """Fake obs for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    obs = np.load(root_dir + 'test_case/expdir/cmp/obs.npy')
    return obs

@pytest.fixture
def fakeR_obs():
    """Fake R obs for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    return load_rds_via_r(root_dir + 'test_case/expdir/cmp/obs.rds')

@pytest.fixture
def fake_obs_error():
    """Fake obs_error for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    obs_error = np.load(root_dir + 'test_case/expdir/cmp/obs_error.npy')
    return obs_error

@pytest.fixture
def fakeR_obs_error():
    """Fake R obs_error for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    return load_rds_via_r(root_dir + 'test_case/expdir/cmp/obs_error.rds')

@pytest.fixture
def fake_MDC():
    """Fake MDC for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    MDC = np.load(root_dir + 'test_case/expdir/cmp/MDC.npy')
    return MDC

@pytest.fixture
def fakeR_MDC():
    """Fake R MDC for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    return load_rds_via_r(root_dir + 'test_case/expdir/cmp/MDC.rds')

@pytest.fixture
def fake_Qfact():
    """Fake Qfact for testing"""
    return 1e-10

@pytest.fixture
def fakeR_Qfact():
    """Fake R Qfact for testing"""
    return 1e-10

@pytest.fixture
def fake_lower_bayes():
    """Fake lower_bayes for testing"""
    return np.array([-150, 40, np.float64(5.0), 0.0, 0.1])

@pytest.fixture
def fakeR_lower_bayes():
    """Fake R lower_bayes for testing"""
    return np.array([-150, 40, np.float64(5.0), 0.0, 0.1])

@pytest.fixture
def fake_upper_bayes():
    """Fake upper_bayes for testing"""
    return np.array([-50, 70, np.float64(15.0), 0.95, 1.0])

@pytest.fixture
def fakeR_upper_bayes():
    """Fake R upper_bayes for testing"""
    return np.array([-50, 70, np.float64(15.0), 0.95, 1.0])

@pytest.fixture
def fake_lower_cost():
    """Fake lower_cost for testing"""
    return np.array([1e-5, 1e-5, 1e-5, 1e-5])

@pytest.fixture
def fakeR_lower_cost():
    """Fake R lower_cost for testing"""
    return np.array([1e-5, 1e-5, 1e-5, 1e-5])

@pytest.fixture
def fake_upper_cost():
    """Fake upper_cost for testing"""
    return np.array([1e5, 1e5, 1e5, 1e5])

@pytest.fixture
def fakeR_upper_cost():
    """Fake R upper_cost for testing"""
    return np.array([1e5, 1e5, 1e5, 1e5])

@pytest.fixture
def fake_par_init():
    """Fake par_init for testing"""
    return np.array([1., 1., 1., 1.])

@pytest.fixture
def fakeR_par_init():
    """Fake R par_init for testing"""
    return np.array([1., 1., 1., 1.])

@pytest.fixture
def fake_chains_init():
    """Fake chains_init for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    chains_init = np.load(root_dir + 'test_case/expdir/cmp/chains_init.npy')
    return chains_init

@pytest.fixture
def fakeR_chains_init():
    """Fake R chains_init for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    return load_rds_via_r(root_dir + 'test_case/expdir/cmp/chains_init.rds')

@pytest.fixture
def fake_chains():
    """Fake chains for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    chains = np.load(root_dir + 'test_case/expdir/cmp/chains.npy')
    return chains

@pytest.fixture
def fakeR_chains():
    """Fake R chains for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    return load_rds_via_r(root_dir + 'test_case/expdir/cmp/chains.rds')

@pytest.fixture
def fake_X():
    """Fake X for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    X = np.load(root_dir + 'test_case/expdir/cmp/X.npy')
    return X

@pytest.fixture
def fakeR_X():
    """Fake R X for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    return load_rds_via_r(root_dir + 'test_case/expdir/cmp/X.rds')

@pytest.fixture
def fake_Z():
    """Fake Z for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    Z = np.load(root_dir + 'test_case/expdir/cmp/Z.npy')
    return Z

@pytest.fixture
def fakeR_Z():
    """Fake R Z for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    return load_rds_via_r(root_dir + 'test_case/expdir/cmp/Z.rds')

@pytest.fixture
def fake_zaccepts():
    """Fake zaccepts for testing"""
    return np.array([27.05458908, 28.79424115, 26.45470906])

@pytest.fixture
def fakeR_zaccepts():
    """Fake R zaccepts for testing"""
    return np.array([29.93401, 31.19376, 31.64367])

@pytest.fixture
def fake_ARs():
    """Fake ARs for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    ARs = np.load(root_dir + 'test_case/expdir/cmp/ARs.npy')
    return ARs

@pytest.fixture
def fakeR_ARs():
    """Fake R ARs for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    return load_rds_via_r(root_dir + 'test_case/expdir/cmp/ARs.rds')

@pytest.fixture
def fake_Rhats():
    """Fake Rhats for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    Rhats = np.load(root_dir + 'test_case/expdir/cmp/Rhats.npy')
    return Rhats

@pytest.fixture
def fakeR_Rhats():
    """Fake R Rhats for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    return load_rds_via_r(root_dir + 'test_case/expdir/cmp/Rhats.rds')

@pytest.fixture
def fake_checkfreq():
    """Fake checkfreq for testing"""
    return 100

@pytest.fixture
def fakeR_checkfreq():
    """Fake R checkfreq for testing"""
    return 100

@pytest.fixture
def fake_runtime():
    """Fake runtime for testing"""
    return 235.37

@pytest.fixture
def fakeR_runtime():
    """Fake R runtime for testing"""
    return 47.664

@pytest.fixture
def fake_chainburned():
    """Fake chainburned for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    chainburned = np.load(root_dir + 'test_case/expdir/cmp/chainburned.npy')
    return chainburned

@pytest.fixture
def fakeR_chainburned():
    """Fake R chainburned for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    return load_rds_via_r(root_dir + 'test_case/expdir/cmp/chainburned.rds')

@pytest.fixture
def fake_zchainsummary():
    """Fake zchainsummary for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    zchainsummary = np.load(root_dir + 'test_case/expdir/cmp/zchainSummary.npy', allow_pickle=True)
    return zchainsummary

@pytest.fixture
def fakeR_zchainsummary():
    """Fake R zchainsummary for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    df = load_rds_dataframe_via_r(root_dir + 'test_case/expdir/cmp/zchainSummary.rds')
    
    result = np.empty((len(df), 5), dtype=object)
    result[:, 0] = df['lon'].values
    result[:, 1] = df['lat'].values
    result[:, 2] = df['log10_Q'].values
    
    result[:, 3] = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc) for t in df['rstart'].values]
    result[:, 4] = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc) for t in df['rstop'].values]
    
    return result

@pytest.fixture
def fake_Rs():
    """Fake Rs for testing"""
    return np.array([1.01357782, 1.0076424,  1.00655972, 1.03531157, 1.00324363, 1.00081679])

@pytest.fixture
def fakeR_Rs():
    """Fake R Rs for testing"""
    return np.array([1.021737, 1.027174, 1.020589, 1.013665, 1.006030, 1.005979])

@pytest.fixture
def fake_probloc():
    """Fake probloc for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    probloc = np.load(root_dir + 'test_case/expdir/cmp/probloc.npy')
    return probloc

@pytest.fixture
def fakeR_probloc():
    """Fake R probloc for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    return load_rds_via_r(root_dir + 'test_case/expdir/cmp/probloc.rds')

@pytest.fixture
def fake_post_median():
    """Fake post_median for testing"""
    return np.array([-116.99282377,   58.6494314,     8.78897933,    0.15395717,    0.37150542])

@pytest.fixture
def fakeR_post_median():
    """Fake R post_median for testing"""
    return np.array([-117.2510021,   58.3504508,    8.7697059,    0.1526500,    0.3560357])

@pytest.fixture
def fake_post_mode():
    """Fake post_mode for testing"""
    return np.array([-100.,    60.,     9.,     0.2,    0.2])

@pytest.fixture
def fakeR_post_mode():
    """Fake R post_mode for testing"""
    return np.array([-100.0,   60.0,    9.0,    0.2,    0.2])

@pytest.fixture
def fake_optsed_costs():
    """Fake optsed_costs for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    optsed_costs = np.load(root_dir + 'test_case/expdir/cmp/optsed_costs.npy')
    return optsed_costs

@pytest.fixture
def fakeR_optsed_costs():
    """Fake R optsed_costs for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    return load_rds_via_r(root_dir + 'test_case/expdir/cmp/optsed_costs.rds')

@pytest.fixture
def fake_optsed_Qs():
    """Fake optsed_Qs for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    optsed_Qs = np.load(root_dir + 'test_case/expdir/cmp/optsed_Qs.npy')
    return optsed_Qs

@pytest.fixture
def fakeR_optsed_Qs():
    """Fake R optsed_Qs for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    return load_rds_via_r(root_dir + 'test_case/expdir/cmp/optsed_Qs.rds')

@pytest.fixture
def fake_optsed_accQ():
    """Fake optsed_accQ for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    optsed_accQ = np.load(root_dir + 'test_case/expdir/cmp/optsed_accQ.npy')
    return optsed_accQ

@pytest.fixture
def fakeR_optsed_accQ():
    """Fake R optsed_accQ for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    return load_rds_via_r(root_dir + 'test_case/expdir/cmp/optsed_accQ.rds')

@pytest.fixture
def fake_accFOR():
    """Fake accFOR for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    accFOR = np.load(root_dir + 'test_case/expdir/cmp/accFOR.npy')
    return accFOR

@pytest.fixture
def fakeR_accFOR():
    """Fake R accFOR for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    return load_rds_via_r(root_dir + 'test_case/expdir/cmp/accFOR.rds')

@pytest.fixture
def fake_maxPSR():
    """Fake maxPSR for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    maxPSR = np.load(root_dir + 'test_case/expdir/cmp/maxPSR.npy')
    return maxPSR

@pytest.fixture
def fakeR_maxPSR():
    """Fake R maxPSR for testing"""
    root_dir = '/home/angelkos/SHAD_project/FREAR_python/'
    return load_rds_via_r(root_dir + 'test_case/expdir/cmp/maxPSR.rds')
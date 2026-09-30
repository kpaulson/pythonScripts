import os
import sys
import platform
import tempfile
import shutil
import cdflib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Look up one level to import config file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import config

# High speed caching matching resonance_io
CUSTOM_TMP = config.get_tmpMirror()
if platform.system() == 'Windows':
    ACTIVE_TMP = None 
    io_message = "(via Local Windows Temp)"
else:
    ACTIVE_TMP = CUSTOM_TMP
    os.makedirs(ACTIVE_TMP, exist_ok=True)
    io_message = "(via Linux RAM disk)"

from contextlib import contextmanager

@contextmanager
def smart_cdf(filepath):
    """Safely copies a CDF to a local temp dir before reading to bypass network drive latency."""
    fd, temp_path = tempfile.mkstemp(suffix='.cdf', dir=ACTIVE_TMP)
    os.close(fd) 
    try:
        shutil.copyfile(filepath, temp_path)
        with cdflib.CDF(temp_path) as cdf:
            yield cdf
    finally:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass

def load_vdc_potential(vdc_filepath):
    """
    Extracts high-cadence spacecraft DC potentials and computes V_avg.
    """
    print(f"[+] Loading DC Potentials {io_message}...")
    with smart_cdf(vdc_filepath) as cdf:
        epoch = cdf.varget('Epoch') 
        unix_time = cdflib.cdfepoch.unixtime(epoch)
        
        v1 = cdf.varget('psp_fld_l2_dfb_wf_V1dc')
        v2 = cdf.varget('psp_fld_l2_dfb_wf_V2dc')
        v3 = cdf.varget('psp_fld_l2_dfb_wf_V3dc')
        v4 = cdf.varget('psp_fld_l2_dfb_wf_V4dc')
        
    for arr in [v1, v2, v3, v4]:
        arr[arr < -1e30] = np.nan
        
    v_avg = (v1 + v2 + v3 + v4) / 4.0
    return unix_time, v_avg

def sync_and_fit_calibration(v_time, v_avg, ne_time, ne_data, te_data, r_au_data):
    """
    Interpolates all relevant low-cadence parameters onto the QTN timeline,
    filters them, and extracts the calibration fit parameters.
    
    Fit equation: Vs = A * ln(ne * R^2 * Te^0.5) + B
    """
    # 1. Sync V_avg to the QTN density timeline
    f_interp_v = interp1d(v_time, v_avg, bounds_error=False, fill_value=np.nan)
    v_avg_synced = f_interp_v(ne_time)
    
    # 2. Convert R from AU to meters
    r_m = r_au_data * 14959787069100.0 # AU to meters (Matches your Jython conversion)
    
    # 3. Calculate log parameter: ln( ne * R^2 * sqrt(Te) )
    with np.errstate(divide='ignore', invalid='ignore'):
        log_stuff = np.log(ne_data * (r_m ** 2) * np.sqrt(te_data))
        
    # 4. Clean out invalid numbers
    valid_mask = np.isfinite(log_stuff) & np.isfinite(v_avg_synced)
    
    if np.sum(valid_mask) < 10: # Minimum points to compute a stable fit
        print("[-] Warning: Insufficient overlapping data points for linear fit.")
        return np.nan, np.nan
        
    # 5. Fit using numpy polyfit (deg=1 is linear)
    slope, intercept = np.polyfit(log_stuff[valid_mask], v_avg_synced[valid_mask], 1)
    
    return slope, intercept

def backsolve_high_cadence_density(v_time, v_avg, slope, intercept, te_time, te_data, r_time, r_au_data):
    """
    Interpolates temperature and distance up to the high-cadence potential timeline,
    then evaluates high-cadence density (n_scpot).
    
    Conversion formula: n_scpot = exp((V_avg - B) / A) / (R^2 * sqrt(Te))
    """
    print("[+] Backsolving for high-cadence spacecraft potential density...")
    
    # Sync low-cadence metrics up to high-cadence potential timeline
    f_interp_te = interp1d(te_time, te_data, bounds_error=False, fill_value=np.nan)
    te_high = f_interp_te(v_time)
    
    f_interp_r = interp1d(r_time, r_au_data, bounds_error=False, fill_value=np.nan)
    r_au_high = f_interp_r(v_time)
    r_m_high = r_au_high * 14959787069100.0
    
    # Apply inverted fit relation
    with np.errstate(divide='ignore', invalid='ignore'):
        n_scpot = np.exp((v_avg - intercept) / slope) / ((r_m_high ** 2) * np.sqrt(te_high))
        
    return n_scpot
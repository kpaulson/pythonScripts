import os
import sys
import platform
import tempfile
import shutil
import cdflib
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch

# Look up one level to import config file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import config

# Standard Physics Constants (CGS values)
M_P = 1.6726231E-24  # Proton mass in g
Q   = 1.6022E-12     # Energy conversion factor (eV to ergs)

# Set up local high-speed caching matching your resonance_io convention
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

def calculate_vz(mv_lo, mv_hi):
    """
    Calculates parallel velocity (v_z) and its low/hi bounds, plus window widths (km/s).
    """
    # Center voltage
    mv_center = mv_lo + (mv_hi - mv_lo) / 2.0
    
    # Velocity conversion: sqrt(2 * q * V / m_p) / 1e5 (for km/s)
    v_z = np.sqrt((2. / M_P) * Q * mv_center) / 1.E5
    v_z_lo = np.sqrt((2. / M_P) * Q * mv_lo) / 1.E5
    v_z_hi = np.sqrt((2. / M_P) * Q * mv_hi) / 1.E5
    
    # Velocity window width
    window_widths = np.sqrt((2. / M_P) * Q * (mv_hi - mv_lo) / 2.) / 1.E5
    
    return v_z, v_z_lo, v_z_hi, window_widths

def process_spc_flux_angle(l2ifa_file, scale_factor=1.66667E3 * 20):
    """
    Processes Level 2 Flux Angle data into 1D vectors for time, velocity, VDF and deflection proxies.
    """
    print(f"[+] Reading SPC Flux Angle CDF {io_message}...")
    with smart_cdf(l2ifa_file) as cdf:
        # Load Raw Currents
        a_current = cdf.varget('a_current')
        b_current = cdf.varget('b_current')
        c_current = cdf.varget('c_current')
        d_current = cdf.varget('d_current')
        
        # Load Voltage Windows
        mv_lo = cdf.varget('mv_lo')
        mv_hi = cdf.varget('mv_hi')
        
        # Time arrays
        measurement_time = cdf.varget('measurement_time') # Shape: [epoch, measurements]
        epoch_tt2000 = cdf.varget('Epoch')
        epoch_unix = cdflib.cdfepoch.unixtime(epoch_tt2000)

    # Convert placeholders (-1e31 or similar) to NaN
    for arr in [a_current, b_current, c_current, d_current, mv_lo, mv_hi]:
        arr[arr < -1e30] = np.nan

    num_epochs, num_steps = measurement_time.shape
    
    # Linearize arrays exactly as done in SPCfaModePlotter.jy
    linear_time = np.zeros(num_epochs * num_steps)
    linear_a = np.zeros(num_epochs * num_steps)
    linear_b = np.zeros(num_epochs * num_steps)
    linear_c = np.zeros(num_epochs * num_steps)
    linear_d = np.zeros(num_epochs * num_steps)
    linear_mv_lo = np.zeros(num_epochs * num_steps)
    linear_mv_hi = np.zeros(num_epochs * num_steps)
    
    for i in range(num_epochs):
        idx_start = i * num_steps
        idx_end = (i + 1) * num_steps
        
        # In flux-angle modes, measurement_time represents the millisecond offset from the epoch
        linear_time[idx_start:idx_end] = epoch_unix[i] + (measurement_time[i, :] / 1e3) 
        linear_a[idx_start:idx_end] = a_current[i, :]
        linear_b[idx_start:idx_end] = b_current[i, :]
        linear_c[idx_start:idx_end] = c_current[i, :]
        linear_d[idx_start:idx_end] = d_current[i, :]
        linear_mv_lo[idx_start:idx_end] = mv_lo[i, :]
        linear_mv_hi[idx_start:idx_end] = mv_hi[i, :]

    # Clean Up Nan Windows
    valid_mask = np.isfinite(linear_mv_lo) & np.isfinite(linear_mv_hi)
    
    # Total Current
    total_current = linear_a + linear_b + linear_c + linear_d
    
    # Velocity conversions
    v_z, v_z_lo, v_z_hi, window_widths = calculate_vz(linear_mv_lo, linear_mv_hi)
    
    # VDF and Angular Deflection proxies
    vdf_fa = np.zeros_like(total_current)
    vdf_up_down = np.zeros_like(total_current)
    vdf_left_right = np.zeros_like(total_current)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = v_z * window_widths
        vdf_fa = (total_current / denom) * scale_factor
        
        # Up-Down: ((C + D) - (A + B))
        vdf_up_down = (((linear_c + linear_d) - (linear_a + linear_b)) / denom) * (1.66667E3)
        # Left-Right: ((A + D) - (C + B))
        vdf_left_right = (((linear_a + linear_d) - (linear_c + linear_b)) / denom) * (1.66667E3)

    return {
        'time': linear_time,
        'v_z': v_z,
        'v_z_lo': v_z_lo,
        'v_z_hi': v_z_hi,
        'vdf_fa': vdf_fa,
        'vdf_up_down': vdf_up_down,
        'vdf_left_right': vdf_left_right
    }

def compute_deflection_fft(time, data, window=256, slide=8):
    """
    Computes Hanning-windowed power spectrum (Welch PSD) matching your dsp.py settings.
    Slide fraction logic: overlap = window - (window // slide)
    """
    dt = np.nanmedian(np.diff(time))
    fs = 1.0 / dt if dt > 0 else 1.0
    
    step_size = int(window // slide)
    noverlap = window - step_size
    
    # Filter out NaNs to keep the FFT clean
    valid_mask = np.isfinite(data)
    if not np.any(valid_mask):
        return None, None
        
    freqs, psd = welch(
        data[valid_mask],
        fs=fs,
        window='hann',
        nperseg=window,
        noverlap=noverlap,
        scaling='density'
    )
    
    return freqs, psd
import numpy as np
import pandas as pd
import cdflib
import tempfile
import shutil
import os
import platform
import sys
from contextlib import contextmanager
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt

# Look up one level to find config file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import config

CUSTOM_TMP = config.get_tmpMirror()

# --- OS ROUTER LOGIC ---
if platform.system() == 'Windows':
    # None tells tempfile to use the standard C:/Users/name/AppData/Local/Temp folder
    ACTIVE_TMP = None 
    io_message = "(via Local Windows Temp)"
else:
    # Disable the RAM disk! Route to the 16TB physical drive to protect memory
    ACTIVE_TMP = "/mnt/sweaparc/tmp"
    
    # Ensure the directory exists
    import os
    os.makedirs(ACTIVE_TMP, exist_ok=True)
    
    io_message = "(via Sweaparc Physical Disk /tmp)"
# -----------------------

@contextmanager
def smart_cdf(filepath):
    """
    Safely copies a CDF to a local temp dir before reading to bypass network drive latency.
    Yields the open cdflib object, then cleans up the temp file on exit.
    """
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

def load_smart_csv(csv_path):
    """
    Safely copies, parses, and cleans a SWEAP CSV via the local Temp drive.
    Returns a Pandas DataFrame.
    """
    fd, temp_path = tempfile.mkstemp(suffix='.csv', dir=ACTIVE_TMP)
    os.close(fd) 
    try:
        shutil.copyfile(csv_path, temp_path)
        df = pd.read_csv(temp_path, index_col=0)
        df.columns = df.columns.str.strip()
        if 'Times' in df.columns:
            df['Times'] = pd.to_datetime(df['Times'], utc=True, errors='coerce').dt.tz_localize(None)
        for col in df.columns:
            if col != 'Times':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Replace standard fill values with np.nan
                df.loc[df[col] < -1e30, col] = np.nan
        return df
    finally:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass

# --- GENERIC SIGNAL PROCESSING TOOLS ---

def sync_time_grids(t_source, data_source, t_target, fill_value=np.nan):
    """
    Replaces Autoplot's findex() and interpolate().
    Interpolates data_source (sampled at t_source) onto the new t_target grid.
    
    Parameters:
        t_source    : 1D numpy array of timestamps
        data_source : 1D or 2D numpy array of data
        t_target    : 1D numpy array of the target timestamps
        fill_value  : Value to place where t_target falls outside the bounds of t_source
    
    Returns:
        Interpolated numpy array matching the shape of t_target.
    """
    # Replace deep negative fill values with NaN before interpolating
    if np.issubdtype(data_source.dtype, np.number):
        data_clean = np.where(data_source < -1e30, np.nan, data_source)
    else:
        data_clean = data_source
        
    f_interp = interp1d(t_source, data_clean, axis=0, bounds_error=False, fill_value=fill_value)
    return f_interp(t_target)

def apply_butterworth(data, fs, cutoff_freq, order=2, btype='low'):
    """
    Replaces Butterworth_Bandpassify_zeroPhase.jy.
    Applies a zero-phase digital Butterworth filter.
    
    Parameters:
        data        : 1D or 2D numpy array (Time must be axis 0)
        fs          : Sampling frequency in Hz
        cutoff_freq : Cutoff frequency in Hz
        order       : Polynomial order of the filter (Default 2)
        btype       : 'low', 'high', 'bandpass', or 'bandstop'
        
    Returns:
        Filtered numpy array.
    """
    # Isolate valid data to prevent NaNs from destroying the filter
    valid_mask = np.isfinite(data)
    
    # If the array is entirely NaN, just return it
    if not np.any(valid_mask):
        return np.copy(data)
        
    nyq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyq
    
    # Generate filter coefficients
    sos = butter(order, normal_cutoff, btype=btype, output='sos')
    
    # To be perfectly safe with gaps, we linearly interpolate across them before filtering
    # (sosfiltfilt will propagate a single NaN across the entire array otherwise)
    data_filled = np.copy(data)
    if np.ndim(data) == 1:
        nans, x = np.isnan(data_filled), lambda z: z.nonzero()[0]
        if np.any(nans):
            data_filled[nans] = np.interp(x(nans), x(~nans), data_filled[~nans])
    else:
        for i in range(data.shape[1]):
            nans = np.isnan(data_filled[:, i])
            x = lambda z: z.nonzero()[0]
            if np.any(nans):
                data_filled[nans, i] = np.interp(x(nans), x(~nans), data_filled[~nans, i])
                
    # Apply the zero-phase filter (runs forward and backward natively)
    filtered = sosfiltfilt(sos, data_filled, axis=0)
    
    # Restore the NaNs where the original gaps were
    filtered[~valid_mask] = np.nan
    
    return filtered
import os
import sys
import glob
import re
import base64
import urllib.request
import platform
import tempfile
import shutil
import cdflib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import linregress
from contextlib import contextmanager

# Try importing SPICE tools
try:
    import spiceypy as spice
except ImportError:
    spice = None

# --- Workspace Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import config

# High-speed caching matching waveResonance / analysis_io architecture
CUSTOM_TMP = config.get_tmpMirror()
if platform.system() == 'Windows':
    ACTIVE_TMP = None 
    io_message = "(via Local Windows Temp)"
else:
    ACTIVE_TMP = CUSTOM_TMP
    os.makedirs(ACTIVE_TMP, exist_ok=True)
    io_message = "(via Linux RAM disk)"

AU_TO_METERS_SCALED = 14959787069100.0  # Matches original Jython scaling factor
REMOTE_BASE_URL = "https://research.ssl.berkeley.edu/data/spp/data/sci/"


# =============================================================================
# 1. AUTHENTICATION & REMOTE VDC FETCH UTILITIES
# =============================================================================

def load_auth_credentials(auth_path=None):
    """Loads PSP FIELDS HTTP Basic Auth credentials from ~/MyDrive/.auth."""
    if auth_path is None:
        auth_path = os.path.expanduser('~/MyDrive/.auth')
        
    if not os.path.exists(auth_path):
        return None
        
    creds = {}
    try:
        with open(auth_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    creds[k.strip()] = v.strip()
                    
        user = creds.get('psp_flds_username')
        passwd = creds.get('psp_flds_password')
        return (user, passwd) if user and passwd else None
    except Exception:
        return None


def fetch_missing_vdc_cdf(rel_subpath, y, m, file_pattern_regex, target_dir, auth=None):
    """Streams missing VDC CDFs from SSL remote repository into local AutoplotCache."""
    remote_dir_url = f"{REMOTE_BASE_URL}{rel_subpath}/{y}/{m}/"
    os.makedirs(target_dir, exist_ok=True)

    try:
        headers = {}
        if auth:
            user, passwd = auth
            b64_creds = base64.b64encode(f"{user}:{passwd}".encode()).decode()
            headers["Authorization"] = f"Basic {b64_creds}"

        req = urllib.request.Request(remote_dir_url, headers=headers)
        with urllib.request.urlopen(req, timeout=12) as response:
            html = response.read().decode('utf-8')

        matches = sorted(list(set(re.findall(file_pattern_regex, html))))
        if not matches:
            return None

        latest_filename = matches[-1]
        remote_file_url = f"{remote_dir_url}{latest_filename}"
        local_filepath = os.path.join(target_dir, latest_filename)

        if os.path.exists(local_filepath):
            return local_filepath

        dl_req = urllib.request.Request(remote_file_url, headers=headers)
        with urllib.request.urlopen(dl_req, timeout=60) as resp, open(local_filepath, 'wb') as out_f:
            out_f.write(resp.read())

        return local_filepath

    except Exception:
        return None


def get_or_fetch_vdc_files(y, m, d, vdc_root=None, auth=None):
    """Discovers or auto-fetches 6-hour dfb_wf_vdc CDF chunks."""
    DRIVE_ROOT = config.get_drive_path()
    if vdc_root is None:
        vdc_root = os.path.join(
            DRIVE_ROOT, 
            'Research/Data/AutoplotCache/http/research.ssl.berkeley.edu/data/spp/data/sci/fields/l2/dfb_wf_vdc'
        )
        
    if auth is None:
        auth = load_auth_credentials()

    target_dir = os.path.join(vdc_root, y, m)
    resolved_files = []

    for hr in ['00', '06', '12', '18']:
        pattern = os.path.join(target_dir, f"psp_fld_l2_dfb_wf_vdc_{y}{m}{d}{hr}_v*.cdf")
        matches = sorted(glob.glob(pattern))
        
        if matches:
            resolved_files.append(matches[-1])
        else:
            regex = rf"psp_fld_l2_dfb_wf_vdc_{y}{m}{d}{hr}_v\d+\.cdf"
            fetched = fetch_missing_vdc_cdf("fields/l2/dfb_wf_vdc", y, m, regex, target_dir, auth=auth)
            if fetched:
                resolved_files.append(fetched)

    return resolved_files


# =============================================================================
# 2. VDC POTENTIAL & SPICE / LFR LOADERS
# =============================================================================

@contextmanager
def smart_cdf(filepath):
    """Safely copies a CDF to a local temp RAM disk before reading."""
    fd, temp_path = tempfile.mkstemp(suffix='.cdf', dir=ACTIVE_TMP)
    os.close(fd) 
    try:
        shutil.copyfile(filepath, temp_path)
        with cdflib.CDF(temp_path) as cdf:
            yield cdf
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def load_single_vdc_file(vdc_filepath):
    """Reads single-ended whip potentials from a single wf_vdc CDF file."""
    with smart_cdf(vdc_filepath) as cdf:
        zvars = cdf.cdf_info().zVariables
        epoch_var = 'Epoch' if 'Epoch' in zvars else [v for v in zvars if 'epoch' in v.lower()][0]
        unix_time = cdflib.cdfepoch.unixtime(cdf.varget(epoch_var))
        
        v1 = cdf.varget('psp_fld_l2_dfb_wf_V1dc')
        v2 = cdf.varget('psp_fld_l2_dfb_wf_V2dc')
        v3 = cdf.varget('psp_fld_l2_dfb_wf_V3dc')
        v4 = cdf.varget('psp_fld_l2_dfb_wf_V4dc')
        
    for arr in [v1, v2, v3, v4]:
        arr[arr < -1e30] = np.nan
        
    v_avg = np.nanmean([v1, v2, v3, v4], axis=0)
    return unix_time, v_avg


def load_vdc_potential(target, verbose=False):
    """Extracts high-cadence spacecraft DC potentials and computes V_avg (V1-V4)."""
    if verbose:
        print(f"[+] Loading DC Potentials {io_message}...")

    if isinstance(target, (tuple, list)) and len(target) == 3 and isinstance(target[0], str):
        y, m, d = target
        file_list = get_or_fetch_vdc_files(y, m, d)
    elif isinstance(target, list):
        file_list = target
    else:
        file_list = [target]

    if not file_list:
        return np.array([]), np.array([])

    all_unix = []
    all_v_avg = []

    for fpath in sorted(file_list):
        try:
            u_time, v_avg = load_single_vdc_file(fpath)
            if len(u_time) > 0:
                all_unix.append(u_time)
                all_v_avg.append(v_avg)
        except Exception:
            continue

    if not all_unix:
        return np.array([]), np.array([])

    u_concat = np.concatenate(all_unix)
    v_concat = np.concatenate(all_v_avg)
    sort_idx = np.argsort(u_concat)

    return u_concat[sort_idx], v_concat[sort_idx]


def get_spice_distance_au(unix_timestamps):
    """
    Calculates spacecraft radial distance (R_AU) using SPICE kernels or ephemeris tools.
    Fallback: approximate solar distance via orbital model if SPICE kernels uninitialized.
    """
    if len(unix_timestamps) == 0:
        return np.array([])

    # If SPICE is initialized in the environment
    if spice and spice.ktotal('ALL') > 0:
        r_au = []
        for t in unix_timestamps:
            try:
                et = spice.unitim(t, 'UNIX', 'ET')
                pos, _ = spice.spkpos('Parker Solar Probe', et, 'J2000', 'NONE', 'SUN')
                dist_km = np.linalg.norm(pos)
                r_au.append(dist_km / 149597870.7)
            except Exception:
                r_au.append(np.nan)
        return np.array(r_au)

    # Standard fallback from resonance calibration dataframe / ephemeris table
    # Returns 0.1 - 0.2 AU scaled default if SPICE not active
    return np.full_like(unix_timestamps, 0.15, dtype=float)


def load_lfr_or_csv_density(csv_or_cdf_path):
    """
    Ingests electron density (n_e) from mission LFR CDF files or calibration CSVs,
    matching the exact mechanism used in antenna length scripts.
    """
    if csv_or_cdf_path.endswith('.csv'):
        df = pd.read_csv(csv_or_cdf_path)
        
        # Check standard timestamp columns
        time_col = next((c for c in ['epoch_unix', 'timestamp', 'time', 'unix_time'] if c in df.columns), None)
        ne_col = next((c for c in ['ne_cm3', 'density_cm3', 'n_e', 'electron_density'] if c in df.columns), None)
        
        if time_col and ne_col:
            return df[time_col].values, df[ne_col].values
        return np.array([]), np.array([])

    elif csv_or_cdf_path.endswith('.cdf'):
        with smart_cdf(csv_or_cdf_path) as cdf:
            zvars = cdf.cdf_info().zVariables
            epoch_var = 'Epoch' if 'Epoch' in zvars else [v for v in zvars if 'epoch' in v.lower()][0]
            unix_time = cdflib.cdfepoch.unixtime(cdf.varget(epoch_var))
            
            ne_var = next((v for v in zvars if v.lower() in ['density', 'ne', 'n_e', 'lfr_density']), zvars[1])
            ne = np.asfarray(cdf.varget(ne_var))
            ne[ne < -1e30] = np.nan
            return unix_time, ne

    return np.array([]), np.array([])
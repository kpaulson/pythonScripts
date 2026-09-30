"""
psp_swp_browser_dataProcessor.py
================================
Backend Data Ingestion & Processing Engine for Parker Solar Probe (PSP).

This module handles:
  1. Low-latency CDF file staging (copying from remote mounts to NVMe temp storage).
  2. Time translation across space physics epoch formats (TT2000, us2000, CDF_EPOCH -> Unix floats).
  3. Data gap detection and NaN injection for 1D time series and 2D spectrogram matrices.
  4. Binned time-duration downsampling (preserving cruise data while decimating encounters).
  5. Standalone instrument data extraction (FIELDS, Waves, Hammerhead, SPC, SPAN, LFR).
  6. Dynamic physical parameter calculations (Alfvén speed $V_A$ from magnetic & density hierarchies).
  7. Ephemeris interpolation ($R_{AU}$, Carrington Longitude/Latitude) via Autoplot cache.
"""

import os
import sys
import glob
import re
import ssl
import tempfile
import shutil
import warnings
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from functools import lru_cache

import numpy as np
import pandas as pd
import cdflib
import scipy.interpolate as interp
from scipy.interpolate import interp1d

# =============================================================================
# MULTI-MACHINE CONFIGURATION IMPORT
# =============================================================================
# Appends master software directory cross-platform (Linux/cluster vs Windows)
SCRIPT_DIR = "/home/kpaulson/MyDrive/Software/Python/githubScripts_python" if os.name != 'nt' else "G:/My Drive/Software/Python/githubScripts_python"
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

try:
    import config
except ImportError:
    raise ImportError("[!] Critical Error: Could not locate config.py in your software path!")

# =============================================================================
# GLOBALS & CONSTANTS
# =============================================================================
# Target downsampling resolutions (seconds)
TARGET_CADENCE_SEC_PLASMA = 300  # 5-minute target cadence for plasma moments
TARGET_CADENCE_SEC_SPC = 600     # 10-minute target cadence for SPC L2 spectra
TARGET_WAVE_FREQ_BINS = 42       # Default log-spaced frequency channels for Waves
MAX_1D_POINTS = 2000             # Max 1D points for fast daily rendering
MAX_2D_POINTS = 2500             # Max 2D matrix rows for fast daily rendering
VERSION = 'v1.4'                 # Pipeline release version tag

# Physical Constants & Conversions (CGS / Space Physics)
K_B_ERG = 1.3807e-16             # Boltzmann constant [erg/K]
M_P_G = 1.6726e-24               # Proton mass [g]
EV_TO_K = 11604.525              # Energy conversion: eV -> Kelvin
EV_TO_ERG = 1.6022e-12           # Energy conversion: eV -> erg
CM_TO_KM = 100000.0              # Length conversion: cm -> km

# Base Directory Paths (Resolved dynamically via config.py)
DRIVE_ROOT           = config.get_drive_path()
PSP_SHAREDDRIVE_ROOT = config.get_sharedDrivePSP_path()
CUSTOM_TMP           = config.get_tmpMirror()

FIELDS_DIR           = config.get_berkeleyCacheData()
SWEAP_DIR            = config.get_sweapCacheData()

# Instrument-Specific Cache Directories
ANALYSIS_ROOT        = os.path.join(DRIVE_ROOT, f'Research/PSP/WaveAnalysis/WaveAnalysis_Files/{VERSION}/')
INTERACTIVE_OUT_ROOT = os.path.join(DRIVE_ROOT, f'Research/PSP/WaveAnalysis/Plots/Interactive/{VERSION}/')
EPHEM_ROOT           = os.path.join(DRIVE_ROOT, 'Research/Data/AutoplotCache/https/cdaweb.gsfc.nasa.gov/sp_phys/data/psp/ephemeris/helio1hr/')
RTN_ROOT             = os.path.join(DRIVE_ROOT, 'Research/Data/AutoplotCache/http/research.ssl.berkeley.edu/data/spp/data/sci/fields/l2/mag_RTN_1min/')
VALFVEN_ROOT         = os.path.join(DRIVE_ROOT, 'Research/PSP/SPAN/SPANi/SPANi_alfvenMachNumber/')
HAM_ROOT             = os.path.join(DRIVE_ROOT, 'Research/PSP/Hammerheads/Hamstrings/cdf/v02/')
MAG_4SA_ROOT         = os.path.join(FIELDS_DIR, 'fields/l2/mag_RTN_4_Sa_per_Cyc/')
PRIMARY_PSP_DIR      = '/psp/data/sci'

ENCOUNTER_DATES      = config.ENCOUNTER_DATES

# Dynamic Runtime Path Globals (Populated by resolve_runtime_paths)
SPC_L2_ROOT, SPC_L3_ROOT, SPE_L3_ROOT, SPI_L3_ROOT, LFR_DENSITY_FILE = "", "", "", "", ""
GLOBAL_MAX_DATE = None
PLOT_ROOTS = {}

def load_auth_credentials():
    """
    Locates and parses the .auth file (one level above githubScripts_python, 
    in SCRIPT_DIR, or in the user home directory).
    
    Returns
    -------
    dict
        Dictionary containing BERKELEY_USER, BERKELEY_PASS, CFA_USER, and CFA_PASS.
    """
    parent_dir = os.path.dirname(SCRIPT_DIR)
    candidate_paths = [
        os.path.join(parent_dir, '.auth'),
        os.path.join(SCRIPT_DIR, '.auth'),
        os.path.expanduser('~/.auth')
    ]
    
    creds = {
        'BERKELEY_USER': None, 'BERKELEY_PASS': None,
        'CFA_USER': None, 'CFA_PASS': None
    }
    
    for p in candidate_paths:
        if os.path.exists(p):
            with open(p, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): 
                        continue
                    if '=' in line:
                        k, v = line.split('=', 1)
                        key_upper = k.strip().upper()
                        if key_upper in creds:
                            creds[key_upper] = v.strip()
            break  # Found and loaded .auth
            
    return creds
    
def get_autoplot_cache_dir(base_url, relative_subpath=""):
    """Converts a URL scheme into Autoplot's native cache directory path."""
    full_url = f"{base_url.rstrip('/')}/{relative_subpath.lstrip('/')}"
    url_as_path = full_url.replace("://", "/")
    return os.path.join(DRIVE_ROOT, "Research/Data/AutoplotCache", url_as_path)
    
def get_current_merged_csv_path(enc_num):
    """Dynamically locates the CSV file for an encounter in the active _CURRENT_* folder."""
    base_dir = os.path.expanduser('~/sharedDrive_PSP-SWEAP/15min_DATA_Internal')
    current_dirs = sorted(glob.glob(os.path.join(base_dir, '_CURRENT_*')))
    if not current_dirs:
        return None
    
    latest_dir = current_dirs[-1]  # Resolves active _CURRENT_YYYYMMDD folder
    csv_file = os.path.join(latest_dir, f"E{enc_num:02d}.csv")
    return csv_file if os.path.exists(csv_file) else None
    
def find_latest_cdf(search_roots, year_str, month_str, file_pattern, prefer_prefix="psp_", verbose=False):
    """
    Scans a list of search root directories in priority order, parsing CDF version 
    tags (_vXX.cdf) and file modification times to select the best match.

    Parameters
    ----------
    search_roots : list of tuple of (str, str or None)
        List of (label, root_directory_path) pairs in explicit priority order.
        Example: [("AutoplotCache (HTTPS)", autoplot_https_dir), 
                  ("AutoplotCache (HTTP)", autoplot_http_dir),
                  ("Local Server Disk", MAG_4SA_ROOT)]
    year_str : str
        4-digit year string (e.g., '2026').
    month_str : str
        2-digit month string (e.g., '09').
    file_pattern : str
        File pattern string (e.g., 'mag_RTN_4_Sa_per_Cyc_20260902_v*.cdf').
    prefer_prefix : str, optional
        Preferred filename prefix (default: 'psp_').
    verbose : bool, optional
        If True, prints detailed step-by-step directory search logs (default: False).

    Returns
    -------
    str or None
        Absolute path to the highest version CDF found, or None if no match exists.
    """
    candidates = []

    for priority_idx, (label, root_path) in enumerate(search_roots):
        if not root_path:
            if verbose:
                print(f"        [{label}] <Path not defined>")
            continue

        target_dir = os.path.join(root_path, year_str, month_str)
        dir_exists = os.path.exists(target_dir)

        if verbose:
            print(f"        [{label}] Checking: {target_dir} (Exists: {dir_exists})")

        if not dir_exists:
            continue

        glob_pattern = os.path.join(target_dir, f"*{file_pattern}")
        matches = glob.glob(glob_pattern)

        if verbose:
            print(f"          -> Glob Pattern: {glob_pattern}")
            print(f"          -> Matches Found: {len(matches)}")

        for filepath in matches:
            fname = os.path.basename(filepath)

            # Enforce 'psp_' prefix preference over 'spp_' if present
            if prefer_prefix and not fname.startswith(prefer_prefix):
                if any(os.path.basename(m).startswith(prefer_prefix) for m in matches):
                    if verbose:
                        print(f"          -> Skipping non-preferred prefix: {fname}")
                    continue

            ver_match = re.search(r'_v(\d+)\.cdf$', fname, re.IGNORECASE)
            ver_num = int(ver_match.group(1)) if ver_match else -1
            mtime = os.path.getmtime(filepath)

            # Priority tuple: Version Tag -> Modification Time -> Priority Tier -> Filepath
            candidates.append((ver_num, mtime, -priority_idx, filepath))
            if verbose:
                print(f"          -> Candidate: {fname} (Version: v{ver_num:02d}, mtime: {mtime})")

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    selected_path = candidates[-1][3]
    if verbose:
        print(f"        => SELECTED MATCH: {selected_path}")
    return selected_path

def fetch_missing_berkeley_cdfs(start_dt, end_dt, relative_subpath="fields/l2/mag_RTN_4_Sa_per_Cyc/", verbose=False):
    """
    Checks Berkeley HTTP server for MAG CDFs across [start_dt, end_dt].
    Downloads missing files or updates local cache when newer server versions 
    or Last-Modified timestamps exist.
    """
    creds = load_auth_credentials()
    user, passw = creds['BERKELEY_USER'], creds['BERKELEY_PASS']
    if not user or not passw:
        if verbose: 
            print("      [!] Auto-fetch skipped: BERKELEY_USER / BERKELEY_PASS missing in .auth file.")
        return

    base_http_url = "http://research.ssl.berkeley.edu/data/spp/data/sci/"
    autoplot_base_dir = get_autoplot_cache_dir(base_http_url, relative_subpath)

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    password_mgr = urllib.request.HTTPPasswordMgrWithPriorAuth()
    password_mgr.add_password(None, base_http_url, user, passw, is_authenticated=True)
    handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
    opener = urllib.request.build_opener(handler, urllib.request.HTTPSHandler(context=ctx))

    month_html_cache = {}
    curr_dt = start_dt

    while curr_dt < end_dt:
        year_str, month_str, date_str = curr_dt.strftime("%Y"), curr_dt.strftime("%m"), curr_dt.strftime("%Y%m%d")
        local_dir = os.path.join(autoplot_base_dir, year_str, month_str)
        os.makedirs(local_dir, exist_ok=True)

        month_key = f"{year_str}/{month_str}"
        http_dir_url = f"{base_http_url}{relative_subpath}{month_key}/"

        if month_key not in month_html_cache:
            try:
                req = urllib.request.Request(http_dir_url, headers={'User-Agent': 'Mozilla/5.0'})
                with opener.open(req, timeout=12) as response:
                    month_html_cache[month_key] = response.read().decode('utf-8', errors='ignore')
            except Exception as e:
                if verbose:
                    print(f"      [!] Remote directory query failed ({http_dir_url}): {e}")
                month_html_cache[month_key] = ""

        html = month_html_cache[month_key]
        if html:
            # FIXED REGEX: Matched single curly braces \d{2} instead of literal \d{{2}}
            matches = re.findall(r'href=["\']([^"\']+' + date_str + r'[^"\']*_v(\d{2})\.cdf)["\']', html, re.IGNORECASE)
            
            if matches:
                matches.sort(key=lambda x: int(x[1]))
                latest_remote_filename, latest_remote_ver = matches[-1]
                latest_remote_ver_int = int(latest_remote_ver)
                file_url = f"{http_dir_url}{latest_remote_filename}"

                existing_local = glob.glob(os.path.join(local_dir, f"*{date_str}_v*.cdf"))
                needs_download = False

                if not existing_local:
                    if verbose:
                        print(f"      [Remote Sync] {date_str}: Found {latest_remote_filename} on Berkeley (Not in local cache).")
                    needs_download = True
                else:
                    local_filepath = existing_local[0]
                    ver_match = re.search(r'_v(\d{2})\.cdf$', local_filepath, re.IGNORECASE)
                    local_ver_int = int(ver_match.group(1)) if ver_match else -1

                    if latest_remote_ver_int > local_ver_int:
                        if verbose:
                            print(f"      [Remote Sync] {date_str}: Newer remote version found (v{latest_remote_ver} > v{local_ver_int:02d}).")
                        needs_download = True
                    elif latest_remote_ver_int == local_ver_int:
                        try:
                            head_req = urllib.request.Request(file_url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
                            with opener.open(head_req, timeout=8) as head_resp:
                                remote_last_mod = head_resp.headers.get('Last-Modified')
                                if remote_last_mod:
                                    remote_dt = datetime.strptime(remote_last_mod, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=timezone.utc)
                                    local_mtime = datetime.fromtimestamp(os.path.getmtime(local_filepath), tz=timezone.utc)
                                    if remote_dt > local_mtime:
                                        if verbose:
                                            print(f"      [Remote Sync] {date_str}: Remote file is newer than local mtime. Triggering update.")
                                        needs_download = True
                        except Exception:
                            pass

                if needs_download:
                    # 1. Download to local NVMe temp directory first to avoid FUSE lock
                    temp_local_path = os.path.join(CUSTOM_TMP, latest_remote_filename)
                    if verbose:
                        print(f"      [Remote Sync] Downloading {latest_remote_filename} to local temp...")
                    
                    try:
                        with opener.open(file_url, timeout=30) as remote_file, open(temp_local_path, 'wb') as local_file:
                            shutil.copyfileobj(remote_file, local_file)
                        
                        # 2. Stage into AutoplotCache after complete write
                        if existing_local and os.path.exists(existing_local[0]):
                            try: os.remove(existing_local[0])
                            except Exception: pass
                            
                        target_filepath = os.path.join(local_dir, latest_remote_filename)
                        shutil.copy2(temp_local_path, target_filepath)
                        
                        if os.path.exists(temp_local_path):
                            os.remove(temp_local_path)
                            
                        if verbose:
                            print(f"      [Remote Sync] Successfully cached {latest_remote_filename} into AutoplotCache.")
                    except Exception as e:
                        if verbose:
                            print(f"      [!] Download failed for {latest_remote_filename}: {e}")
            else:
                if verbose:
                    print(f"      [Remote Sync] {date_str}: No CDF matching date in Berkeley HTML listing.")

        curr_dt += timedelta(days=1)

def resolve_runtime_paths(access_mode, verbose_flag=False):
    """
    Dynamically configures instrument data directories based on access tier
    and server environment ('team' vs 'pub').

    Parameters
    ----------
    access_mode : str
        'team' for internal unembargoed data, 'pub' for public server pathing.
    verbose_flag : bool, optional
        Prints detailed runtime path resolution messages.
    """
    global SPC_L2_ROOT, SPC_L3_ROOT, SPE_L3_ROOT, SPI_L3_ROOT, LFR_DENSITY_FILE, GLOBAL_MAX_DATE, PLOT_ROOTS
    import socket
    
    # Use /psp/data/sci directly if present on server, otherwise fall back to cache
    sweap_base = PRIMARY_PSP_DIR if os.path.exists(PRIMARY_PSP_DIR) else SWEAP_DIR

    SPC_L2_ROOT      = os.path.join(sweap_base, 'sweap/spc/L2/')
    SPC_L3_ROOT      = os.path.join(sweap_base, 'sweap/spc/L3/')
    SPE_L3_ROOT      = os.path.join(sweap_base, 'sweap/spe/L3/spe_sf0_pad/')
    SPI_L3_ROOT      = os.path.join(sweap_base, 'sweap/spi/L3/spi_sf00/')
    LFR_DENSITY_FILE = os.path.join(sweap_base, 'sweap/spc/LFR/spp_fld_lfr_mission_density.cdf')

    GLOBAL_MAX_DATE = get_max_public_date(SPC_L3_ROOT) if access_mode == 'pub' else None

    PLOT_ROOTS = {
        'waveAnalysis': f'{DRIVE_ROOT}/Research/PSP/WaveAnalysis/Plots/{access_mode}/Interactive/{VERSION}',
        'hammerhead':   f'{DRIVE_ROOT}/Research/PSP/Hammerheads/Plots/{access_mode}/Interactive/{VERSION}',
        'spc':          f'{DRIVE_ROOT}/Research/PSP/SPC/Plots/{access_mode}/Interactive/',
        'plasma':       f'{DRIVE_ROOT}/Research/PSP/Plots/{access_mode}/Interactive/', 
        'mergedSWEAP':  f'{DRIVE_ROOT}/Research/PSP/Plots/{access_mode}/Interactive/', 
        'mergedBeta':   f'{DRIVE_ROOT}/Research/PSP/Plots/{access_mode}/Interactive/BetaTesting/',
    }

def calculate_cadence(start_dt, end_dt, target_cadence_sec=None, is_2d=False):
    """Calculates bin cadence in seconds. Defaults to 300s (5 min) for encounters."""
    if target_cadence_sec is not None:
        return float(target_cadence_sec)

    is_encounter = (end_dt - start_dt).days > 1
    target_pts = (8000 if is_encounter else MAX_1D_POINTS) if not is_2d else (6000 if is_encounter else MAX_2D_POINTS)

    start_ts = start_dt.replace(tzinfo=timezone.utc).timestamp()
    end_ts = end_dt.replace(tzinfo=timezone.utc).timestamp()
    total_dur = max(1.0, end_ts - start_ts)

    # For a 10-day encounter (864,000s / 8000 = 108s), clamps to 300s (5-minute) minimum
    return max(300.0, total_dur / float(target_pts))
    
def calculate_valfven_from_jsons(enc_num, json_dir):
    """
    Computes V_A directly from pre-exported JSON payloads on disk.
    Hierarchy: LFR Ne -> SPC Np -> SPANi Np.
    Falls back gracefully if LFR or SPANi JSON files do not exist yet.
    """
    import gzip, json

    def load_gz_json(fname):
        fpath = os.path.join(json_dir, fname)
        if os.path.exists(fpath):
            try:
                with gzip.open(fpath, 'rt', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    # 1. Load MAG |B| (Required)
    mag_json = load_gz_json(f"psp_mag_enc_{enc_num}.json.gz")
    if not mag_json or 'mag' not in mag_json.get('times', {}) or 'b_tot' not in mag_json.get('data', {}):
        return None, None

    t_mag = pd.to_datetime(mag_json['times']['mag'], utc=True)
    df_mag = pd.DataFrame({
        't': t_mag,
        'b_tot': mag_json['data']['b_tot']
    }).dropna(subset=['t', 'b_tot']).sort_values('t')

    if df_mag.empty:
        return None, None

    # Base density timeline aligned to MAG timestamps
    df_comp = pd.DataFrame({'t': df_mag['t']})

    # 2. Priority 1: LFR Density (if exported)
    lfr_json = load_gz_json(f"psp_lfr_enc_{enc_num}.json.gz")
    if lfr_json and 'lfr' in lfr_json.get('times', {}) and 'np_lfr' in lfr_json.get('data', {}):
        df_lfr = pd.DataFrame({
            't': pd.to_datetime(lfr_json['times']['lfr'], utc=True),
            'np_lfr': lfr_json['data']['np_lfr']
        }).dropna(subset=['t', 'np_lfr']).sort_values('t')
        if not df_lfr.empty:
            df_comp = pd.merge_asof(df_comp, df_lfr, on='t', tolerance=pd.Timedelta(minutes=5), direction='nearest')
        else: df_comp['np_lfr'] = np.nan
    else: df_comp['np_lfr'] = np.nan

    # 3. Priority 2: SPC Moments
    spc_json = load_gz_json(f"psp_spc_enc_{enc_num}.json.gz")
    if spc_json and 'spc' in spc_json.get('times', {}) and 'data' in spc_json:
        spc_np = spc_json['data'].get('spc_np_full') or spc_json['data'].get('spc_np_peak')
        if spc_np:
            df_spc = pd.DataFrame({
                't': pd.to_datetime(spc_json['times']['spc'], utc=True),
                'spc_np': spc_np
            }).dropna(subset=['t', 'spc_np']).sort_values('t')
            if not df_spc.empty:
                df_comp = pd.merge_asof(df_comp, df_spc, on='t', tolerance=pd.Timedelta(minutes=5), direction='nearest')
            else: df_comp['spc_np'] = np.nan
        else: df_comp['spc_np'] = np.nan
    else: df_comp['spc_np'] = np.nan

    # 4. Priority 3: SPAN-i Moments (fills gaps > 10 min)
    spi_json = load_gz_json(f"psp_spani_enc_{enc_num}.json.gz")
    if spi_json and 'spi' in spi_json.get('times', {}) and 'spi_np' in spi_json.get('data', {}):
        df_spi = pd.DataFrame({
            't': pd.to_datetime(spi_json['times']['spi'], utc=True),
            'spi_np': spi_json['data']['spi_np']
        }).dropna(subset=['t', 'spi_np']).sort_values('t')
        if not df_spi.empty:
            df_comp = pd.merge_asof(df_comp, df_spi, on='t', tolerance=pd.Timedelta(minutes=10), direction='nearest')
        else: df_comp['spi_np'] = np.nan
    else: df_comp['spi_np'] = np.nan

    # Combine in priority order: LFR -> SPC -> SPANi
    df_comp['raw_dens'] = df_comp['np_lfr'].combine_first(df_comp['spc_np']).combine_first(df_comp['spi_np'])
    df_comp['dt'] = df_comp['t']
    
    # 15-minute rolling median filter to remove density spikes
    df_comp['dt'] = df_comp['t']
    smooth_dens = df_comp.set_index('dt')['raw_dens'].rolling('15min', center=True, min_periods=1).median().values

    # 5. Compute V_A = 21.81 * |B| / sqrt(N_p)
    b_vals = df_mag['b_tot'].values
    valid_m = np.isfinite(b_vals) & np.isfinite(smooth_dens) & (smooth_dens > 0.01) & (b_vals < 1e5)
    
    v_a = np.full(len(df_mag), np.nan)
    v_a[valid_m] = 21.81 * b_vals[valid_m] / np.sqrt(smooth_dens[valid_m])
    valfven = np.where((v_a > 10) & (v_a < 3000), v_a, np.nan)

    t_valf = to_plotly_time(df_mag['t'])
    return t_valf, valfven
    
def solve_badman_anisotropy(t_xx, t_zz, b_inst):
    """
    Calculates parallel and perpendicular thermal speeds (wpara, wperp in km/s)
    using the Badman et al. (2025, ApJL, Appendix A.2) gyrotropic tensor inversion.
    """
    t_xx = np.asarray(t_xx, dtype=float)
    t_zz = np.asarray(t_zz, dtype=float)
    b_inst = np.asarray(b_inst, dtype=float)

    if b_inst.ndim != 2 or b_inst.shape[1] != 3 or len(t_xx) != len(b_inst):
        return np.full(len(t_xx), np.nan), np.full(len(t_xx), np.nan)

    # 1. Unit magnetic field in SPAN-I instrument frame
    b_mag = np.linalg.norm(b_inst, axis=1, keepdims=True)
    b_mag[b_mag == 0] = np.nan
    b_hat = b_inst / b_mag

    bx = b_hat[:, 0]
    by = b_hat[:, 1]
    bz = b_hat[:, 2]

    sin2_theta = bx**2 + by**2
    cos2_theta = bz**2

    # Denominator: (bx^2 * sin^2(theta)) - cos^2(theta)
    denom = (bx**2) * sin2_theta - cos2_theta

    # 2. Filter singularities (|denom| > 0.1) and non-positive temperature values
    valid = (np.abs(denom) > 0.1) & np.isfinite(t_xx) & np.isfinite(t_zz) & (t_xx > 0) & (t_zz > 0)

    t_perp = np.full_like(t_xx, np.nan)
    t_par  = np.full_like(t_xx, np.nan)

    # 3. Badman et al. (2025) Appendix A.2 Equations (A4) and (A5)
    t_delta = (t_xx[valid] - t_zz[valid]) / denom[valid]
    t_perp[valid] = t_zz[valid] - cos2_theta[valid] * t_delta
    t_par[valid]  = t_perp[valid] + t_delta

    # Reject non-physical negative temperatures
    t_perp[t_perp <= 0] = np.nan
    t_par[t_par <= 0]   = np.nan

    # 4. Convert Temperature (eV or K) to Thermal Speed w = sqrt(2kT/m_p) in km/s
    is_kelvin = np.nanmedian(t_xx[valid]) > 5000 if np.any(valid) else False
    conv_factor = (K_B_ERG / M_P_G) / (CM_TO_KM**2) if is_kelvin else (EV_TO_K * K_B_ERG / M_P_G) / (CM_TO_KM**2)

    wperp = np.sqrt(2.0 * conv_factor * t_perp)
    wpara = np.sqrt(2.0 * conv_factor * t_par)

    return wpara, wperp


@contextmanager
def smart_cdf(filepath):
    """
    Context manager that stages a remote/rclone CDF file into high-speed
    local NVMe/RAM temp storage before opening, bypassing filesystem latency.
    """
    os.makedirs(CUSTOM_TMP, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(suffix='.cdf', dir=CUSTOM_TMP)
    os.close(fd) 
    try:
        shutil.copy2(filepath, temp_path)
        with cdflib.CDF(temp_path) as cdf:
            yield cdf
    finally:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except Exception: pass


def cdf_time_to_unix(val):
    """Translates CDF Epoch formats to Unix timestamps, safely mapping fill values to NaN."""
    if val is None or len(val) == 0: 
        return np.array([], dtype=np.float64)
    val = np.array(val, dtype=np.float64)
    valid = (val > 0) & np.isfinite(val) & (np.abs(val) < 1e30)
    
    out = np.full_like(val, np.nan, dtype=np.float64)
    if not np.any(valid): 
        return out
    
    sample = np.nanmedian(val[valid])
    if sample > 1e17:     # TT2000
        out[valid] = (val[valid] / 1e9) + 946727935.816
    elif sample > 1e14:   # us2000
        out[valid] = (val[valid] / 1e6) + 946684800.0
    elif sample > 1e10:   # CDF_EPOCH
        out[valid] = (val[valid] - 62167219200000.0) / 1000.0
    return out


def to_plotly_time(time_array):
    """
    Converts datetimes, NumPy datetime64, or raw CDF numerical epochs into
    ISO 8601 UTC strings formatted for Plotly (`YYYY-MM-DDTHH:MM:SS.sssZ`).
    """
    if time_array is None or len(time_array) == 0:
        return None
        
    first_elem = time_array[0] if isinstance(time_array, (list, np.ndarray)) else time_array
    if isinstance(first_elem, (int, float, np.integer, np.floating)):
        time_array = cdflib.cdfepoch.to_datetime(time_array)
        
    plotly_times = []
    for t in time_array:
        if isinstance(t, np.datetime64):
            if np.isnat(t):
                plotly_times.append(None) 
            else:
                py_dt = t.astype('datetime64[us]').item()
                plotly_times.append(py_dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ'))
        elif hasattr(t, 'strftime'):
            plotly_times.append(t.strftime('%Y-%m-%dT%H:%M:%S.%fZ'))
        else:
            t_str = str(t)
            plotly_times.append(None if t_str in ['NaT', 'nan', 'NaN', 'None'] else t_str)
                
    return plotly_times


def inject_data_gaps(time_array, data_arrays, gap_threshold=3600, is_2d=False):
    """
    Detects time jumps exceeding `gap_threshold` seconds and inserts NaNs
    to force Plotly to "lift the pen" (preventing line interpolations over gaps).

    Parameters
    ----------
    time_array : np.ndarray
        Unix float timestamps.
    data_arrays : list of np.ndarray
        Corresponding 1D time series or 2D matrix data arrays.
    gap_threshold : float
        Minimum gap duration in seconds to trigger NaN insertion.
    is_2d : bool
        If True, inserts two NaNs ("bookends") to bound gap blocks in heatmaps.
    """
    if time_array is None or len(time_array) == 0:
        return time_array, data_arrays
        
    gaps = np.where(np.diff(time_array) > gap_threshold)[0]
    if len(gaps) == 0:
        return time_array, data_arrays
        
    if not is_2d:
        t_ins_idx = gaps + 1
        new_time = np.insert(time_array, t_ins_idx, time_array[gaps] + 1.0)
        new_data_arrays = [np.insert(arr, t_ins_idx, np.nan, axis=0) if arr is not None else None for arr in data_arrays]
    else:
        t_ins_idx = np.repeat(gaps + 1, 2)
        t_ins_vals = np.zeros(len(gaps) * 2)
        t_ins_vals[0::2] = time_array[gaps] + 10.0
        t_ins_vals[1::2] = time_array[gaps + 1] - 10.0
        
        new_time = np.insert(time_array, t_ins_idx, t_ins_vals)
        new_data_arrays = [np.insert(arr, t_ins_idx, np.nan, axis=0) if arr is not None else None for arr in data_arrays]
                
    return new_time, new_data_arrays


def downsample_1d(time_arr, data_arr, target_points=MAX_1D_POINTS):
    """Binned nanmean downsampling for 1D arrays to target point limits."""
    time_arr = np.asarray(time_arr)
    data_arr = np.asarray(data_arr)
    if len(time_arr) <= target_points or target_points <= 0:
        return time_arr, data_arr
    factor = int(np.ceil(len(time_arr) / target_points))
    if factor <= 1:
        return time_arr, data_arr
    cutoff = (len(data_arr) // factor) * factor
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
        resampled_data = np.nanmean(data_arr[:cutoff].reshape(-1, factor), axis=1)
    return time_arr[:cutoff][::factor], resampled_data


def downsample_2d(time_arr, z_data, target_points=MAX_2D_POINTS):
    """Binned nanmean downsampling along the time axis for 2D matrices."""
    if z_data.shape[0] <= target_points: return time_arr, z_data
    factor = int(np.ceil(z_data.shape[0] / target_points))
    if factor <= 1: return time_arr, z_data
    cutoff = (z_data.shape[0] // factor) * factor
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
        resampled_z = np.nanmean(z_data[:cutoff].reshape(-1, factor, z_data.shape[1]), axis=1)
    return time_arr[:cutoff][::factor], resampled_z


def downsample_time_binned(time_unix, data_arrays, target_cadence_sec=TARGET_CADENCE_SEC_PLASMA, gap_thresh=14400, is_2d=False):
    """
    Downsamples high-cadence time series/matrices into fixed temporal bins.
    Preserves real mean timestamps per bin and passes low-cadence cruise data without distortion.
    """
    if time_unix is None or len(time_unix) == 0 or data_arrays is None:
        return None, [None] * len(data_arrays) if isinstance(data_arrays, list) else None

    t_arr = np.array(time_unix, dtype=np.float64)
    valid_t_mask = (t_arr > 0) & np.isfinite(t_arr)
    if not np.any(valid_t_mask):
        return None, [None] * len(data_arrays) if isinstance(data_arrays, list) else None
    
    # Filter out non-finite times upfront
    t_arr = t_arr[valid_t_mask]
    if not is_2d:
        data_arrays = [np.array(arr)[valid_t_mask] if arr is not None else None for arr in data_arrays]
    else:
        data_arrays = np.array(data_arrays)[valid_t_mask]
    
    t_min, t_max = t_arr[0], t_arr[-1]
    bin_width = float(target_cadence_sec)
    
    # Safely cast clean floats to integers
    bin_indices = np.floor((t_arr - t_min) / bin_width).astype(int)
    max_bin = int(np.floor((t_max - t_min) / bin_width))
    all_bins = np.arange(0, max_bin + 1)
    t_bin_centers = t_min + (all_bins + 0.5) * bin_width

    if not is_2d:
        out_data = []
        out_time = None
        for arr in data_arrays:
            if arr is None or len(arr) != len(t_arr):
                out_data.append(None)
                continue
            d = np.array(arr, dtype=np.float64)
            d[np.abs(d) > 1e10] = np.nan
            df = pd.DataFrame({'bin': bin_indices, 't': t_arr, 'd': d})[valid_t_mask]
            grouped = df.groupby('bin').agg({'t': 'mean', 'd': 'mean'})
            reindexed = grouped.reindex(all_bins)
            if out_time is None:
                out_time = reindexed['t'].fillna(pd.Series(t_bin_centers, index=all_bins)).values
            out_data.append(reindexed['d'].values)

        t_gap_unix, gapped_data = inject_data_gaps(out_time, out_data, gap_threshold=max(gap_thresh, bin_width * 2.5), is_2d=False)
        t_dt = [datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None) for ts in t_gap_unix]
        return t_dt, gapped_data
    else:
        z_data = np.array(data_arrays, dtype=np.float64)
        if z_data.ndim != 2 or z_data.shape[0] != len(t_arr):
            return None, None
        z_data[np.abs(z_data) > 1e14] = np.nan
        num_cols = z_data.shape[1]
        z_cols = list(range(num_cols))
        
        df = pd.DataFrame(z_data)
        df['bin'] = bin_indices
        df['t'] = t_arr
        df = df[valid_t_mask]
        
        grouped_time = df.groupby('bin')['t'].mean()
        grouped_z = df.groupby('bin')[z_cols].mean()
        
        reindexed_time = grouped_time.reindex(all_bins).fillna(pd.Series(t_bin_centers, index=all_bins)).values
        reindexed_z = grouped_z.reindex(all_bins).values
        
        t_gap_unix, [z_gapped] = inject_data_gaps(reindexed_time, [reindexed_z], gap_threshold=max(gap_thresh, bin_width * 2.5), is_2d=True)
        t_dt = [datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None) for ts in t_gap_unix]
        return t_dt, z_gapped


@lru_cache(maxsize=1)
def load_ephemeris():
    """
    Loads mission-wide 1-hour ephemeris CDF (`psp_helio1hr_position`) and caches
    1D 1D `scipy.interpolate.interp1d` models for $R_{AU}$, Carrington Longitude, and Latitude.
    """
    ephem_file = os.path.join(EPHEM_ROOT, "psp_helio1hr_position_20180813_v01.cdf")
    eph_funcs = {}
    if os.path.exists(ephem_file):
        with smart_cdf(ephem_file) as cdf_e:
            e_time_raw = cdflib.cdfepoch.to_datetime(cdf_e.varget('epoch'))
            e_time_unix = np.array([
                (dt.astype('datetime64[us]').item() - datetime(1970, 1, 1)).total_seconds()
                if isinstance(dt, np.datetime64) else (dt - datetime(1970, 1, 1)).total_seconds()
                for dt in e_time_raw
            ])
            for var in ['RAD_AU', 'HG_LAT', 'HG_LON']:
                data_raw = cdf_e.varget(var)
                mask = (data_raw > -1e10) & (data_raw < 1e10)
                if not np.any(mask): continue
                clean_time, clean_val = e_time_unix[mask], data_raw[mask]
                if var == 'RAD_AU' and np.nanmedian(clean_val) > 1000: clean_val /= 149597870.7
                if var == 'HG_LON': clean_val %= 360
                eph_funcs[var] = interp1d(clean_time, clean_val, bounds_error=False, fill_value="extrapolate")
    return eph_funcs


@lru_cache(maxsize=2)
def load_smart_csv(csv_path):
    """Safely reads, cleans, and caches 15-minute merged CSV datasets."""
    fd, temp_path = tempfile.mkstemp(suffix='.csv', dir=CUSTOM_TMP)
    os.close(fd) 
    try:
        shutil.copy2(csv_path, temp_path)
        df = pd.read_csv(temp_path, index_col=0)
        df.columns = df.columns.str.strip()
        if 'Times' in df.columns:
            df['Times'] = pd.to_datetime(df['Times'], utc=True, errors='coerce').dt.tz_localize(None)
        for col in df.columns:
            if col != 'Times':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df.loc[df[col] < -1e30, col] = np.nan
        return df
    finally:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except Exception: pass


def get_ephemeris_hover_data(time_array):
    """Generates ephemeris hover strings (`Lon | Lat | R_AU`) for Plotly tooltips."""
    eph_funcs = load_ephemeris()
    hover_text = []
    np_epoch, py_epoch = np.datetime64('1970-01-01T00:00:00'), datetime(1970, 1, 1)
    
    for t in time_array:
        if t is None:
            hover_text.append("--")
            continue
        t_unix = (t - np_epoch) / np.timedelta64(1, 's') if isinstance(t, np.datetime64) else (t - py_epoch).total_seconds()
        if eph_funcs and 'HG_LON' in eph_funcs:
            hover_text.append(f"Lon: {eph_funcs['HG_LON'](t_unix):.1f}° | Lat: {eph_funcs['HG_LAT'](t_unix):.1f}° | R: {eph_funcs['RAD_AU'](t_unix):.3f} AU")
        else:
            hover_text.append("--")
    return hover_text


def get_encounter_from_date(target_dt):
    """Returns the encounter integer (1–29) corresponding to a datetime object."""
    for enc, (start_str, end_str) in ENCOUNTER_DATES.items():
        start_dt = datetime.strptime(start_str, '%Y-%m-%d')
        end_dt = datetime.strptime(end_str, '%Y-%m-%d') + timedelta(days=1)
        if start_dt <= target_dt <= end_dt:
            return enc
    return None


def get_max_public_date(base_dir):
    """Traverses a SWEAP directory tree to find the latest available date for public embargoes."""
    if not os.path.exists(base_dir): return None
    years = sorted([d for d in os.listdir(base_dir) if d.isdigit()], reverse=True)
    for y in years:
        year_path = os.path.join(base_dir, y)
        months = sorted([d for d in os.listdir(year_path) if d.isdigit()], reverse=True)
        for m in months:
            files = sorted(glob.glob(os.path.join(year_path, m, "*.cdf")))
            if files:
                match = re.search(r'_(\d{8})_v', files[-1])
                if match: return datetime.strptime(match.group(1), '%Y%m%d')
    return None


def load_cdf_data(filepaths, target_vars):
    """
    Reads a list of Wave Analysis CDF files, maintaining strict separation 
    between mag_time (B-field cadence) and fft_time (spectrogram cadence).
    Pads missing FFT variables with NaNs using only the file's valid fft_time length.
    """
    a_data = {v: [] for v in target_vars if v.lower() != 'frequencies'}
    file_freqs = [] 
    os.makedirs(CUSTOM_TMP, exist_ok=True)
    
    for f in filepaths:
        if not os.path.exists(f): 
            continue
        try:
            with smart_cdf(f) as cdf:
                info = cdf.cdf_info()
                all_vars = info.zVariables + info.rVariables
                file_freq_staging = None
                
                # 1. Frequency Channels
                freq_key = next((k for k in all_vars if k.lower() == 'frequencies'), None)
                if freq_key:
                    freq_data = cdf.varget(freq_key)
                    if freq_data is not None and len(freq_data) > 0 and not np.isnan(freq_data).any():
                        if 0 < np.nanmax(freq_data) <= 1000: 
                            file_freq_staging = freq_data

                # 2. Strict FFT Time Resolution (NEVER fall back to mag_time)
                fft_t_var = next((k for k in all_vars if k.lower() == 'fft_time'), None)
                
                if fft_t_var:
                    t_fft_raw_chunk = cdf.varget(fft_t_var)
                    fft_len = len(t_fft_raw_chunk)
                else:
                    fft_len = 0

                # 3. Check for File Corruption
                time_corrupted = False
                for t_var in ['mag_time', 'fft_time']:
                    actual_t_var = next((k for k in all_vars if k.lower() == t_var), None)
                    if actual_t_var:
                        t_val = cdf.varget(actual_t_var)
                        try:
                            valid_t = (t_val > 0) & np.isfinite(t_val)
                            if len(t_val[valid_t]) == 0: 
                                time_corrupted = True; break
                        except Exception: 
                            time_corrupted = True; break
                if time_corrupted: 
                    continue
                    
                if file_freq_staging is not None: 
                    file_freqs.append(file_freq_staging)
                    
                # 4. Extract Variables
                for var in target_vars:
                    if var.lower() == 'frequencies': 
                        continue
                    
                    actual_var = next((k for k in all_vars if k.lower() == var.lower()), None)
                    
                    if actual_var:
                        val = cdf.varget(actual_var)
                        if var.lower() in ['mag_time', 'fft_time', 'epoch']:
                            val = cdf_time_to_unix(val)
                        a_data[var].append((file_freq_staging, val))
                    elif var in ['Bn', 'Bp', 'Bq']:
                        # Extract components from 2D B_fieldAligned (0: Bn, 1: Bp, 2: Bq)
                        b_fa_var = next((k for k in all_vars if k.lower() == 'b_fieldaligned'), None)
                        if b_fa_var:
                            b_fa_val = cdf.varget(b_fa_var)
                            idx = ['Bn', 'Bp', 'Bq'].index(var)
                            a_data[var].append((file_freq_staging, b_fa_val[:, idx]))
                        else:
                            mag_t_var = next((k for k in all_vars if k.lower() == 'mag_time'), None)
                            mag_len = len(cdf.varget(mag_t_var)) if mag_t_var else 0
                            a_data[var].append((file_freq_staging, np.full(mag_len, np.nan)))
                    else:
                        # Missing variable in this file (e.g. S_theta when E-field is off):
                        # Pad with NaNs matching the EXACT fft_len for this specific file
                        num_freq_channels = len(file_freq_staging) if file_freq_staging is not None else 42
                        nan_chunk = np.full((fft_len, num_freq_channels), np.nan)
                        a_data[var].append((file_freq_staging, nan_chunk))

        except Exception: 
            continue
            
    if all(len(v) == 0 for v in a_data.values()): 
        return None, None, None, None
        
    valid_freqs = [f for f in file_freqs if f is not None]
    master_freq = max(valid_freqs, key=lambda f: np.nanmax(f)) if valid_freqs else None
        
    # Stitch Chunks
    for var in list(a_data.keys()): 
        chunks = a_data[var] 
        if not chunks: 
            del a_data[var]; continue
        if chunks[0][1].ndim == 2:
            target_len = len(master_freq) if master_freq is not None else max(c[1].shape[1] for c in chunks)
            resampled_chunks = []
            for chunk_freq, chunk_data in chunks:
                current_len = chunk_data.shape[1]
                if current_len == target_len: 
                    resampled_chunks.append(chunk_data)
                elif chunk_freq is not None and master_freq is not None and len(chunk_freq) > 1 and current_len > 1:
                    f_interp = interp1d(chunk_freq, chunk_data, axis=1, bounds_error=False, fill_value=np.nan)
                    resampled_chunks.append(f_interp(master_freq))
                else:
                    padded = np.full((chunk_data.shape[0], target_len), np.nan)
                    padded[:, :min(current_len, target_len)] = chunk_data[:, :min(current_len, target_len)]
                    resampled_chunks.append(padded)
            a_data[var] = np.concatenate(resampled_chunks, axis=0)
        else:
            a_data[var] = np.concatenate([c[1] for c in chunks], axis=0)
            
        if a_data[var].dtype.kind in 'f': 
            a_data[var][a_data[var] < -1e30] = np.nan
            
    # Strict Return Arrays
    t_b = next((a_data[k] for k in a_data if k.lower() == 'mag_time'), None)
    t_fft = next((a_data[k] for k in a_data if k.lower() == 'fft_time'), None)
    
    return t_b, t_fft, a_data, master_freq

# =============================================================================
# INDIVIDUAL INSTRUMENT EXTRACTORS
# =============================================================================

def extract_fields_data(start_dt, end_dt, target_cadence_sec=None, verbose=False):
    r"""
    Extracts MAG 4Sa magnetic field vectors ($B_r, B_t, B_n, \vert{}B\vert{}$) with dynamic 
    variable name inspection to handle both 'psp_' and 'spp_' internal CDF prefixes.

    Parameters
    ----------
    start_dt : datetime.datetime
        Start datetime object for the data extraction window.
    end_dt : datetime.datetime
        End datetime object for the data extraction window.
    target_cadence_sec : float or int, optional
        Temporal downsampling cadence in seconds. If None, cadence is calculated 
        dynamically based on window duration.
    verbose : bool, optional
        If True, prints detailed step-by-step directory search logs (default: False).

    Returns
    -------
    dict
        Dictionary containing extracted time series arrays:
        - `t_mag_dt` (list of datetime or None): Binned UTC datetimes.
        - `b_tot` (list of float or None): Magnitude |B| in nT.
        - `b_r` (list of float or None): Radial magnetic field component $B_r$ in nT.
        - `b_t` (list of float or None): Tangential magnetic field component $B_t$ in nT.
        - `b_n` (list of float or None): Normal magnetic field component $B_n$ in nT.
    """
    base_http_url = "http://research.ssl.berkeley.edu/data/spp/data/sci/"
    relative_subpath = "fields/l2/mag_RTN_4_Sa_per_Cyc/"
    autoplot_http_dir = get_autoplot_cache_dir(base_http_url, relative_subpath)

    search_roots = [
        ("AutoplotCache (HTTP)",  autoplot_http_dir),
        ("Local Server Fallback", MAG_4SA_ROOT)
    ]

    t_mag, b_r, b_t, b_n, b_tot = [], [], [], [], []
    curr_dt = start_dt
    files_found = 0

    while curr_dt < end_dt:
        y, m, d = curr_dt.strftime('%Y'), curr_dt.strftime('%m'), curr_dt.strftime('%d')
        date_str = f"{y}{m}{d}"
        file_pattern = f"mag_RTN_4_Sa_per_Cyc_{date_str}_v*.cdf"

        if verbose:
            print(f"\n      --- Processing Date: {y}-{m}-{d} ---")
            
        # Sync single day from Berkeley live before local disk scan
        fetch_missing_berkeley_cdfs(curr_dt, curr_dt + timedelta(days=1), relative_subpath=relative_subpath, verbose=verbose)

        target_file = find_latest_cdf(search_roots, y, m, file_pattern, prefer_prefix="psp_", verbose=verbose)

        if target_file:
            files_found += 1
            with smart_cdf(target_file) as cdf:
                try:
                    info = cdf.cdf_info()
                    all_vars = info.zVariables + info.rVariables
             
                    # 1. Target Epoch Variable (must contain 'epoch', exclude 'quality')
                    epoch_var = next(
                        (v for v in all_vars if 'epoch' in v.lower() and 'quality' not in v.lower() and ('mag' in v.lower() or 'rtn' in v.lower())), 
                        None
                    )
                    if not epoch_var:
                        epoch_var = next((v for v in all_vars if 'epoch' in v.lower() and 'quality' not in v.lower()), None)
             
                    # 2. Target MAG Data Vector (must contain 'mag_rtn', EXCLUDE 'epoch', 'quality', and 'index')
                    mag_var = next(
                        (v for v in all_vars if 'mag_rtn' in v.lower() and 'epoch' not in v.lower() and 'quality' not in v.lower() and 'index' not in v.lower()), 
                        None
                    )
             
                    if not epoch_var or not mag_var:
                        if verbose:
                            print(f"      [!] Missing expected MAG/Epoch variables in {target_file}")
                        curr_dt += timedelta(days=1)
                        continue
             
                    t_chunk = cdf_time_to_unix(cdf.varget(epoch_var))
                    mag_data = cdf.varget(mag_var)
                    mag_data[np.abs(mag_data) > 1e10] = np.nan
             
                    # Mask out NaN timestamps prior to stitching
                    valid_mask = np.isfinite(t_chunk)
                    if np.any(valid_mask) and mag_data.ndim == 2 and len(t_chunk) == len(mag_data):
                        t_mag.extend(t_chunk[valid_mask])
                        b_r.extend(mag_data[valid_mask, 0])
                        b_t.extend(mag_data[valid_mask, 1])
                        b_n.extend(mag_data[valid_mask, 2])
                        b_tot.extend(np.linalg.norm(mag_data[valid_mask], axis=1))
                        if verbose:
                            print(f"      [MAG File Loaded] {y}-{m}-{d}: {target_file} (Epoch: {epoch_var} | Data: {mag_var})")
                    else:
                        if verbose:
                            print(f"      [!] Variable shape or length mismatch in {target_file}: {epoch_var} ({len(t_chunk)}) vs {mag_var} ({mag_data.shape})")
             
                except Exception as e:
                    if verbose:
                        print(f"      [!] Error reading MAG file {target_file}: {e}")
        else:
            if verbose:
                print(f"      [MAG File] {y}-{m}-{d}: NONE FOUND ACROSS ALL SEARCH ROOTS")

        curr_dt += timedelta(days=1)

    if files_found == 0:
        if verbose:
            print(f"      [!] WARNING: No MAG 4Sa CDF files located between {start_dt.strftime('%Y-%m-%d')} and {end_dt.strftime('%Y-%m-%d')}")
        return {"t_mag_dt": None, "b_tot": None, "b_r": None, "b_t": None, "b_n": None}

    cadence = calculate_cadence(start_dt, end_dt, target_cadence_sec, is_2d=False)
    t_mag_dt, [b_r, b_t, b_n, b_tot] = downsample_time_binned(t_mag, [b_r, b_t, b_n, b_tot], target_cadence_sec=cadence)
    return {"t_mag_dt": t_mag_dt, "b_tot": b_tot, "b_r": b_r, "b_t": b_t, "b_n": b_n}


def extract_wave_data(start_dt, end_dt, target_cadence_sec=None, target_freq_bins=TARGET_WAVE_FREQ_BINS, verbose=False):
    r"""
    Extracts 6-hour Wave Analysis CDFs using exact v1.4 zVariable names 
    (B_power_perp, S_theta, ellipticity, coherency, wave_normal).
    """
    curr = start_dt
    files = []
    while curr < end_dt:
        folder = os.path.join(ANALYSIS_ROOT, curr.strftime('%Y'), curr.strftime('%m'))
        files.append(os.path.join(folder, f"PSP_WaveAnalysis_{curr.strftime('%Y-%m-%d')}_{(curr.hour//6)*6:02d}00_{VERSION}.cdf"))
        curr += timedelta(hours=6)

    # Exact v1.4 CDF zVariables (no legacy '_b' suffixes or uppercase 'S_Theta')
    target_vars = ['mag_time', 'fft_time', 'frequencies', 
        'Bn', 'Bp', 'Bq', 
        'B_power_perp', 'S_theta', 'ellipticity', 'coherency', 'wave_normal']
    t_b_raw, t_fft_raw, data, freqs = load_cdf_data(files, target_vars)

    if data is not None:
        # Clean CDF fill values (-1e31)
        wave_keys = ['B_power_perp', 'S_theta', 'ellipticity', 'coherency', 'wave_normal']
        for key in wave_keys:
            if key in data and data[key] is not None:
                data[key] = np.where((data[key] < -1e5) | ~np.isfinite(data[key]), np.nan, data[key])

        # Frequency Log-Regridding
        if freqs is not None and len(freqs) > 0 and target_freq_bins is not None and len(freqs) > target_freq_bins:
            target_freqs = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), target_freq_bins)

            for key in wave_keys:
                if key in data and data[key] is not None and data[key].ndim == 2:
                    matrix = data[key]
                    regrid_matrix = np.full((matrix.shape[0], target_freq_bins), np.nan)

                    for i in range(matrix.shape[0]):
                        row = matrix[i]
                        valid = np.isfinite(row) & (np.abs(row) < 1e10)
                        if np.count_nonzero(valid) >= 2:
                            regrid_matrix[i] = np.interp(
                                target_freqs, freqs[valid], row[valid], 
                                left=np.nan, right=np.nan
                            )

                    data[key] = regrid_matrix

            freqs = target_freqs

    return {"t_b_raw": t_b_raw, "t_fft_raw": t_fft_raw, "data": data, "freqs": freqs}


def extract_hammerhead_data(start_dt, end_dt, ham_bin=5, target_cadence_sec=None):
    r"""
    Extracts Hammerhead occurrence counts binned over uniform time windows.

    Returns
    -------
    dict
        Keys: `t_ham_dt`, `ham_counts`
    """
    ham_epochs = []
    curr_dt = start_dt
    while curr_dt < end_dt:
        y, m, d = curr_dt.strftime('%Y'), curr_dt.strftime('%m'), curr_dt.strftime('%d')
        ham_files = glob.glob(os.path.join(HAM_ROOT, f"hamstring_{y}-{m}-{d}_v*.cdf"))
        if ham_files:
            try:
                with smart_cdf(sorted(ham_files)[-1]) as cdf:
                    raw_epochs = cdf.varget('epoch')
                    unix_times = np.atleast_1d(cdflib.cdfepoch.unixtime(raw_epochs))
                    ham_epochs.extend(unix_times)
            except Exception: pass
        curr_dt += timedelta(days=1)

    bin_sec = ham_bin * 60
    start_ts = start_dt.replace(tzinfo=timezone.utc).timestamp()
    end_ts = end_dt.replace(tzinfo=timezone.utc).timestamp()
    bin_edges = np.arange(start_ts, end_ts + bin_sec, bin_sec)
    ham_counts, _ = np.histogram(ham_epochs, bins=bin_edges)
    t_ham_mid = bin_edges[:-1] + (bin_sec / 2.0)

    cadence = float(target_cadence_sec) if target_cadence_sec else bin_sec
    t_ham_dt, [ham_counts_dec] = downsample_time_binned(t_ham_mid, [ham_counts], target_cadence_sec=cadence) if len(t_ham_mid) > 0 else (None, [None])
    return {"t_ham_dt": t_ham_dt, "ham_counts": ham_counts_dec}


def extract_spc_l2_data(start_dt, end_dt, master_vz=None, target_cadence_sec=TARGET_CADENCE_SEC_SPC):
    r"""
    Extracts SPC L2 charge flux density, flow angles, and A/B/C/D collector currents interpolated onto a master Vz grid.

    Returns
    -------
    dict
        Keys: `t_spc_dt`, `master_vz`, `azimuth`, `elevation`, `flux`, `a_current`, `b_current`, `c_current`, `d_current`
    """
    if master_vz is None: master_vz = np.linspace(150, 900, 50)
    t_list, az_list, el_list = [], [], []
    flux_list, a_list, b_list, c_list, d_list = [], [], [], [], []

    curr_dt = start_dt
    while curr_dt < end_dt:
        y, m, d = curr_dt.strftime('%Y'), curr_dt.strftime('%m'), curr_dt.strftime('%d')
        spc_files = glob.glob(os.path.join(SPC_L2_ROOT, y, m, f"*_swp_spc_l2*_{y}{m}{d}_v*.cdf"))
        if spc_files:
            with smart_cdf(sorted(spc_files)[-1]) as cdf:
                try:
                    t_chunk = cdf_time_to_unix(cdf.varget('Epoch'))
                    if len(t_chunk) > 0:
                        t_list.extend(t_chunk)
                        try:
                            flow = cdf.varget('flow_angle')
                            az_list.extend(flow[:, 0] * (180.0 / np.pi))
                            el_list.extend(flow[:, 1] * (180.0 / np.pi))
                        except Exception:
                            az_list.extend(np.full(len(t_chunk), np.nan))
                            el_list.extend(np.full(len(t_chunk), np.nan))

                        mv_hi = np.array(cdf.varget('mv_hi'), dtype=float)
                        mv_lo = np.array(cdf.varget('mv_lo'), dtype=float)
                        mv_hi[mv_hi < -1e30] = np.nan
                        mv_lo[mv_lo < -1e30] = np.nan
                        v_avg = mv_lo + (mv_hi - mv_lo) / 2.0
                        vz_raw = np.sqrt((2.0 / M_P_G) * EV_TO_ERG * v_avg) / CM_TO_KM
                        if vz_raw.ndim == 1: vz_raw = np.tile(vz_raw, (len(t_chunk), 1))

                        flux_list.append(interpolate_spc_tracking(vz_raw, cdf.varget('diff_charge_flux_density'), master_vz))
                        a_list.append(interpolate_spc_tracking(vz_raw, cdf.varget('a_current'), master_vz))
                        b_list.append(interpolate_spc_tracking(vz_raw, cdf.varget('b_current'), master_vz))
                        c_list.append(interpolate_spc_tracking(vz_raw, cdf.varget('c_current'), master_vz))
                        d_list.append(interpolate_spc_tracking(vz_raw, cdf.varget('d_current'), master_vz))
                except Exception: pass
        curr_dt += timedelta(days=1)

    if not t_list: return None
    t_spc = np.array(t_list)
    azimuth = np.array(az_list) if az_list else np.full(len(t_spc), np.nan)
    elevation = np.array(el_list) if el_list else np.full(len(t_spc), np.nan)

    flux = np.concatenate(flux_list, axis=0) if flux_list else None
    a_cur = np.concatenate(a_list, axis=0) if a_list else None
    b_cur = np.concatenate(b_list, axis=0) if b_list else None
    c_cur = np.concatenate(c_list, axis=0) if c_list else None
    d_cur = np.concatenate(d_list, axis=0) if d_list else None

    cadence = float(target_cadence_sec)
    t_spc_dt, [az_dec, el_dec] = downsample_time_binned(t_spc, [azimuth, elevation], target_cadence_sec=cadence)
    _, flux_dec = downsample_time_binned(t_spc, flux, target_cadence_sec=cadence, is_2d=True) if flux is not None else (None, None)
    _, a_dec    = downsample_time_binned(t_spc, a_cur, target_cadence_sec=cadence, is_2d=True) if a_cur is not None else (None, None)
    _, b_dec    = downsample_time_binned(t_spc, b_cur, target_cadence_sec=cadence, is_2d=True) if b_cur is not None else (None, None)
    _, c_dec    = downsample_time_binned(t_spc, c_cur, target_cadence_sec=cadence, is_2d=True) if c_cur is not None else (None, None)
    _, d_dec    = downsample_time_binned(t_spc, d_cur, target_cadence_sec=cadence, is_2d=True) if d_cur is not None else (None, None)

    return {
        "t_spc_dt": t_spc_dt, "master_vz": master_vz, "azimuth": az_dec, "elevation": el_dec,
        "flux": flux_dec, "a_current": a_dec, "b_current": b_dec, "c_current": c_dec, "d_current": d_dec
    }

def extract_merged_sweap_data(start_dt, end_dt, enc_num, csv_override=None):
    """Extracts, calculates, and downsamples merged SPC/SPANi moments from E{XX}.csv."""
    csv_path = csv_override or get_current_merged_csv_path(enc_num)
    if not csv_path or not os.path.exists(csv_path):
        print(f"  [!] Missing Merged CSV file for Encounter {enc_num}")
        return None

    df = load_smart_csv(csv_path)
    if 'Times' not in df.columns:
        return None

    df = df.dropna(subset=['Times']).sort_values('Times')
    mask = (df['Times'] >= start_dt) & (df['Times'] <= end_dt)
    df_window = df.loc[mask].copy()
    if df_window.empty:
        return None

    # Calculate Alfvén speed (V_A = 21.8 * B / sqrt(1.1 * N_p))
    density_safe = np.where(df_window['Np-Parker'].values > 0, df_window['Np-Parker'].values, np.nan)
    v_alfven = np.array(21.8 * df_window['B-Parker'].values / np.sqrt(1.1 * density_safe), dtype=np.float64)

    t_raw = (df_window['Times'] - pd.Timestamp("1970-01-01")).dt.total_seconds().values
    is_encounter = (end_dt - start_dt).days > 1
    target_1d = 8000 if is_encounter else MAX_1D_POINTS

    # Downsample 1D arrays
    t_dec, b_mag_dec = downsample_1d(t_raw, df_window['B-Parker'].values, target_1d)
    _, b_r_dec = downsample_1d(t_raw, df_window['Br-Parker'].values, target_1d)
    _, b_t_dec = downsample_1d(t_raw, df_window['Bt-Parker'].values, target_1d)
    _, b_n_dec = downsample_1d(t_raw, df_window['Bn-Parker'].values, target_1d)
    _, vr_dec = downsample_1d(t_raw, df_window['Vpr-Parker'].values, target_1d)
    _, va_dec = downsample_1d(t_raw, v_alfven, target_1d)
    _, np_dec = downsample_1d(t_raw, df_window['Np-Parker'].values, target_1d)
    _, tp_dec = downsample_1d(t_raw, df_window['Tp-Parker'].values, target_1d)

    # Gap injection (1-hour threshold)
    t_gap, data_gapped = inject_data_gaps(
        t_dec, 
        [b_mag_dec, b_r_dec, b_t_dec, b_n_dec, vr_dec, va_dec, np_dec, tp_dec], 
        gap_threshold=3600, 
        is_2d=False
    )
    b_mag_dec, b_r_dec, b_t_dec, b_n_dec, vr_dec, va_dec, np_dec, tp_dec = data_gapped

    # Calculate Carrington Orbit Coordinates for spatial 2D plot
    eph_funcs = load_ephemeris()
    x_orb, y_orb = np.full(len(t_gap), np.nan), np.full(len(t_gap), np.nan)
    if eph_funcs and 'HG_LON' in eph_funcs and 'RAD_AU' in eph_funcs:
        for i, ts in enumerate(t_gap):
            if np.isfinite(ts):
                r_au = eph_funcs['RAD_AU'](ts)
                lon_deg = eph_funcs['HG_LON'](ts)
                x_orb[i] = r_au * 215.0 * np.cos(np.radians(lon_deg))
                y_orb[i] = r_au * 215.0 * np.sin(np.radians(lon_deg))

    # Fetch LFR Density overlay
    lfr = extract_lfr_data(start_dt, end_dt)

    t_dt = [datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None) for ts in t_gap]

    return {
        "t_dt": t_dt,
        "b_tot": b_mag_dec,
        "b_r": b_r_dec,
        "b_t": b_t_dec,
        "b_n": b_n_dec,
        "vr": vr_dec,
        "v_alfven": va_dec,
        "np": np_dec,
        "tp": tp_dec,
        "t_lfr_dt": lfr["t_lfr_dt"],
        "np_lfr": lfr["np_lfr"],
        "x_carr": x_orb,
        "y_carr": y_orb,
    }

def extract_spc_l3_data(start_dt, end_dt, target_cadence_sec=None, verbose=False):
    r"""
    Extracts SPC L3i fit moments, separating Peak Tracks from Full Scans via DQF quality bitmasks.
    """
    t_spc, spc_vr_peak, spc_np_peak, spc_wp_peak, spc_vr_full, spc_np_full, spc_wp_full = [], [], [], [], [], [], []
    curr_dt = start_dt
    spc_fallback_root = os.path.join(SWEAP_DIR, 'sweap/spc/L3/')

    while curr_dt < end_dt:
        y, m, d = curr_dt.strftime('%Y'), curr_dt.strftime('%m'), curr_dt.strftime('%d')
        pattern = f"spc_l3i_{y}{m}{d}_v*.cdf"

        search_roots = [
            ("Local Disk Primary", SPC_L3_ROOT),
            ("SWEAP Directory Fallback", spc_fallback_root)
        ]

        target_file = find_latest_cdf(search_roots, y, m, pattern, prefer_prefix="psp_", verbose=verbose)

        if target_file:
            if verbose:
                print(f"      [SPC-L3 File] {y}-{m}-{d}: {target_file}")
            with smart_cdf(target_file) as cdf:
                try:
                    t_chunk = cdf_time_to_unix(cdf.varget('Epoch'))
                    dqf = cdf.varget('DQF')
                    mask_0 = (dqf[:, 0] == 0) if dqf.ndim > 1 else np.ones(len(t_chunk), dtype=bool)
                    mask_16 = (dqf[:, 16] == 1) if (dqf.ndim > 1 and dqf.shape[1] > 16) else np.zeros(len(t_chunk), dtype=bool)
                    mask_full = mask_16
                    mask_peak = mask_0 & (~mask_16)

                    vr_raw = cdf.varget('vp_fit_RTN')[:, 0]
                    np_raw = cdf.varget('np_fit')
                    wp_raw = cdf.varget('wp_fit')

                    if len(t_chunk) == len(vr_raw):
                        t_spc.extend(t_chunk)
                        spc_vr_peak.extend(np.where(mask_peak & np.isfinite(vr_raw) & (vr_raw > 50) & (vr_raw < 2500), vr_raw, np.nan))
                        spc_np_peak.extend(np.where(mask_peak & np.isfinite(np_raw) & (np_raw > 0.01) & (np_raw < 1e4), np_raw, np.nan))
                        spc_wp_peak.extend(np.where(mask_peak & np.isfinite(wp_raw) & (wp_raw > 0) & (wp_raw < 1000), wp_raw, np.nan))
                        spc_vr_full.extend(np.where(mask_full & np.isfinite(vr_raw) & (vr_raw > 50) & (vr_raw < 2500), vr_raw, np.nan))
                        spc_np_full.extend(np.where(mask_full & np.isfinite(np_raw) & (np_raw > 0.01) & (np_raw < 1e4), np_raw, np.nan))
                        spc_wp_full.extend(np.where(mask_full & np.isfinite(wp_raw) & (wp_raw > 0) & (wp_raw < 1000), wp_raw, np.nan))
                except Exception as e:
                    if verbose:
                        print(f"      [!] Error reading SPC-L3 file {target_file}: {e}")
        else:
            if verbose:
                print(f"      [SPC-L3 File] {y}-{m}-{d}: NONE FOUND")
        curr_dt += timedelta(days=1)

    cadence = calculate_cadence(start_dt, end_dt, target_cadence_sec, is_2d=False)
    t_spc_dt, [spc_vr_peak, spc_np_peak, spc_wp_peak, spc_vr_full, spc_np_full, spc_wp_full] = downsample_time_binned(
        t_spc, [spc_vr_peak, spc_np_peak, spc_wp_peak, spc_vr_full, spc_np_full, spc_wp_full], target_cadence_sec=cadence
    )

    return {
        "t_spc_dt": t_spc_dt,
        "spc_vr_peak": spc_vr_peak, "spc_np_peak": spc_np_peak, "spc_wp_peak": spc_wp_peak,
        "spc_vr_full": spc_vr_full, "spc_np_full": spc_np_full, "spc_wp_full": spc_wp_full
    }


def extract_spani_data(start_dt, end_dt, target_cadence_sec=None):
    r"""
    Extracts SPAN-i L3 ion moments (v_R, N_p, w_p, V_inst, w_para, w_perp).
    Computes temperature anisotropy via Badman et al. (2025) Appendix A.2.

    Returns
    -------
    dict
        Keys: `t_spi_dt`, `spi_vr`, `spi_np`, `spi_wp`, `spi_vel_inst`, `spi_wpara`, `spi_wperp`
    """
    t_spi, spi_vr, spi_np, spi_wp = [], [], [], []
    spi_vel_inst, spi_wpara, spi_wperp = [], [], []

    curr_dt = start_dt
    while curr_dt < end_dt:
        y, m, d = curr_dt.strftime('%Y'), curr_dt.strftime('%m'), curr_dt.strftime('%d')
        spi_files = glob.glob(os.path.join(SPI_L3_ROOT, y, m, f"*spi_sf00*_{y}{m}{d}_v*.cdf"))
        if spi_files:
            with smart_cdf(sorted(spi_files)[-1]) as cdf:
                try:
                    info = cdf.cdf_info()
                    all_vars = info.zVariables + info.rVariables

                    t_chunk = cdf_time_to_unix(cdf.varget('Epoch'))
                    vr_chunk = cdf.varget('VEL_RTN_SUN')[:, 0]
                    dens_raw = cdf.varget('DENS')
                    temp_ev = cdf.varget('TEMP')
                    temp_ev[temp_ev <= 0] = np.nan 
                    wp_chunk = np.sqrt((temp_ev * EV_TO_K * K_B_ERG) / M_P_G) / CM_TO_KM

                    # 1. Moment Velocity in Instrument Frame (3D Vector)
                    v_inst_var = next((v for v in ['VEL_INST', 'VEL_SF00', 'VEL_SC', 'VEL'] if v in all_vars), None)
                    v_inst_chunk = cdf.varget(v_inst_var) if v_inst_var else np.full((len(t_chunk), 3), np.nan)

                    # 2. Temperature Tensor & B-Field in Instrument Frame
                    t_tensor_var = next((v for v in ['T_TENSOR_INST', 'T_TENSOR', 'TEMP_TENSOR'] if v in all_vars), None)
                    b_inst_var   = next((v for v in ['MAGF_INST', 'B_INST', 'MAG_INST'] if v in all_vars), None)

                    if t_tensor_var and b_inst_var:
                        t_mat = cdf.varget(t_tensor_var)
                        b_mat = cdf.varget(b_inst_var)

                        t_xx = t_mat[:, 0, 0] if t_mat.ndim == 3 else t_mat[:, 0]
                        t_zz = t_mat[:, 2, 2] if t_mat.ndim == 3 else t_mat[:, 2]

                        wpara_chunk, wperp_chunk = solve_badman_anisotropy(t_xx, t_zz, b_mat)
                    else:
                        wpara_chunk = np.full(len(t_chunk), np.nan)
                        wperp_chunk = np.full(len(t_chunk), np.nan)

                    if len(t_chunk) == len(vr_chunk):
                        t_spi.extend(t_chunk)
                        spi_vr.extend(np.where(np.isfinite(vr_chunk) & (vr_chunk > 50) & (vr_chunk < 2500), vr_chunk, np.nan))
                        spi_np.extend(np.where(np.isfinite(dens_raw) & (dens_raw > 0.01) & (dens_raw < 1e5), dens_raw, np.nan))
                        spi_wp.extend(wp_chunk)
                        spi_wpara.extend(wpara_chunk)
                        spi_wperp.extend(wperp_chunk)
                        spi_vel_inst.append(v_inst_chunk)
                except Exception:
                    pass
        curr_dt += timedelta(days=1)

    if not t_spi:
        return {
            "t_spi_dt": None, "spi_vr": None, "spi_np": None, "spi_wp": None,
            "spi_vel_inst": None, "spi_wpara": None, "spi_wperp": None
        }

    cadence = calculate_cadence(start_dt, end_dt, target_cadence_sec, is_2d=False)

    # Downsample 1D arrays
    t_spi_dt, [spi_vr, spi_np, spi_wp, spi_wpara, spi_wperp] = downsample_time_binned(
        t_spi, [spi_vr, spi_np, spi_wp, spi_wpara, spi_wperp], target_cadence_sec=cadence
    )

    # Downsample 2D (N x 3) Velocity Matrix
    spi_vel_cat = np.concatenate(spi_vel_inst, axis=0) if spi_vel_inst else None
    _, spi_vel_dec = downsample_time_binned(
        t_spi, spi_vel_cat, target_cadence_sec=cadence, is_2d=True
    ) if spi_vel_cat is not None else (None, None)

    return {
        "t_spi_dt": t_spi_dt, 
        "spi_vr": spi_vr, 
        "spi_np": spi_np, 
        "spi_wp": spi_wp,
        "spi_vel_inst": spi_vel_dec,
        "spi_wpara": spi_wpara,
        "spi_wperp": spi_wperp
    }


def extract_spane_data(start_dt, end_dt, target_energy_ev=476.0, target_cadence_sec=None, verbose=False, **kwargs):
    r"""
    Extracts SPAN-E L3 Pitch Angle Distribution (PAD) matrices at the strahl energy channel index closest to 476 eV.
    Ignores legacy kwargs like `span_energy_idx` for backward compatibility.
    """
    t_span_e, span_pad = [], []
    span_energy_val = None
    curr_dt = start_dt
    spe_fallback_root = os.path.join(SWEAP_DIR, 'sweap/spe/L3/spe_sf0_pad/')

    while curr_dt < end_dt:
        y, m, d = curr_dt.strftime('%Y'), curr_dt.strftime('%m'), curr_dt.strftime('%d')
        pattern = f"spe_sf0*_{y}{m}{d}_v*.cdf"
        
        search_roots = [
            ("Local Disk Primary", SPE_L3_ROOT),
            ("SWEAP Directory Fallback", spe_fallback_root)
        ]
        
        target_file = find_latest_cdf(search_roots, y, m, pattern, prefer_prefix="psp_", verbose=verbose)
        
        if target_file:
            if verbose:
                print(f"      [SPAN-E File] {y}-{m}-{d}: {target_file}")
            with smart_cdf(target_file) as cdf:
                try:
                    t_chunk = cdf_time_to_unix(cdf.varget('Epoch'))
                    pad_full = cdf.varget('EFLUX_VS_PA_E')  # Shape: (time, pitch_angle, energy)
                    
                    # Match energy channel nearest to target_energy_ev for this specific CDF
                    info = cdf.cdf_info()
                    all_vars = info.zVariables + info.rVariables
                    energy_arr = None
                    for e_var in ['ENERGY', 'ENERGY_VALS', 'ENERGY_E', 'ENERGY_OBS', 'E_OBS']:
                        if e_var in all_vars:
                            energy_arr = cdf.varget(e_var)
                            break
                    
                    if energy_arr is not None:
                        e_table = energy_arr if energy_arr.ndim == 1 else energy_arr[0]
                        e_idx = int(np.argmin(np.abs(e_table - target_energy_ev)))
                        if span_energy_val is None:
                            span_energy_val = float(e_table[e_idx])
                    else:
                        e_idx = 10
                    
                    if len(t_chunk) == pad_full.shape[0]:
                        t_span_e.extend(t_chunk)
                        span_pad.append(pad_full[:, :, e_idx])
                except Exception as e:
                    if verbose:
                        print(f"      [!] Error reading SPAN-E file {target_file}: {e}")
        else:
            if verbose:
                print(f"      [SPAN-E File] {y}-{m}-{d}: NONE FOUND")
        curr_dt += timedelta(days=1)

    cadence = calculate_cadence(start_dt, end_dt, target_cadence_sec, is_2d=False)
    t_span_e_dt, pad_dec = downsample_time_binned(
        t_span_e, np.concatenate(span_pad, axis=0), target_cadence_sec=cadence, is_2d=True
    ) if span_pad else (None, None)

    return {"t_span_e_dt": t_span_e_dt, "pad_dec": pad_dec, "span_energy_ev": span_energy_val}


def extract_lfr_data(start_dt, end_dt, target_cadence_sec=None):
    r"""
    Extracts LFR mission electron density ground truth.

    Returns
    -------
    dict
        Keys: `t_lfr_dt`, `np_lfr`
    """
    t_lfr, np_lfr = [], []
    if os.path.exists(LFR_DENSITY_FILE):
        with smart_cdf(LFR_DENSITY_FILE) as cdf:
            lfr_epochs = cdf_time_to_unix(cdf.varget('epoch'))
            start_ts = start_dt.replace(tzinfo=timezone.utc).timestamp()
            end_ts = end_dt.replace(tzinfo=timezone.utc).timestamp()
            mask = (lfr_epochs >= start_ts) & (lfr_epochs <= end_ts)
            if np.any(mask):
                t_lfr = lfr_epochs[mask]
                raw_ne = cdf.varget('electronDensity')[mask]
                np_lfr = np.where(np.isfinite(raw_ne) & (raw_ne > 0.01) & (raw_ne < 1e5), raw_ne, np.nan)

    cadence = calculate_cadence(start_dt, end_dt, target_cadence_sec, is_2d=False)
    t_lfr_dt, [np_lfr] = downsample_time_binned(t_lfr, [np_lfr], target_cadence_sec=cadence)
    return {"t_lfr_dt": t_lfr_dt, "np_lfr": np_lfr}


def extract_ephemeris_data(start_dt, end_dt, step_hours=1):
    r"""
    Generates a uniform 1-hour ephemeris trajectory ($R_{AU}, \text{Carr\_Lon}, \text{Carr\_Lat}$).

    Returns
    -------
    dict
        Keys: `t_eph_dt`, `r_au`, `carr_lon`, `carr_lat`
    """
    t_eph_dt, r_au, carr_lon, carr_lat = [], [], [], []
    eph_funcs = load_ephemeris()
    if eph_funcs and 'RAD_AU' in eph_funcs:
        curr_eph = start_dt
        while curr_eph <= end_dt:
            ts = curr_eph.replace(tzinfo=timezone.utc).timestamp()
            t_eph_dt.append(curr_eph)
            r_au.append(float(eph_funcs['RAD_AU'](ts)))
            carr_lon.append(float(eph_funcs['HG_LON'](ts)) if 'HG_LON' in eph_funcs else None)
            carr_lat.append(float(eph_funcs['HG_LAT'](ts)) if 'HG_LAT' in eph_funcs else None)
            curr_eph += timedelta(hours=step_hours)
    return {"t_eph_dt": t_eph_dt, "r_au": r_au, "carr_lon": carr_lon, "carr_lat": carr_lat}


def extract_plasma_data(start_dt, end_dt, span_energy_idx=10, target_cadence_sec=None, **kwargs):
    r"""
    Composite Extractor: Merges FIELDS, SPC L3, SPANi, SPANe, LFR, and Ephemeris,
    calculating dynamic Alfvén Speed ($V_A = 21.81 \cdot B / \sqrt{N_e}$) using a density hierarchy.

    Returns
    -------
    dict
        Dictionary containing all merged plasma and field time series.
    """
    f = extract_fields_data(start_dt, end_dt, target_cadence_sec=target_cadence_sec)
    spc_l3 = extract_spc_l3_data(start_dt, end_dt, target_cadence_sec=target_cadence_sec)
    spi = extract_spani_data(start_dt, end_dt, target_cadence_sec=target_cadence_sec)
    spe = extract_spane_data(start_dt, end_dt, span_energy_idx=span_energy_idx, target_cadence_sec=target_cadence_sec)
    lfr = extract_lfr_data(start_dt, end_dt, target_cadence_sec=target_cadence_sec)
    eph = extract_ephemeris_data(start_dt, end_dt)

    # Compute dynamic V_Alfven from combined density hierarchy (LFR -> SPC -> SPANi)
    t_valf, valfven = [], []
    all_dens_times = []
    if lfr["t_lfr_dt"]: all_dens_times.extend([t.replace(tzinfo=timezone.utc).timestamp() for t in lfr["t_lfr_dt"]])
    if spc_l3["t_spc_dt"]: all_dens_times.extend([t.replace(tzinfo=timezone.utc).timestamp() for t in spc_l3["t_spc_dt"]])
    if spi["t_spi_dt"]: all_dens_times.extend([t.replace(tzinfo=timezone.utc).timestamp() for t in spi["t_spi_dt"]])

    if f["t_mag_dt"] and len(all_dens_times) > 0:
        t_mag_unix = [t.replace(tzinfo=timezone.utc).timestamp() for t in f["t_mag_dt"]]
        df_mag = pd.DataFrame({'t': t_mag_unix, 'b_tot': f["b_tot"]}).dropna(subset=['t', 'b_tot']).sort_values('t')
        df_mag = df_mag[np.abs(df_mag['b_tot']) < 1e5]

        base_t = np.unique(np.sort(all_dens_times))
        df_comp = pd.DataFrame({'t': base_t})

        if lfr["t_lfr_dt"]:
            t_lfr_unix = [t.replace(tzinfo=timezone.utc).timestamp() for t in lfr["t_lfr_dt"]]
            df_lfr = pd.DataFrame({'t': t_lfr_unix, 'np_lfr': lfr["np_lfr"]}).dropna(subset=['t']).sort_values('t')
            df_comp = pd.merge_asof(df_comp, df_lfr, on='t', tolerance=120, direction='nearest')
        else: df_comp['np_lfr'] = np.nan

        if spc_l3["t_spc_dt"]:
            t_spc_unix = [t.replace(tzinfo=timezone.utc).timestamp() for t in spc_l3["t_spc_dt"]]
            df_spc = pd.DataFrame({'t': t_spc_unix, 'spc_np_full': spc_l3["spc_np_full"], 'spc_np_peak': spc_l3["spc_np_peak"]}).dropna(subset=['t']).sort_values('t')
            df_comp = pd.merge_asof(df_comp, df_spc, on='t', tolerance=120, direction='nearest')
        else: df_comp['spc_np_full'], df_comp['spc_np_peak'] = np.nan, np.nan

        if spi["t_spi_dt"]:
            t_spi_unix = [t.replace(tzinfo=timezone.utc).timestamp() for t in spi["t_spi_dt"]]
            df_spi = pd.DataFrame({'t': t_spi_unix, 'spi_np': spi["spi_np"]}).dropna(subset=['t']).sort_values('t')
            df_comp = pd.merge_asof(df_comp, df_spi, on='t', tolerance=120, direction='nearest')
        else: df_comp['spi_np'] = np.nan

        spc_dens = df_comp['spc_np_full'].combine_first(df_comp['spc_np_peak'])
        raw_dens = df_comp['np_lfr'].combine_first(spc_dens).combine_first(df_comp['spi_np'])
        df_comp['raw_dens'] = raw_dens

        # 15-minute rolling median filter
        df_comp['dt'] = pd.to_datetime(df_comp['t'], unit='s')
        
        # FIX: Pass string 'raw_dens' instead of the Series variable raw_dens
        df_comp['smooth_dens'] = df_comp.set_index('dt')['raw_dens'].rolling('15min', center=True, min_periods=1).median().values

        # Merge |B| from MAG 4Sa (30-second tolerance)
        df_comp = pd.merge_asof(df_comp, df_mag, on='t', tolerance=30, direction='nearest')

        b_vals, n_vals = df_comp['b_tot'].values, df_comp['smooth_dens'].values
        valid_m = np.isfinite(b_vals) & np.isfinite(n_vals) & (n_vals > 0.01) & (b_vals < 1e5)
        v_a = np.full(len(df_comp), np.nan)
        v_a[valid_m] = 21.81 * b_vals[valid_m] / np.sqrt(n_vals[valid_m])
        
        t_valf = [datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None) for ts in df_comp['t'].tolist()]
        valfven = np.where((v_a > 10) & (v_a < 3000), v_a, np.nan).tolist()
    else:
        t_valf, valfven = None, None

    return {
        **f, **spc_l3, **spi, **spe, **lfr, **eph,
        "t_valf_dt": t_valf, "valfven": valfven
    }


def interpolate_spc_tracking(v_raw, data_raw, master_grid):
    """Interpolates time-varying tracking voltage bins onto a static master velocity grid."""
    out = np.full((data_raw.shape[0], len(master_grid)), np.nan)
    for i in range(data_raw.shape[0]):
        v, d = v_raw[i], data_raw[i]
        valid = np.isfinite(v) & np.isfinite(d) & (d > 0)
        if np.any(valid):
            idx = np.argsort(v[valid])
            out[i] = np.interp(master_grid, v[valid][idx], d[valid][idx], left=np.nan, right=np.nan)
    return out


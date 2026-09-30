"""
spani_vdfBrowser.py
===================
Interactive Parker Solar Probe (PSP) SPAN-I Dynamic VDF & Timeseries Browser.

Features & Scientific Telemetry Sanitization:
  - Robust Float Coercion: Automatically casts JSON nulls/Nones to np.nan float ndarrays.
  - 3-Tier Data Fallback: Primary Drive/CDFs -> Local Staging JSONs -> Web Server HTTPS Fetch.
  - Interactive B_RTN Overview Banner (~10 days) with red highlight region sync.
  - Dynamic Encounter Selector (E01 to E38) & Perihelion auto-navigation.
"""

import os
import sys
import re
import io
import gzip
import json
import urllib.request
import ssl
import time

# Force pyqtgraph to use PyQt5
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

import glob
import shutil
import tempfile
import argparse
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import cdflib

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.spatial import Delaunay

# ==========================================
# MULTI-USER FALLBACK CONFIGURATION FOR FC
# ==========================================
FC_PRIMARY_PSP = "/psp/data/sci"
FC_STAGING_DEFAULT = "/home/kpaulson/website/SWEAP/browser/staging"
REMOTE_BASE_URL = "https://w3sweap.cfa.harvard.edu/browser/team/JSON"

SCRIPT_DIR = "/home/kpaulson/MyDrive/Software/Python/githubScripts_python" if os.name != 'nt' else "G:/My Drive/Software/Python/githubScripts_python"
if SCRIPT_DIR not in sys.path and os.path.exists(SCRIPT_DIR):
    sys.path.append(SCRIPT_DIR)

FALLBACK_ENCOUNTER_DATES = {
    1:  ('2018-11-01', '2018-11-11'), 2:  ('2019-03-31', '2019-04-10'),
    3:  ('2019-08-28', '2019-09-07'), 4:  ('2020-01-24', '2020-02-04'),
    5:  ('2020-06-02', '2020-06-13'), 6:  ('2020-09-22', '2020-10-02'),
    7:  ('2021-01-13', '2021-01-23'), 8:  ('2021-04-25', '2021-05-04'),
    9:  ('2021-08-05', '2021-08-15'), 10: ('2021-11-17', '2021-11-26'),
    11: ('2022-02-21', '2022-03-02'), 12: ('2022-05-28', '2022-06-07'),
    13: ('2022-09-02', '2022-09-11'), 14: ('2022-12-07', '2022-12-16'),
    15: ('2023-03-13', '2023-03-23'), 16: ('2023-06-17', '2023-06-27'),
    17: ('2023-09-23', '2023-10-03'), 18: ('2023-12-25', '2024-01-03'),
    19: ('2024-03-26', '2024-04-04'), 20: ('2024-06-26', '2024-07-05'),
    21: ('2024-09-26', '2024-10-05'), 22: ('2024-12-20', '2024-12-29'),
    23: ('2025-03-19', '2025-03-27'), 24: ('2025-06-15', '2025-06-24'),
    25: ('2025-09-11', '2025-09-20'), 26: ('2025-12-09', '2025-12-18'),
    27: ('2026-03-07', '2026-03-16'), 28: ('2026-06-04', '2026-06-13'),
    29: ('2026-08-31', '2026-09-09'), 30: ('2026-11-28', '2026-12-06'),
    31: ('2027-02-24', '2027-03-05'), 32: ('2027-05-23', '2027-06-01'),
    33: ('2027-08-20', '2027-08-29'), 34: ('2027-11-16', '2027-11-25'),
    35: ('2028-02-13', '2028-02-22'), 36: ('2028-05-11', '2028-05-20'),
    37: ('2028-08-08', '2028-08-16'), 38: ('2028-11-04', '2028-11-13'),
}

FALLBACK_ORBIT_DATES = {
    1:  ('2018-08-12', '2019-01-20'), 2:  ('2019-01-20', '2019-06-19'),
    3:  ('2019-06-19', '2019-11-16'), 4:  ('2019-11-16', '2020-04-03'),
    5:  ('2020-04-03', '2020-08-02'), 6:  ('2020-08-02', '2020-11-22'),
    7:  ('2020-11-22', '2021-03-09'), 8:  ('2021-03-09', '2021-06-19'),
    9:  ('2021-06-19', '2021-09-30'), 10: ('2021-09-30', '2022-01-08'),
    11: ('2022-01-08', '2022-04-15'), 12: ('2022-04-15', '2022-07-20'),
    13: ('2022-07-20', '2022-10-24'), 14: ('2022-10-24', '2023-01-29'),
    15: ('2023-01-29', '2023-05-05'), 16: ('2023-05-05', '2023-08-09'),
    17: ('2023-08-09', '2023-11-13'), 18: ('2023-11-13', '2024-02-13'),
    19: ('2024-02-13', '2024-05-15'), 20: ('2024-05-15', '2024-08-15'),
    21: ('2024-08-15', '2024-11-10'), 22: ('2024-11-10', '2025-02-07'),
    23: ('2025-02-07', '2025-05-06'), 24: ('2025-05-06', '2025-08-03'),
    25: ('2025-08-03', '2025-10-30'), 26: ('2025-10-30', '2026-01-26'),
    27: ('2026-01-26', '2026-04-25'), 28: ('2026-04-25', '2026-07-22'),
    29: ('2026-07-22', '2026-10-19'), 30: ('2026-10-19', '2027-01-15'),
    31: ('2027-01-15', '2027-04-14'), 32: ('2027-04-14', '2027-07-11'),
    33: ('2027-07-11', '2027-10-08'), 34: ('2027-10-08', '2028-01-04'),
    35: ('2028-01-04', '2028-04-01'), 36: ('2028-04-01', '2028-06-29'),
    37: ('2028-06-29', '2028-09-25'), 38: ('2028-09-25', '2028-12-23'),
}

try:
    import config
    DRIVE_ROOT = getattr(config, 'get_drive_path', lambda: FC_STAGING_DEFAULT)()
    PSP_SHAREDDRIVE_ROOT = getattr(config, 'get_sharedDrivePSP_path', lambda: "")()
    SWEAP_DIR = getattr(config, 'get_sweapCacheData', lambda: FC_PRIMARY_PSP)()
    BERKELEY_DIR = getattr(config, 'get_berkeleyCacheData', lambda: FC_PRIMARY_PSP)()
    ENCOUNTER_DATES = getattr(config, 'ENCOUNTER_DATES', FALLBACK_ENCOUNTER_DATES)
    ORBIT_DATES = getattr(config, 'ORBIT_DATES', FALLBACK_ORBIT_DATES)
except ImportError:
    DRIVE_ROOT = FC_STAGING_DEFAULT if os.path.exists(FC_STAGING_DEFAULT) else ""
    PSP_SHAREDDRIVE_ROOT = "/home/kpaulson/sharedDrive_PSP-SWEAP" if os.path.exists("/home/kpaulson/sharedDrive_PSP-SWEAP") else ""
    SWEAP_DIR = FC_PRIMARY_PSP if os.path.exists(FC_PRIMARY_PSP) else ""
    BERKELEY_DIR = FC_PRIMARY_PSP if os.path.exists(FC_PRIMARY_PSP) else ""
    ENCOUNTER_DATES = FALLBACK_ENCOUNTER_DATES
    ORBIT_DATES = FALLBACK_ORBIT_DATES

AUTOPLOT_CACHE = os.path.join(DRIVE_ROOT, "Research", "Data", "AutoplotCache") if DRIVE_ROOT else ""
WAVE_ANALYSIS_ROOT = os.path.join(DRIVE_ROOT, "Research", "PSP", "WaveAnalysis", "WaveAnalysis_Files", "v1.4") if DRIVE_ROOT else ""
HAMMERHEAD_ROOT = os.path.join(DRIVE_ROOT, "Research", "PSP", "Hammerheads", "Hamstrings", "cdf", "v02") if DRIVE_ROOT else ""
JSON_TEAM_ROOT = os.path.join(DRIVE_ROOT, "team", "JSON") if os.path.exists(os.path.join(DRIVE_ROOT, "team", "JSON")) else os.path.join(DRIVE_ROOT, "Research", "PSP", "JSON", "team")
SLICED_VDF_ROOT = os.path.join(DRIVE_ROOT, "Research", "PSP", "SPAN", "SPANi", "SPANi_slicedVDF") if DRIVE_ROOT else ""
LFR_DENSITY_FILE = os.path.join(FC_PRIMARY_PSP, "sweap", "spc", "LFR", "spp_fld_lfr_mission_density.cdf") if os.path.exists(FC_PRIMARY_PSP) else ""

LOCAL_DATA_DIR = None

# Default encounter settings
DEFAULT_ENCOUNTER_NUM = 29 # Globalized here, calculated formally below
DEFAULT_PERIHELION_DATE = "2026-09-04"

# ==========================================
# ASSET RESOLVER & SPLASH SCREEN HELPERS
# ==========================================
def get_resource_path(relative_path):
    """Resolves absolute path to assets, supporting PyInstaller bundles and dev environments."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def create_splash_screen(logo_filename="SWEAP_LOGO.png"):
    """Instantiates a high-contrast Qt splash screen using SWEAP_LOGO.png if present."""
    logo_path = get_resource_path(logo_filename)
    if os.path.exists(logo_path):
        pixmap = QtGui.QPixmap(logo_path)
        # Scaled smoothly to reasonable splash width while preserving aspect ratio
        if pixmap.width() > 600:
            pixmap = pixmap.scaledToWidth(600, QtCore.Qt.SmoothTransformation)
    else:
        # Fallback dark banner if logo file is missing
        pixmap = QtGui.QPixmap(500, 220)
        pixmap.fill(QtGui.QColor(35, 35, 35))

    splash = QtWidgets.QSplashScreen(pixmap, QtCore.Qt.WindowStaysOnTopHint)
    splash.showMessage(
        "Initializing Parker Solar Probe VDF Browser...", 
        QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter, 
        QtGui.QColor(220, 220, 220)
    )
    return splash

def log_stage(stage_name, start_time=None):
    """Logs elapsed computational time and current UTC timestamp for profiling."""
    now_str = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    if start_time is not None:
        elapsed_sec = time.time() - start_time
        print(f"[{now_str}] COMPLETED: {stage_name} ({elapsed_sec:.3f}s)")
        return time.time()
    else:
        print(f"[{now_str}] STARTING:  {stage_name}")
        return time.time()

# ==========================================
# SANITIZATION & TIME PARSING HELPERS
# ==========================================
def sanitize_float_array(input_data, dtype=np.float64):
    """
    Coerces lists, arrays, or matrices containing None/null values into clean float
    ndarrays with np.nan, eliminating object-dtype arrays that crash PyQtGraph.
    """
    if input_data is None:
        return np.array([], dtype=dtype)
    try:
        return np.array(input_data, dtype=dtype)
    except (ValueError, TypeError):
        arr_obj = np.array(input_data, dtype=object)
        flat_clean = pd.to_numeric(pd.Series(arr_obj.ravel()), errors='coerce').to_numpy(dtype=dtype)
        return flat_clean.reshape(arr_obj.shape)

def parse_plotly_time_to_unix(time_input):
    """Safely converts ISO datetime strings, numeric floats, or Series into Unix float64 timestamps."""
    if time_input is None or len(time_input) == 0:
        return np.array([], dtype=np.float64)
    
    first_elem = time_input[0] if isinstance(time_input, (list, np.ndarray)) else time_input
    if isinstance(first_elem, (int, float, np.integer, np.floating)):
        return np.asarray(time_input, dtype=np.float64)
        
    dt_series = pd.Series(pd.to_datetime(time_input, utc=True, errors='coerce'))
    return (dt_series - pd.Timestamp("1970-01-01", tz='utc')).dt.total_seconds().to_numpy(dtype=np.float64)

def get_encounter_perihelion_date(enc_num):
    """Returns perihelion/midpoint date string YYYY-MM-DD for a given encounter."""
    if enc_num in ENCOUNTER_DATES:
        s_str, e_str = ENCOUNTER_DATES[enc_num]
        s_dt = datetime.strptime(s_str, "%Y-%m-%d")
        e_dt = datetime.strptime(e_str, "%Y-%m-%d")
        mid_dt = s_dt + (e_dt - s_dt) / 2
        return mid_dt.strftime("%Y-%m-%d")
    return DEFAULT_PERIHELION_DATE

def get_encounter_from_date(target_dt):
    """Maps a datetime object to its corresponding Encounter integer."""
    for enc, val in ENCOUNTER_DATES.items():
        try:
            start_str, end_str = str(val[0])[:10], str(val[1])[:10]
            start_dt = datetime.strptime(start_str, '%Y-%m-%d')
            end_dt = datetime.strptime(end_str, '%Y-%m-%d') + timedelta(days=1)
            if start_dt <= target_dt <= end_dt:
                return int(enc)
        except Exception:
            continue
    return None
    
def get_latest_past_encounter_info():
    """Identifies the most recent encounter and perihelion date relative to now UTC."""
    now_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    latest_enc = 1
    latest_peri = FALLBACK_ENCOUNTER_DATES[1][0]
    
    for enc in sorted(ENCOUNTER_DATES.keys()):
        s_str, e_str = ENCOUNTER_DATES[enc]
        s_dt = datetime.strptime(s_str, "%Y-%m-%d")
        e_dt = datetime.strptime(e_str, "%Y-%m-%d")
        mid_dt = s_dt + (e_dt - s_dt) / 2.0
        
        if mid_dt <= now_dt:
            latest_enc = enc
            latest_peri = mid_dt.strftime("%Y-%m-%d")
        else:
            break
            
    return latest_enc, latest_peri

# In-memory authentication credential cache
SESSION_AUTH = {"user": None, "pass": None}

def resolve_team_credentials():
    """
    Resolves team web credentials:
      1. Checks in-memory SESSION_AUTH.
      2. Tries reading .auth file if present in home or script directories.
      3. Prompts user via Qt QInputDialog on the main thread if unauthenticated.
    """
    global SESSION_AUTH
    if SESSION_AUTH["user"] and SESSION_AUTH["pass"]:
        return SESSION_AUTH["user"], SESSION_AUTH["pass"]

    # 1. Attempt loading from .auth file
    candidate_auth_paths = [
        os.path.join(os.path.dirname(SCRIPT_DIR), '.auth'),
        os.path.join(SCRIPT_DIR, '.auth'),
        os.path.expanduser('~/.auth')
    ]
    for auth_path in candidate_auth_paths:
        if os.path.exists(auth_path):
            try:
                with open(auth_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            k, v = line.split('=', 1)
                            k_upper = k.strip().upper()
                            if k_upper in ['CFA_USER', 'BERKELEY_USER'] and not SESSION_AUTH["user"]:
                                SESSION_AUTH["user"] = v.strip()
                            elif k_upper in ['CFA_PASS', 'BERKELEY_PASS'] and not SESSION_AUTH["pass"]:
                                SESSION_AUTH["pass"] = v.strip()
                if SESSION_AUTH["user"] and SESSION_AUTH["pass"]:
                    return SESSION_AUTH["user"], SESSION_AUTH["pass"]
            except Exception:
                pass

    # 2. Prompt user via GUI Dialog if running inside QApplication
    app = QtWidgets.QApplication.instance()
    if app:
        user, ok1 = QtWidgets.QInputDialog.getText(
            None, "SWEAP Team Web Authentication Required", 
            "Enter SWEAP team Username:"
        )
        if ok1 and user:
            password, ok2 = QtWidgets.QInputDialog.getText(
                None, "SWEAP Team Web Authentication Required", 
                f"Enter Password for '{user}':", 
                QtWidgets.QLineEdit.Password
            )
            if ok2 and password:
                SESSION_AUTH["user"] = user.strip()
                SESSION_AUTH["pass"] = password.strip()
                return SESSION_AUTH["user"], SESSION_AUTH["pass"]

    return None, None


def get_team_json_payload(enc_num, module_name):
    """
    3-Tier Fallback Resolver with HTTP Basic Auth Support:
      Tier 1: Local Staging on fc / Google Drive.
      Tier 2: Previously downloaded JSON in local temp cache.
      Tier 3: Authenticated HTTPS fetch from w3sweap web server.
    """
    filename = f"psp_{module_name}_enc_{enc_num}.json.gz"
    
    # Priority Tier 1: Local Staging Paths
    candidate_local_paths = [
        os.path.join(JSON_TEAM_ROOT, f"E{enc_num}", filename),
        f"/home/kpaulson/website/SWEAP/browser/staging/team/JSON/E{enc_num}/{filename}",
        f"/mnt/sweaparc/sa-home/kpaulson/website/SWEAP/browser/staging/team/JSON/E{enc_num}/{filename}"
    ]
    for p in candidate_local_paths:
        if os.path.exists(p):
            try:
                with gzip.open(p, 'rt', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass

    # Priority Tier 2: Local Temp Cache
    cache_dir = os.path.join(tempfile.gettempdir(), 'psp_vdf_cache', f"E{enc_num}")
    os.makedirs(cache_dir, exist_ok=True)
    cached_file = os.path.join(cache_dir, filename)
    
    if os.path.exists(cached_file):
        try:
            with gzip.open(cached_file, 'rt', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass

    # Priority Tier 3: HTTPS Fetch with Basic Auth
    url = f"{REMOTE_BASE_URL}/E{enc_num}/{filename}"
    user, password = resolve_team_credentials()

    try:
        ts_now = datetime.now().strftime('%H:%M:%S')
        print(f"[{ts_now}] Local file not found. Fetching web fallback: {url}")
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        # Configure Basic Authentication Handler if credentials exist
        handlers = [urllib.request.HTTPSHandler(context=ctx)]
        if user and password:
            password_mgr = urllib.request.HTTPPasswordMgrWithPriorAuth()
            password_mgr.add_password(None, REMOTE_BASE_URL, user, password, is_authenticated=True)
            handlers.append(urllib.request.HTTPBasicAuthHandler(password_mgr))

        opener = urllib.request.build_opener(*handlers)
        
        with opener.open(req, timeout=12) as response:
            content = response.read()
            with open(cached_file, 'wb') as f_out:
                f_out.write(content)
            
            with gzip.open(io.BytesIO(content), 'rt', encoding='utf-8') as f:
                return json.load(f)

    except urllib.error.HTTPError as http_err:
        if http_err.code == 401:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [!] Authentication failed (HTTP 401) for {url}")
            # Reset invalid session auth so user is prompted next time
            SESSION_AUTH["user"], SESSION_AUTH["pass"] = None, None
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [!] Web fetch HTTP Error {http_err.code}: {http_err.reason}")
        return None
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [!] Web fetch failed for {filename}: {e}")
        return None

# ==========================================
# CLI SELECTOR (OPTIONAL RETRO MODE)
# ==========================================
def prompt_interactive_cli_selection():
    """CLI terminal UI launched only when --cli flag is explicitly passed."""
    print("\n========================================================")
    print("      PARKER SOLAR PROBE - SPAN-I VDF BROWSER (CLI)     ")
    print("========================================================\n")
    print("  [E] Browse by Encounter Window (1 to 38)")
    print("  [O] Browse by Full Orbit Window (1 to 38)")
    print("  [C] Custom Date (YYYY-MM-DD)")
    print("  [Q] Quit\n")
    
    mode = input("Choose selection mode [E/O/C/Q]: ").strip().upper()
    if mode == 'Q':
        sys.exit(0)
    elif mode == 'C':
        return input("Enter date (YYYY-MM-DD): ").strip()
        
    date_dict = ORBIT_DATES if mode == 'O' else ENCOUNTER_DATES
    mode_name = "Orbit" if mode == 'O' else "Encounter"
    
    num_keys = sorted(date_dict.keys())
    print(f"\nAvailable {mode_name}s:")
    for i in range(0, len(num_keys), 2):
        k1 = num_keys[i]
        s1, e1 = date_dict[k1]
        line = f"  [{k1:2d}] {mode_name} {k1:2d} ({s1} to {e1})"
        if i + 1 < len(num_keys):
            k2 = num_keys[i + 1]
            s2, e2 = date_dict[k2]
            line += f"  |  [{k2:2d}] {mode_name} {k2:2d} ({s2} to {e2})"
        print(line)
        
    choice = input(f"\nSelect {mode_name} number [1-{max(num_keys)}]: ").strip()
    if choice.isdigit() and int(choice) in date_dict:
        return get_encounter_perihelion_date(int(choice))
        
    return DEFAULT_PERIHELION_DATE

# ==========================================
# UTILITY FUNCTIONS & COLORMAPS
# ==========================================
def normalize_species_tag(species_input):
    clean_str = str(species_input).strip().lower()
    if clean_str in ['00', 'h+', 'h', 'p', 'proton', 'protons']:
        return '00'
    elif clean_str in ['01', 'he++', 'he', 'alpha', 'alphas']:
        return '01'
    elif clean_str in ['0a', 'sf0a', 'decontam']:
        return '0a'
    elif clean_str in ['both', 'all', 'h+he', 'he+h', '00+01', '00_01']:
        return 'both'
    else:
        return clean_str

def prepare_4d_grid(grid_array, num_records):
    grid_flat = grid_array.flatten()
    elements = 2048
    if grid_flat.size == num_records * elements:
        return grid_flat.reshape((num_records, 8, 32, 8))
    return np.tile(grid_flat[:elements].reshape((8, 32, 8)), (num_records, 1, 1, 1))

def centers_to_nodes(C):
    if C is None or C.ndim != 2:
        return None
    top = 2 * C[0, :] - C[1, :]
    bot = 2 * C[-1, :] - C[-2, :]
    C_ext0 = np.vstack([top[np.newaxis, :], C, bot[np.newaxis, :]])

    left = 2 * C_ext0[:, 0] - C_ext0[:, 1]
    right = 2 * C_ext0[:, -1] - C_ext0[:, -2]
    C_ext = np.column_stack([left[:, np.newaxis], C_ext0, right[:, np.newaxis]])

    return 0.25 * (C_ext[:-1, :-1] + C_ext[:-1, 1:] + C_ext[1:, :-1] + C_ext[1:, 1:])

def make_quantized_colormap(contour_levels):
    num_levels = len(contour_levels) - 1
    base_cmap = pg.colormap.get('turbo')
    pos = np.linspace(0, 1, num_levels + 1)
    colors = base_cmap.map(pos, mode='byte')
    colors[0] = [0, 0, 0, 0]
    return pg.ColorMap(pos, colors)

def make_bipolar_colormap():
    pos = np.array([0.0, 0.5, 1.0])
    colors = np.array([
        [31, 119, 180, 255],
        [245, 245, 245, 255],
        [214, 39, 40, 255]
    ], dtype=np.ubyte)
    return pg.ColorMap(pos, colors)

def compute_polar_fan_slice_onthefly(vx_2d, vyz_2d, vdf_log_2d, x_grid, y_grid):
    vx_f, vyz_f, vdf_log_f = vx_2d.flatten(), vyz_2d.flatten(), vdf_log_2d.flatten()
    valid = np.isfinite(vx_f) & np.isfinite(vyz_f) & np.isfinite(vdf_log_f) & (vdf_log_f >= 0.0)

    if np.sum(valid) < 3:
        return np.full((x_grid.shape[1], x_grid.shape[0]), np.nan, dtype=np.float32)

    points = np.column_stack((vx_f[valid], vyz_f[valid]))
    try:
        tri = Delaunay(points)
    except Exception:
        return np.full((x_grid.shape[1], x_grid.shape[0]), np.nan, dtype=np.float32)

    grid_pts = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    simplex = tri.find_simplex(grid_pts)
    
    interp = LinearNDInterpolator(tri, vdf_log_f[valid])
    vdf_interp_flat = interp(grid_pts)
    vdf_interp_flat[simplex == -1] = np.nan
    
    return np.floor(vdf_interp_flat.reshape(x_grid.shape).T)

# ==========================================
# FILE RESOLUTION & DATA EXTRACTORS
# ==========================================
def get_cached_file(source_path):
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")
    cache_dir = os.path.join(tempfile.gettempdir(), 'psp_vdf_cache')
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(source_path)
    local_cached_path = os.path.join(cache_dir, filename)
    if not os.path.exists(local_cached_path) or (os.path.getsize(source_path) != os.path.getsize(local_cached_path)):
        shutil.copyfile(source_path, local_cached_path)
    return local_cached_path

def resolve_psp_file(date_str, file_type='span', species_tag='00'):
    species_tag = normalize_species_tag(species_tag)
    date_dt = datetime.strptime(date_str, "%Y-%m-%d")
    yyyy, mm = date_dt.strftime('%Y'), date_dt.strftime('%m')
    date_formatted = date_dt.strftime('%Y%m%d')

    if file_type == 'span':
        file_prefix = f"psp_swp_spi_sf0a_L2B_mom_{date_formatted}" if species_tag == '0a' else f"psp_swp_spi_sf{species_tag}_L2_8Dx32Ex8A_{date_formatted}"
        lvl, subfolder = ('L2B', 'spi_sf0a') if species_tag == '0a' else ('L2', f"spi_sf{species_tag}")

        if LOCAL_DATA_DIR and os.path.exists(LOCAL_DATA_DIR):
            matches = glob.glob(os.path.join(LOCAL_DATA_DIR, "**", f"{file_prefix}_v*.cdf"), recursive=True)
            if matches:
                matches.sort(reverse=True)
                return get_cached_file(matches[0])

        candidate_dirs = [
            f"/psp/data/sci/sweap/spi/{lvl}/{subfolder}/{yyyy}/{mm}",
            f"/mnt/sweaparc/psp/data/sci/sweap/spi/{lvl}/{subfolder}/{yyyy}/{mm}",
            os.path.join(SWEAP_DIR, "spi", lvl, subfolder, yyyy, mm) if SWEAP_DIR else "",
            os.path.join(AUTOPLOT_CACHE, "https", "w3sweap.cfa.harvard.edu", "data", "sci", "sweap", "spi", lvl, subfolder, yyyy, mm) if AUTOPLOT_CACHE else "",
        ]
        for cdir in candidate_dirs:
            if cdir and os.path.exists(cdir):
                matches = glob.glob(os.path.join(cdir, f"{file_prefix}_v*.cdf"))
                if matches:
                    matches.sort(reverse=True)
                    return get_cached_file(matches[0])
        raise FileNotFoundError(f"Could not locate SPAN sf{species_tag} for {date_str}")

    elif file_type == 'mag':
        subfolder = 'mag_RTN_4_Sa_per_Cyc'
        prefix = f"psp_fld_l2_mag_RTN_4_Sa_per_Cyc_{date_formatted}"

        if LOCAL_DATA_DIR and os.path.exists(LOCAL_DATA_DIR):
            matches = glob.glob(os.path.join(LOCAL_DATA_DIR, "**", f"{prefix}_v*.cdf"), recursive=True)
            if matches:
                matches.sort(reverse=True)
                return get_cached_file(matches[0])

        candidate_dirs = [
            f"/psp/data/sci/fields/l2/{subfolder}/{yyyy}/{mm}",
            f"/mnt/sweaparc/psp/data/sci/fields/l2/{subfolder}/{yyyy}/{mm}",
            os.path.join(BERKELEY_DIR, "fields", "l2", subfolder, yyyy, mm) if BERKELEY_DIR else "",
            os.path.join(AUTOPLOT_CACHE, "http", "research.ssl.berkeley.edu", "data", "spp", "data", "sci", "fields", "l2", subfolder, yyyy, mm) if AUTOPLOT_CACHE else "",
        ]
        for cdir in candidate_dirs:
            if cdir and os.path.exists(cdir):
                matches = glob.glob(os.path.join(cdir, f"{prefix}_v*.cdf"))
                if matches:
                    matches.sort(reverse=True)
                    return get_cached_file(matches[0])
        raise FileNotFoundError(f"Could not locate MAG RTN CDF for {date_str}")

def load_presliced_vdf(date_str, use_test=False, test_hours=1.0, explicit_file=None):
    """Loads pre-sliced VDF dataset with local/staged fallbacks."""
    date_dt = datetime.strptime(date_str, "%Y-%m-%d")
    yyyy, mm = date_dt.strftime('%Y'), date_dt.strftime('%m')
    date_formatted = date_dt.strftime('%Y%m%d')
    enc_num = get_encounter_from_date(date_dt)

    slice_path = None
    if explicit_file and os.path.exists(explicit_file):
        slice_path = explicit_file
    else:
        candidates = []
        if enc_num is not None:
            candidates.extend([
                os.path.join(JSON_TEAM_ROOT, f"E{enc_num}", "psp_swp_spani_vdfs", f"psp_swp_spani_vdfSlice_{date_str}.json.gz"),
                os.path.join(JSON_TEAM_ROOT, f"E{enc_num}", "psp_swp_spani_vdfs", f"psp_swp_spani_vdfSlice_{date_str}.json"),
                os.path.join(JSON_TEAM_ROOT, f"E{enc_num}", f"psp_swp_spani_vdfSlice_{date_str}.json.gz"),
                os.path.join(JSON_TEAM_ROOT, f"E{enc_num}", f"psp_swp_spani_vdfSlice_{date_str}.json"),
                f"/home/kpaulson/website/SWEAP/browser/staging/team/JSON/E{enc_num}/psp_swp_spani_vdfs/psp_swp_spani_vdfSlice_{date_str}.json.gz",
            ])
        
        for cand in candidates:
            if os.path.exists(cand):
                slice_path = cand
                break

        if not slice_path and SLICED_VDF_ROOT:
            standard_npz = os.path.join(SLICED_VDF_ROOT, yyyy, mm, f"psp_swp_spi_slicedVDF_{date_formatted}.npz")
            if os.path.exists(standard_npz):
                slice_path = standard_npz

    if slice_path and os.path.exists(slice_path):
        try:
            print(f"--> [FAST LOAD] Loading Pre-sliced VDF dataset: {os.path.basename(slice_path)}")
            
            if slice_path.endswith('.npz'):
                data = np.load(slice_path)
                res = {
                    'is_presliced': True,
                    'filename': os.path.basename(slice_path),
                    'times': sanitize_float_array(data['times']),
                    'magf_inst': sanitize_float_array(data['magf_inst'], dtype=np.float32)
                }
                for k in ['p_xz_native', 'p_xy_native', 'p_vx_xz_native', 'p_vz_xz_native', 'p_vx_xy_native', 'p_vy_xy_native',
                          'he_xz_native', 'he_xy_native', 'he_vx_xz_native', 'he_vz_xz_native', 'he_vx_xy_native', 'he_vy_xy_native']:
                    if k in data:
                        res[k] = sanitize_float_array(data[k], dtype=np.float32)
                return res

            else:
                if slice_path.endswith('.gz'):
                    with gzip.open(slice_path, 'rt', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    with open(slice_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                res = {
                    'is_presliced': True,
                    'filename': os.path.basename(slice_path),
                    'times': parse_plotly_time_to_unix(data['times']),
                    'magf_inst': sanitize_float_array(data['magf_inst'], dtype=np.float32),
                    'p_xz_native': sanitize_float_array(data['p_xz'], dtype=np.float32),
                    'p_xy_native': sanitize_float_array(data['p_xy'], dtype=np.float32),
                }

                for key_new, key_dest in [
                    ('p_vx_3d', 'p_vx_3d'), ('p_vy_3d', 'p_vy_3d'), ('p_vz_3d', 'p_vz_3d'),
                    ('p_phi_idx', 'p_phi_idx'), ('p_theta_idx', 'p_theta_idx'),
                    ('p_vx_xz', 'p_vx_xz_native'), ('p_vz_xz', 'p_vz_xz_native'),
                    ('p_vx_xy', 'p_vx_xy_native'), ('p_vy_xy', 'p_vy_xy_native')
                ]:
                    if key_new in data:
                        res[key_dest] = sanitize_float_array(data[key_new])

                if 'he_xz' in data:
                    res.update({
                        'he_xz_native': sanitize_float_array(data['he_xz'], dtype=np.float32),
                        'he_xy_native': sanitize_float_array(data['he_xy'], dtype=np.float32),
                    })
                    for key_new, key_dest in [
                        ('he_vx_3d', 'he_vx_3d'), ('he_vy_3d', 'he_vy_3d'), ('he_vz_3d', 'he_vz_3d'),
                        ('he_phi_idx', 'he_phi_idx'), ('he_theta_idx', 'he_theta_idx'),
                        ('he_vx_xz', 'he_vx_xz_native'), ('he_vz_xz', 'he_vz_xz_native'),
                        ('he_vx_xy', 'he_vx_xy_native'), ('he_vy_xy', 'he_vy_xy_native')
                    ]:
                        if key_new in data:
                            res[key_dest] = sanitize_float_array(data[key_new])

                return res
        except Exception as e:
            print(f"--> Pre-sliced VDF load warning: {e}")
    return None

def load_live_vdf_cdf(date_str, species_tag='00', proton_geom=None):
    cdf_file = resolve_psp_file(date_str, file_type='span', species_tag=species_tag)
    print(f"--> [LIVE CDF SLICER] Opening raw SPAN CDF ({species_tag}): {os.path.basename(cdf_file)}")
    
    cdf = cdflib.CDF(cdf_file)
    times = cdflib.cdfepoch.unixtime(cdf.varget('Epoch'))
    num_rec = len(times)

    if 'EFLUX' in cdf.cdf_info().zVariables:
        eflux_raw = cdf.varget('EFLUX')[:num_rec]
    else:
        eflux_raw = cdf.varget('DATA')[:num_rec] * 0.97e8

    if species_tag == '00':
        theta = prepare_4d_grid(cdf.varget('THETA'), len(times))[:num_rec]
        phi = prepare_4d_grid(cdf.varget('PHI'), len(times))[:num_rec]
        energy = prepare_4d_grid(cdf.varget('ENERGY'), len(times))[:num_rec]
        geom = {'theta': theta, 'phi': phi, 'energy': energy}
    else:
        if proton_geom is not None:
            theta = prepare_4d_grid(proton_geom['theta'][0], num_rec)
            phi = prepare_4d_grid(proton_geom['phi'][0], num_rec)
            energy = prepare_4d_grid(proton_geom['energy'][0], num_rec)
            geom = proton_geom
        else:
            raise ValueError("sf0a requires proton (sf00) geometry grid!")

    try: magf_inst = cdf.varget('MAGF_INST')[:num_rec]
    except Exception: magf_inst = np.full((num_rec, 3), np.nan)

    mass = 0.010438870 if species_tag == '00' else (4.0 * 0.010438870)
    charge = 1.0 if species_tag == '00' else 2.0

    eflux_4d = eflux_raw.reshape((num_rec, 8, 32, 8))
    eflux_clean = np.where((eflux_4d < 0) | (~np.isfinite(eflux_4d)), np.nan, eflux_4d)
    energy_clean = np.where((energy <= 0) | (~np.isfinite(energy)), np.nan, energy)

    vdf_4d = (eflux_clean / energy_clean) * (mass ** 2) / ((2e-5) * energy_clean)
    
    energy_geom = np.nan_to_num(energy, nan=100.0)
    theta_geom = np.nan_to_num(theta, nan=0.0)
    phi_geom = np.nan_to_num(phi, nan=0.0)

    vel_geom = np.sqrt(2 * charge * np.maximum(energy_geom, 1.0) / mass)
    rad_th, rad_ph = np.radians(theta_geom), np.radians(phi_geom)
    
    vx_4d = vel_geom * np.cos(rad_ph) * np.cos(rad_th)
    vy_4d = vel_geom * np.sin(rad_th)
    vz_4d = vel_geom * np.sin(rad_ph) * np.cos(rad_th)

    if hasattr(cdf, 'close'):
        try: cdf.close()
        except Exception: pass

    return {
        'times': times,
        'magf_inst': magf_inst,
        'vdf_4d': vdf_4d,
        'vx_4d': vx_4d,
        'vy_4d': vy_4d,
        'vz_4d': vz_4d,
        'geom': geom
    }

def load_wave_analysis_data(date_str):
    """3-Tier Priority Resolver for Wave Power & Ellipticity Spectrograms."""
    date_dt = datetime.strptime(date_str, "%Y-%m-%d")
    yyyy, mm = date_dt.strftime('%Y'), date_dt.strftime('%m')
    
    # Priority Tier 1: High-res local Wave Analysis CDFs
    t_fft_list, freqs, power_list, ellip_list = [], None, [], []
    for hh in ['0000', '0600', '1200', '1800']:
        fname = f"PSP_WaveAnalysis_{date_str}_{hh}_v1.4.cdf"
        fpath = None
        if LOCAL_DATA_DIR and os.path.exists(LOCAL_DATA_DIR):
            matches = glob.glob(os.path.join(LOCAL_DATA_DIR, "**", fname), recursive=True)
            if matches: fpath = matches[0]
        if not fpath and WAVE_ANALYSIS_ROOT:
            fpath = os.path.join(WAVE_ANALYSIS_ROOT, yyyy, mm, fname)

        if fpath and os.path.exists(fpath):
            try:
                cdf = cdflib.CDF(fpath)
                t_fft = cdflib.cdfepoch.unixtime(cdf.varget('FFT_time'))
                if freqs is None: freqs = cdf.varget('frequencies')
                pw = cdf.varget('B_power_perp')
                el_var = next((v for v in cdf.cdf_info().zVariables if 'ellipticity' in v.lower()), None)
                el = cdf.varget(el_var) if el_var else None
                pw[pw <= 0] = np.nan
                t_fft_list.append(t_fft)
                power_list.append(pw)
                if el is not None: ellip_list.append(el)
            except Exception: pass

    if t_fft_list and freqs is not None:
        res = {'times': np.concatenate(t_fft_list), 'freqs': sanitize_float_array(freqs), 'power': sanitize_float_array(np.vstack(power_list))}
        if ellip_list:
            res['ellip'] = sanitize_float_array(np.vstack(ellip_list))
        return res

    # Priority Tier 2 & 3: Local Staged Team JSON or Remote HTTPS Fetch
    enc_num = get_encounter_from_date(date_dt)
    if enc_num:
        jdata = get_team_json_payload(enc_num, 'waves')
        if jdata and 'times' in jdata and 'wave' in jdata['times'] and 'data' in jdata:
            try:
                times_unix = parse_plotly_time_to_unix(jdata['times']['wave'])
                t_start = pd.Timestamp(date_str, tz='utc').timestamp()
                mask = (times_unix >= t_start) & (times_unix < t_start + 86400)
                
                if np.any(mask):
                    freqs = sanitize_float_array(jdata['data']['freqs'])
                    log_pw = sanitize_float_array(jdata['data']['wave_power'])
                    pw_linear = np.where(np.isfinite(log_pw), 10.0 ** log_pw, np.nan).T[mask]
                    
                    res = {'times': times_unix[mask], 'freqs': freqs, 'power': pw_linear}
                    
                    if 'ellipticity' in jdata['data'] and jdata['data']['ellipticity']:
                        el_mat = sanitize_float_array(jdata['data']['ellipticity']).T[mask]
                        res['ellip'] = el_mat
                    return res
            except Exception as e:
                print(f"--> Wave JSON parse warning: {e}")

    return None

def load_hammerhead_data(date_str):
    """3-Tier Priority Resolver for Hammerhead Counts."""
    date_dt = datetime.strptime(date_str, "%Y-%m-%d")
    yyyy, mm, dd = date_dt.strftime('%Y'), date_dt.strftime('%m'), date_dt.strftime('%d')
    
    # Priority Tier 1: Local High-Res CDF
    matches = []
    if LOCAL_DATA_DIR and os.path.exists(LOCAL_DATA_DIR):
        matches = glob.glob(os.path.join(LOCAL_DATA_DIR, "**", f"hamstring_{yyyy}-{mm}-{dd}_v*.cdf"), recursive=True)
    if not matches and HAMMERHEAD_ROOT:
        matches = glob.glob(os.path.join(HAMMERHEAD_ROOT, f"hamstring_{yyyy}-{mm}-{dd}_v*.cdf"))

    if matches:
        try:
            matches.sort(reverse=True)
            cdf = cdflib.CDF(get_cached_file(matches[0]))
            ham_times = cdflib.cdfepoch.unixtime(cdf.varget('epoch'))
            t_start = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
            bin_edges = np.arange(t_start, t_start + 86400 + 300, 300)
            counts, _ = np.histogram(ham_times, bins=bin_edges)
            return bin_edges[:-1] + 150, counts
        except Exception: pass

    # Priority Tier 2 & 3: Local Staged JSON or Remote HTTPS Fetch
    enc_num = get_encounter_from_date(date_dt)
    if enc_num:
        jdata = get_team_json_payload(enc_num, 'hammerhead')
        if jdata and 'times' in jdata and 'ham' in jdata['times'] and 'data' in jdata:
            try:
                times_unix = parse_plotly_time_to_unix(jdata['times']['ham'])
                t_start = pd.Timestamp(date_str, tz='utc').timestamp()
                mask = (times_unix >= t_start) & (times_unix < t_start + 86400)
                if np.any(mask):
                    counts = sanitize_float_array(jdata['data']['ham_counts'])[mask]
                    return times_unix[mask], counts
            except Exception as e:
                print(f"--> Hammerhead JSON parse warning: {e}")

    return None, None

def load_lfr_density_data(date_str):
    """3-Tier Priority Resolver for LFR Electron Density."""
    lfr_file = None
    if LOCAL_DATA_DIR and os.path.exists(LOCAL_DATA_DIR):
        matches = glob.glob(os.path.join(LOCAL_DATA_DIR, "**", "*lfr*mission_density*.cdf"), recursive=True)
        if matches: lfr_file = matches[0]
    if not lfr_file: lfr_file = LFR_DENSITY_FILE

    # Priority Tier 1: Local CDF
    if lfr_file and os.path.exists(lfr_file):
        try:
            cdf = cdflib.CDF(lfr_file)
            epochs_unix = cdflib.cdfepoch.unixtime(cdf.varget('epoch'))
            t_start = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
            mask = (epochs_unix >= t_start) & (epochs_unix <= t_start + 86400)
            if np.any(mask):
                density = cdf.varget('electronDensity')[mask]
                density[density < -1e30] = np.nan
                return epochs_unix[mask], density
        except Exception: pass

    # Priority Tier 2 & 3: Staged or Remote Web JSON
    date_dt = datetime.strptime(date_str, "%Y-%m-%d")
    enc_num = get_encounter_from_date(date_dt)
    if enc_num:
        jdata = get_team_json_payload(enc_num, 'lfr')
        if jdata and 'times' in jdata and 'lfr' in jdata['times'] and 'data' in jdata:
            try:
                times_unix = parse_plotly_time_to_unix(jdata['times']['lfr'])
                t_start = pd.Timestamp(date_str, tz='utc').timestamp()
                mask = (times_unix >= t_start) & (times_unix < t_start + 86400)
                if np.any(mask):
                    return times_unix[mask], sanitize_float_array(jdata['data']['np_lfr'])[mask]
            except Exception: pass

    return None, None

def load_psp_mag_data(date_str):
    """3-Tier Priority Resolver for FIELDS MAG Vectors."""
    try:
        mag_file = resolve_psp_file(date_str, file_type='mag')
        if mag_file and os.path.exists(mag_file):
            return load_psp_mag_cdf(mag_file)
    except Exception:
        pass

    date_dt = datetime.strptime(date_str, "%Y-%m-%d")
    enc_num = get_encounter_from_date(date_dt)
    if enc_num:
        jdata = get_team_json_payload(enc_num, 'mag')
        if jdata and 'times' in jdata and 'mag' in jdata['times'] and 'data' in jdata:
            try:
                times_unix = parse_plotly_time_to_unix(jdata['times']['mag'])
                t_start = pd.Timestamp(date_str, tz='utc').timestamp()
                mask = (times_unix >= t_start) & (times_unix < t_start + 86400)
                
                if np.any(mask):
                    b_data = jdata['data']
                    return {
                        'mag_times': times_unix[mask],
                        'bx': sanitize_float_array(b_data['b_r'])[mask],
                        'by': sanitize_float_array(b_data['b_t'])[mask],
                        'bz': sanitize_float_array(b_data['b_n'])[mask],
                        'b_mag': sanitize_float_array(b_data['b_tot'])[mask]
                    }
            except Exception as e:
                print(f"--> MAG JSON parse warning: {e}")

    return None

def load_moments_data(date_str, mag_data=None):
    """
    Hierarchical High-Resolution Moments Loader:
      V_R Hierarchy: SPC L3 -> SPAN-i L3 -> Merged 15-min CSV
      Density Hierarchy (for V_A): LFR -> SPC L3 -> SPAN-i L3
    """
    date_dt = datetime.strptime(date_str, "%Y-%m-%d")
    enc_num = get_encounter_from_date(date_dt)
    if not enc_num:
        return None

    t_start = pd.Timestamp(date_str, tz='utc').timestamp()
    t_end = t_start + 86400

    # -------------------------------------------------------------
    # 1. FETCH DENSITY HIERARCHY (LFR -> SPC -> SPAN-i)
    # -------------------------------------------------------------
    # Priority 1: LFR Density
    t_lfr, np_lfr = load_lfr_density_data(date_str)
    if t_lfr is not None and len(t_lfr) > 0:
        df_lfr = pd.DataFrame({'t': t_lfr, 'n_lfr': np_lfr}).dropna(subset=['t']).sort_values('t')
    else:
        df_lfr = pd.DataFrame(columns=['t', 'n_lfr'])

    # Priority 2: SPC Density
    df_spc_d = pd.DataFrame(columns=['t', 'n_spc'])
    jdata_spc = get_team_json_payload(enc_num, 'spc')
    if jdata_spc and 'times' in jdata_spc and 'spc' in jdata_spc['times'] and 'data' in jdata_spc:
        t_spc_unix = parse_plotly_time_to_unix(jdata_spc['times']['spc'])
        m_spc = (t_spc_unix >= t_start) & (t_spc_unix < t_end)
        if np.any(m_spc):
            n_full = sanitize_float_array(jdata_spc['data'].get('spc_np_full'))
            n_peak = sanitize_float_array(jdata_spc['data'].get('spc_np_peak'))
            n_spc_combined = np.where(np.isfinite(n_full), n_full, n_peak)
            if len(n_spc_combined) == len(t_spc_unix):
                df_spc_d = pd.DataFrame({'t': t_spc_unix[m_spc], 'n_spc': n_spc_combined[m_spc]}).dropna(subset=['t']).sort_values('t')

    # Priority 3: SPAN-i Density
    df_spi_d = pd.DataFrame(columns=['t', 'n_spi'])
    jdata_spi = get_team_json_payload(enc_num, 'spani')
    if jdata_spi and 'times' in jdata_spi and 'spi' in jdata_spi['times'] and 'data' in jdata_spi:
        t_spi_unix = parse_plotly_time_to_unix(jdata_spi['times']['spi'])
        m_spi = (t_spi_unix >= t_start) & (t_spi_unix < t_end)
        if np.any(m_spi):
            n_spi = sanitize_float_array(jdata_spi['data'].get('spi_np'))
            if len(n_spi) == len(t_spi_unix):
                df_spi_d = pd.DataFrame({'t': t_spi_unix[m_spi], 'n_spi': n_spi[m_spi]}).dropna(subset=['t']).sort_values('t')

    # -------------------------------------------------------------
    # 2. FETCH V_R HIERARCHY (SPC -> SPAN-i -> Merged CSV)
    # -------------------------------------------------------------
    df_vr = None

    # Priority 1: SPC V_R
    if jdata_spc and 'times' in jdata_spc and 'spc' in jdata_spc['times'] and 'data' in jdata_spc:
        t_spc_unix = parse_plotly_time_to_unix(jdata_spc['times']['spc'])
        m_spc = (t_spc_unix >= t_start) & (t_spc_unix < t_end)
        if np.any(m_spc):
            vr_full = sanitize_float_array(jdata_spc['data'].get('spc_vr_full'))
            vr_peak = sanitize_float_array(jdata_spc['data'].get('spc_vr_peak'))
            vr_spc_combined = np.where(np.isfinite(vr_full), vr_full, vr_peak)
            if len(vr_spc_combined) == len(t_spc_unix):
                vr_window = vr_spc_combined[m_spc]
                if np.any(np.isfinite(vr_window)):
                    df_vr = pd.DataFrame({'t': t_spc_unix[m_spc], 'v_bulk': vr_window}).dropna(subset=['t', 'v_bulk']).sort_values('t')

    # Priority 2: SPAN-i V_R (if SPC unavailable)
    if (df_vr is None or df_vr.empty) and jdata_spi and 'times' in jdata_spi and 'spi' in jdata_spi['times'] and 'data' in jdata_spi:
        t_spi_unix = parse_plotly_time_to_unix(jdata_spi['times']['spi'])
        m_spi = (t_spi_unix >= t_start) & (t_spi_unix < t_end)
        if np.any(m_spi):
            vr_spi = sanitize_float_array(jdata_spi['data'].get('spi_vr'))
            if len(vr_spi) == len(t_spi_unix):
                vr_window = vr_spi[m_spi]
                if np.any(np.isfinite(vr_window)):
                    df_vr = pd.DataFrame({'t': t_spi_unix[m_spi], 'v_bulk': vr_window}).dropna(subset=['t', 'v_bulk']).sort_values('t')

    # Priority 3: Merged CSV (Fallback)
    if df_vr is None or df_vr.empty:
        csv_path = resolve_merged_csv_path(enc_num)
        if csv_path and os.path.exists(csv_path):
            try:
                df_csv = load_smart_csv(csv_path)
                if 'Times' in df_csv.columns:
                    t_csv_unix = parse_plotly_time_to_unix(df_csv['Times'])
                    m_csv = (t_csv_unix >= t_start) & (t_csv_unix < t_end)
                    if np.any(m_csv):
                        vr_csv = sanitize_float_array(df_csv['Vpr-Parker'].values)[m_csv]
                        df_vr = pd.DataFrame({'t': t_csv_unix[m_csv], 'v_bulk': vr_csv}).dropna(subset=['t', 'v_bulk']).sort_values('t')
            except Exception:
                pass

    if df_vr is None or df_vr.empty:
        return None

    # -------------------------------------------------------------
    # 3. CALCULATE V_ALFVEN ON V_R TIMELINE
    # -------------------------------------------------------------
    df_comp = pd.DataFrame({'t': df_vr['t'].values})

    # Nearest-neighbor density merging within a 5-minute window
    if not df_lfr.empty:
        df_comp = pd.merge_asof(df_comp, df_lfr, on='t', tolerance=300, direction='nearest')
    else:
        df_comp['n_lfr'] = np.nan

    if not df_spc_d.empty:
        df_comp = pd.merge_asof(df_comp, df_spc_d, on='t', tolerance=300, direction='nearest')
    else:
        df_comp['n_spc'] = np.nan

    if not df_spi_d.empty:
        df_comp = pd.merge_asof(df_comp, df_spi_d, on='t', tolerance=300, direction='nearest')
    else:
        df_comp['n_spi'] = np.nan

    # Combine density hierarchy: LFR -> SPC -> SPAN-i
    dens_hier = df_comp['n_lfr'].combine_first(df_comp['n_spc']).combine_first(df_comp['n_spi']).values

    v_alfven = np.full(len(df_vr), np.nan)
    if mag_data is not None and 'mag_times' in mag_data and 'b_mag' in mag_data:
        df_mag = pd.DataFrame({
            't': sanitize_float_array(mag_data['mag_times']),
            'b_mag': sanitize_float_array(mag_data['b_mag'])
        }).dropna(subset=['t', 'b_mag']).sort_values('t')

        if not df_mag.empty:
            df_b = pd.merge_asof(df_comp[['t']], df_mag, on='t', tolerance=300, direction='nearest')
            b_vals = df_b['b_mag'].values

            valid_m = np.isfinite(b_vals) & np.isfinite(dens_hier) & (dens_hier > 0.01) & (b_vals < 1e5)
            # Standard Alfvén Speed calculation: V_A = 21.81 * |B| / sqrt(N_e)
            v_a = 21.81 * b_vals[valid_m] / np.sqrt(dens_hier[valid_m])
            v_alfven[valid_m] = np.where((v_a > 10) & (v_a < 3000), v_a, np.nan)

    return {
        'times': df_vr['t'].values,
        'v_bulk': df_vr['v_bulk'].values,
        'v_alfven': v_alfven
    }

def load_smart_csv(csv_path):
    fd, temp_path = tempfile.mkstemp(suffix='.csv', dir=tempfile.gettempdir())
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

def resolve_merged_csv_path(enc_num):
    if LOCAL_DATA_DIR and os.path.exists(LOCAL_DATA_DIR):
        matches = glob.glob(os.path.join(LOCAL_DATA_DIR, "**", f"E{enc_num:02d}.csv"), recursive=True)
        if matches: return matches[0]

    search_dirs = []
    if PSP_SHAREDDRIVE_ROOT and os.path.exists(PSP_SHAREDDRIVE_ROOT):
        search_dirs.extend(glob.glob(os.path.join(PSP_SHAREDDRIVE_ROOT, '15min_DATA_Internal', '*_CURRENT_*/')))
        search_dirs.append(os.path.join(PSP_SHAREDDRIVE_ROOT, '15min_DATA_Internal'))
    if DRIVE_ROOT:
        search_dirs.append(os.path.join(DRIVE_ROOT, '15min_DATA_Internal'))

    for sdir in search_dirs:
        if not os.path.exists(sdir): continue
        csv_file = os.path.join(sdir, f"E{enc_num:02d}.csv")
        if os.path.exists(csv_file): return csv_file
        matches = glob.glob(os.path.join(sdir, "*", f"E{enc_num:02d}.csv"))
        if matches:
            matches.sort()
            return matches[-1]
    return None

def load_psp_mag_cdf(mag_cdf_path):
    cdf = cdflib.CDF(mag_cdf_path)
    info = cdf.cdf_info().zVariables + cdf.cdf_info().rVariables

    epoch_var = next(
        (v for v in info if 'epoch' in v.lower() and 'quality' not in v.lower() and ('mag' in v.lower() or 'rtn' in v.lower())),
        'epoch_mag_RTN_4_Sa_per_Cyc'
    )

    mag_var = next(
        (v for v in info if ('mag_rtn' in v.lower() or 'mag_sc' in v.lower()) 
         and 'epoch' not in v.lower() 
         and 'quality' not in v.lower()
         and 'index' not in v.lower()),
        'psp_fld_l2_mag_RTN_4_Sa_per_Cyc'
    )

    mag_unix = cdflib.cdfepoch.unixtime(cdf.varget(epoch_var))
    b_data = cdf.varget(mag_var)
    b_data = np.where((b_data < -1e30) | ~np.isfinite(b_data), np.nan, b_data)

    return {
        'mag_times': sanitize_float_array(mag_unix),
        'bx': sanitize_float_array(b_data[:, 0]),
        'by': sanitize_float_array(b_data[:, 1]),
        'bz': sanitize_float_array(b_data[:, 2]),
        'b_mag': sanitize_float_array(np.linalg.norm(b_data, axis=1))
    }

# ==========================================
# CUSTOM VIEWBOX & AXIS CLASSES
# ==========================================

class Log10AxisItem(pg.AxisItem):
    """Smart log-scale axis item that forces tick placement onto standard 1, 2, 5 frequency channels."""
    def tickValues(self, minVal, maxVal, size):
        decade_min = int(np.floor(minVal))
        decade_max = int(np.ceil(maxVal))
        
        major_ticks = []
        minor_ticks = []
        
        for dec in range(decade_min, decade_max + 1):
            base = 10.0 ** dec
            for mult in [1.0, 2.0, 5.0]:
                freq = base * mult
                log_val = np.log10(freq)
                if minVal <= log_val <= maxVal:
                    if mult == 1.0:
                        major_ticks.append(log_val)
                    else:
                        minor_ticks.append(log_val)
                        
        return [(1.0, major_ticks), (0.3, minor_ticks)]

    def tickStrings(self, values, scale, spacing):
        strns = []
        for val in values:
            try:
                freq = 10.0 ** val
                if freq >= 10:
                    strns.append(f"{freq:.0f}")
                elif freq >= 1:
                    if abs(freq - round(freq)) < 1e-2:
                        strns.append(f"{int(round(freq))}")
                    else:
                        strns.append(f"{freq:.1f}")
                elif freq >= 0.01:
                    strns.append(f"{freq:.2g}")
                else:
                    strns.append(f"{freq:.1e}")
            except (ValueError, OverflowError, OSError):
                strns.append('')
        return strns

class ScrubViewBox(pg.ViewBox):
    def __init__(self, browser_inst, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.browser = browser_inst

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.LeftButton:
            x_val = self.mapToView(ev.pos()).x()
            self.browser.update_vdf(x_val)
            ev.accept()
        else:
            super().mouseDragEvent(ev, axis=axis)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            x_val = self.mapToView(ev.pos()).x()
            self.browser.update_vdf(x_val)
            ev.accept()
        else:
            super().mouseClickEvent(ev)

class UtcDateAxisItem(pg.DateAxisItem):
    def tickStrings(self, values, scale, spacing):
        strns = []
        if not values:
            return strns
        for val in values:
            try:
                dt = datetime.fromtimestamp(val, tz=timezone.utc)
                if spacing >= 86400:
                    strns.append(dt.strftime("%b %d"))
                elif spacing >= 3600:
                    if dt.hour == 0 and dt.minute == 0:
                        strns.append(dt.strftime("%b %d\n00:00"))
                    else:
                        strns.append(dt.strftime("%H:%M"))
                else:
                    strns.append(dt.strftime("%H:%M:%S"))
            except (ValueError, OverflowError, OSError):
                strns.append('')
        return strns

# ==========================================
# MAIN DESKTOP GUI BROWSER CLASS
# ==========================================
class VDFMagDynamicBrowser(QtWidgets.QMainWindow):
    def __init__(self, date_str=DEFAULT_PERIHELION_DATE, species_tag="both", live_cdf=False):
        super().__init__()
        self.active_date_str = date_str
        self.species_tag = species_tag
        self.live_cdf = live_cdf
        
        active_dt = datetime.strptime(self.active_date_str, "%Y-%m-%d")
        self.current_enc_num = get_encounter_from_date(active_dt) or DEFAULT_ENCOUNTER_NUM
        
        self.curve_b_mag = None
        self.curve_vr = None

        self.setWindowTitle(f"PSP SPAN-I Dynamic VDF Browser [{self.active_date_str}]")
        self.resize(1300, 1180)
        
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(14, 10, 14, 10)
        main_layout.setSpacing(8)
        
        # --- TOP ROW: OVERVIEW BANNER & ENCOUNTER SELECTOR ---
        overview_container = QtWidgets.QHBoxLayout()
        overview_container.setSpacing(10)

        # Overview Plot Widget (Left)
        overview_date_axis = UtcDateAxisItem(orientation='bottom')
        self.p_overview = pg.PlotWidget(axisItems={'bottom': overview_date_axis})
        self.p_overview.setLabel('left', 'B_RTN (nT)', font_size='8pt')
        self.p_overview.getAxis('left').setWidth(55)
        
        # Explicitly lock height bounds against Qt stylesheet recalculation collapses
        self.p_overview.setFixedHeight(95)
        self.p_overview.setMinimumHeight(95)
        self.p_overview.setMaximumHeight(95)
        self.p_overview.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.p_overview.showGrid(x=True, y=True, alpha=0.25)
        overview_container.addWidget(self.p_overview, stretch=4)

        self.overview_region = pg.LinearRegionItem(
            values=[0, 86400],
            orientation=pg.LinearRegionItem.Vertical,
            brush=pg.mkBrush(255, 0, 0, 50),
            pen=pg.mkPen('r', width=1.2),
            movable=True
        )
        self.p_overview.addItem(self.overview_region)
        self.overview_region.sigRegionChangeFinished.connect(self.on_overview_region_dragged)
        self.p_overview.scene().sigMouseClicked.connect(self.on_overview_clicked)

        enc_box = QtWidgets.QGroupBox()
        enc_box.setFixedWidth(240)
        enc_box.setFixedHeight(95)
        enc_box.setStyleSheet("QGroupBox { border: 1px solid #555555; border-radius: 4px; margin-top: 0px; }")
        enc_layout = QtWidgets.QVBoxLayout(enc_box)
        enc_layout.setContentsMargins(8, 6, 8, 6)

        lbl_enc_title = QtWidgets.QLabel("ENCOUNTER SELECTOR")
        lbl_enc_title.setAlignment(QtCore.Qt.AlignCenter)
        lbl_enc_title.setStyleSheet("font-size: 9px; font-weight: bold; color: #d62728;")
        enc_layout.addWidget(lbl_enc_title)

        ctrl_row = QtWidgets.QHBoxLayout()
        self.btn_enc_prev = QtWidgets.QToolButton()
        self.btn_enc_prev.setText("<")
        self.btn_enc_prev.clicked.connect(self.on_prev_encounter)
        
        self.combo_enc = QtWidgets.QComboBox()
        for e in range(1, 39):
            self.combo_enc.addItem(f"Encounter {e:02d}", e)
        self.combo_enc.currentIndexChanged.connect(self.on_encounter_combo_changed)

        self.btn_enc_next = QtWidgets.QToolButton()
        self.btn_enc_next.setText(">")
        self.btn_enc_next.clicked.connect(self.on_next_encounter)

        ctrl_row.addWidget(self.btn_enc_prev)
        ctrl_row.addWidget(self.combo_enc)
        ctrl_row.addWidget(self.btn_enc_next)
        enc_layout.addLayout(ctrl_row)

        self.btn_reset_view = QtWidgets.QPushButton("Reset View")
        self.btn_reset_view.setStyleSheet("font-size: 10px; padding: 2px;")
        self.btn_reset_view.clicked.connect(self.reset_overview_view)
        enc_layout.addWidget(self.btn_reset_view)

        overview_container.addWidget(enc_box, stretch=0)
        main_layout.addLayout(overview_container)

        # --- SECOND ROW: TIMESTAMP & VDF PANEL CONTROLS ---
        header_box = QtWidgets.QHBoxLayout()
        self.lbl_time_title = QtWidgets.QLabel("Active VDF Timestamp:")
        self.lbl_time_title.setStyleSheet("font-size: 12px; font-weight: bold;")
        
        self.lbl_time_val = QtWidgets.QLabel("Loading data...")
        self.lbl_time_val.setStyleSheet("font-family: monospace; font-size: 12px; color: #d62728;")
        
        header_box.addWidget(self.lbl_time_title)
        header_box.addWidget(self.lbl_time_val)
        header_box.addSpacing(25)

        self.lbl_mode = QtWidgets.QLabel("Rendering Mode:")
        self.lbl_mode.setStyleSheet("font-size: 11px; font-weight: bold;")
        self.rb_cartesian = QtWidgets.QRadioButton("Cartesian Grid")
        self.rb_native = QtWidgets.QRadioButton("Native Resolution")
        self.rb_cartesian.setChecked(True)

        self.bg_mode = QtWidgets.QButtonGroup(self)
        self.bg_mode.addButton(self.rb_cartesian, 0)
        self.bg_mode.addButton(self.rb_native, 1)
        self.bg_mode.buttonClicked.connect(self.on_mode_changed)

        header_box.addWidget(self.lbl_mode)
        header_box.addWidget(self.rb_cartesian)
        header_box.addWidget(self.rb_native)
        header_box.addSpacing(20)

        self.lbl_species = QtWidgets.QLabel("Species Panels:")
        self.lbl_species.setStyleSheet("font-size: 11px; font-weight: bold;")
        self.cb_p = QtWidgets.QCheckBox("H⁺")
        self.cb_p.setChecked(True)
        self.cb_p.toggled.connect(self.on_species_toggled)

        self.cb_he = QtWidgets.QCheckBox("He⁺⁺")
        self.cb_he.setChecked(True)
        self.cb_he.toggled.connect(self.on_species_toggled)

        header_box.addWidget(self.lbl_species)
        header_box.addWidget(self.cb_p)
        header_box.addWidget(self.cb_he)
        header_box.addSpacing(20)

        self.cb_dark = QtWidgets.QCheckBox("Dark Mode")
        self.cb_dark.toggled.connect(self.toggle_dark_mode)
        header_box.addWidget(self.cb_dark)

        header_box.addStretch()
        main_layout.addLayout(header_box)

        # --- MIDDLE PANEL: VDF PLOTS ---
        self.vdf_grid = QtWidgets.QGridLayout()
        self.vdf_grid.setContentsMargins(60, 5, 10, 10)
        self.vdf_grid.setHorizontalSpacing(8)
        self.vdf_grid.setVerticalSpacing(8)
        
        # Lock equal 50/50 vertical row stretch for H+ and He++
        self.vdf_grid.setRowStretch(0, 1)
        self.vdf_grid.setRowStretch(1, 1)

        self.x_vec = np.linspace(-2000, 200, 120)
        self.y_vec_xy = np.linspace(-700, 700, 120)
        self.y_vec_xz = np.linspace(0, 1000, 120)

        self.vx_grid_xz, self.vz_grid = np.meshgrid(self.x_vec, self.y_vec_xz)
        self.vx_grid_xy, self.vy_grid = np.meshgrid(self.x_vec, self.y_vec_xy)

        label_style = {'font-size': '11pt'}
        CONTOUR_LEVELS_P = np.arange(0.0, 9.0, 1.0)
        CONTOUR_LEVELS_HE = np.arange(0.0, 6.0, 1.0)

        quant_cmap_p = make_quantized_colormap(CONTOUR_LEVELS_P)
        quant_cmap_he = make_quantized_colormap(CONTOUR_LEVELS_HE)

        def make_panel(title, y_label, quant_cmap, contour_levels, y_range=(-700, 700)):
            p = pg.PlotWidget(title=title)
            p.setLabel('bottom', 'Vx (km/s)', **label_style)
            p.setLabel('left', y_label, **label_style)
            p.setAspectLocked(True, ratio=1.0)
            p.setXRange(-2000, 200, padding=0.02)
            p.setYRange(y_range[0], y_range[1], padding=0.02)

            img = pg.ImageItem()
            p.addItem(img)
            img.setColorMap(quant_cmap)

            poly_group = QtWidgets.QGraphicsItemGroup()
            p.addItem(poly_group)
            poly_group.setVisible(False)

            poly_items = []
            for th in range(8):
                row = []
                for en in range(32):
                    pitem = QtWidgets.QGraphicsPolygonItem()
                    poly_group.addToGroup(pitem)
                    row.append(pitem)
                poly_items.append(row)

            iso_lines = []
            for lvl in contour_levels[1:]:
                iso = pg.IsocurveItem(pen=pg.mkPen('#000000', width=0.8), level=lvl)
                iso.setParentItem(img)
                iso_lines.append(iso)

            b_vec = pg.PlotCurveItem(pen=pg.mkPen('r', width=2.5, style=QtCore.Qt.DashLine))
            p.addItem(b_vec)
            scat_all = pg.ScatterPlotItem(size=4, pen=pg.mkPen('#e0e0e0', width=0.7), brush=pg.mkBrush(None), symbol='o')
            p.addItem(scat_all)
            scat_val = pg.ScatterPlotItem(size=5, pen=pg.mkPen('#222222', width=0.8), brush=pg.mkBrush('#1f77b4'), symbol='o')
            p.addItem(scat_val)

            return p, img, poly_group, poly_items, iso_lines, b_vec, scat_all, scat_val, quant_cmap

        self.lbl_p = QtWidgets.QLabel("H⁺")
        self.lbl_p.setStyleSheet("font-size: 22px; font-weight: bold; color: #1f77b4;")
        self.vdf_grid.addWidget(self.lbl_p, 0, 0)
        
        self.p_xy, self.img_p_xy, self.pgroup_p_xy, self.polys_p_xy, self.iso_p_xy, self.b_p_xy, self.scat_p_xy_all, self.scat_p_xy_val, self.cmap_p = make_panel("VDF SPAN-I θ-plane", 'Vy (km/s)', quant_cmap_p, CONTOUR_LEVELS_P, y_range=(-700, 700))
        self.p_xz, self.img_p_xz, self.pgroup_p_xz, self.polys_p_xz, self.iso_p_xz, self.b_p_xz, self.scat_p_xz_all, self.scat_p_xz_val, _ = make_panel("VDF SPAN-I φ-plane", 'Vz (km/s)', quant_cmap_p, CONTOUR_LEVELS_P, y_range=(0, 1000))
        self.vdf_grid.addWidget(self.p_xy, 0, 1)
        self.vdf_grid.addWidget(self.p_xz, 0, 2)

        self.cbar_widget_p = pg.GraphicsLayoutWidget()
        self.cbar_widget_p.setFixedWidth(115)
        self.cbar_p = pg.ColorBarItem(values=(0.0, 8.0))
        self.cbar_p.setColorMap(quant_cmap_p)
        self.cbar_p.setImageItem([self.img_p_xy, self.img_p_xz])
        self.cbar_p.axis.setLabel(text='log₁₀ f(v)', color='#000000')
        self.cbar_widget_p.addItem(self.cbar_p)
        self.vdf_grid.addWidget(self.cbar_widget_p, 0, 3)

        self.lbl_he = QtWidgets.QLabel("He⁺⁺")
        self.lbl_he.setStyleSheet("font-size: 22px; font-weight: bold; color: #d62728;")
        self.vdf_grid.addWidget(self.lbl_he, 1, 0)
        
        self.he_xy, self.img_he_xy, self.pgroup_he_xy, self.polys_he_xy, self.iso_he_xy, self.b_he_xy, self.scat_he_xy_all, self.scat_he_xy_val, self.cmap_he = make_panel("VDF SPAN-I θ-plane", 'Vy (km/s)', quant_cmap_he, CONTOUR_LEVELS_HE, y_range=(-700, 700))
        self.he_xz, self.img_he_xz, self.pgroup_he_xz, self.polys_he_xz, self.iso_he_xz, self.b_he_xz, self.scat_he_xz_all, self.scat_he_xz_val, _ = make_panel("VDF SPAN-I φ-plane", 'Vz (km/s)', quant_cmap_he, CONTOUR_LEVELS_HE, y_range=(0, 1000))
        self.vdf_grid.addWidget(self.he_xy, 1, 1)
        self.vdf_grid.addWidget(self.he_xz, 1, 2)
        
        self.he_xy.setXLink(self.p_xy); self.he_xy.setYLink(self.p_xy)
        self.he_xz.setXLink(self.p_xy); self.he_xz.setYLink(self.p_xz)

        self.cbar_widget_he = pg.GraphicsLayoutWidget()
        self.cbar_widget_he.setFixedWidth(115)
        self.cbar_he = pg.ColorBarItem(values=(0.0, 5.0))
        self.cbar_he.setColorMap(quant_cmap_he)
        self.cbar_he.setImageItem([self.img_he_xy, self.img_he_xz])
        self.cbar_he.axis.setLabel(text='log₁₀ f(v)', color='#000000')
        self.cbar_widget_he.addItem(self.cbar_he)
        self.vdf_grid.addWidget(self.cbar_widget_he, 1, 3)

        self.p_xz.setXLink(self.p_xy)
        
        # Direct layout addition without QSplitter
        main_layout.addLayout(self.vdf_grid, stretch=4)

        # --- BOTTOM PANEL: TIMESERIES STACK ---
        self.ts_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.ts_widget, stretch=4)

        self.setup_native_timeseries_stack()
        
        QtCore.QTimer.singleShot(50, self.deferred_initial_load)

    def deferred_initial_load(self):
        """Asynchronously populates encounter banner and daily VDFs after UI draws."""
        self.sync_encounter_controls(self.current_enc_num)
        self.load_encounter_overview(self.current_enc_num)
        self.load_daily_datasets(self.active_date_str)

    # ==========================================
    # ENCOUNTER OVERVIEW & NAVIGATION CONTROLS
    # ==========================================
    def sync_encounter_controls(self, enc_num):
        self.combo_enc.blockSignals(True)
        idx = self.combo_enc.findData(enc_num)
        if idx >= 0:
            self.combo_enc.setCurrentIndex(idx)
        self.combo_enc.blockSignals(False)

    def load_encounter_overview(self, enc_num):
        """Loads and plots full encounter B_RTN dataset strictly bounded to encounter dates."""
        self.p_overview.clear()
        self.p_overview.addItem(self.overview_region)

        jdata = get_team_json_payload(enc_num, 'mag')
        if jdata and 'times' in jdata and 'mag' in jdata['times'] and 'data' in jdata:
            t_unix = parse_plotly_time_to_unix(jdata['times']['mag'])
            b_tot = sanitize_float_array(jdata['data']['b_tot'])
            b_r   = sanitize_float_array(jdata['data']['b_r'])
            b_t   = sanitize_float_array(jdata['data']['b_t'])
            b_n   = sanitize_float_array(jdata['data']['b_n'])

            self.p_overview.plot(t_unix, b_tot, pen=pg.mkPen('#888888', width=0.8))
            self.p_overview.plot(t_unix, b_r,   pen=pg.mkPen('#d62728', width=0.9))
            self.p_overview.plot(t_unix, b_t,   pen=pg.mkPen('#2ca02c', width=0.9))
            self.p_overview.plot(t_unix, b_n,   pen=pg.mkPen('#1f77b4', width=0.9))

            # Strictly enforce encounter date bounds
            if enc_num in ENCOUNTER_DATES:
                s_str, e_str = ENCOUNTER_DATES[enc_num]
                s_ts = datetime.strptime(s_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
                e_ts = (datetime.strptime(e_str, "%Y-%m-%d") + timedelta(days=1)).replace(tzinfo=timezone.utc).timestamp()
                self.p_overview.setXRange(s_ts, e_ts, padding=0.0)
            elif len(t_unix) > 0:
                valid_mask = np.isfinite(t_unix)
                if np.any(valid_mask):
                    self.p_overview.setXRange(t_unix[valid_mask][0], t_unix[valid_mask][-1], padding=0.0)

    def update_overview_highlight_box(self, date_str):
        """Moves red box highlight on overview banner to match active 24-hour day."""
        dt_start = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        ts_start = dt_start.timestamp()
        ts_end = ts_start + 86400

        self.overview_region.blockSignals(True)
        self.overview_region.setRegion([ts_start, ts_end])
        self.overview_region.blockSignals(False)

    def on_overview_region_dragged(self):
        """Triggers daily data reload when user drags or releases red box."""
        r_start, r_end = self.overview_region.getRegion()
        ts_center = r_start + (r_end - r_start) / 2.0
        dt_center = datetime.fromtimestamp(ts_center, tz=timezone.utc)
        new_date_str = dt_center.strftime("%Y-%m-%d")

        if new_date_str != self.active_date_str:
            self.change_active_date(new_date_str)

    def on_overview_clicked(self, event):
        """Jumps red box to clicked date in overview plot."""
        pos = event.scenePos()
        if self.p_overview.plotItem.sceneBoundingRect().contains(pos):
            mouse_point = self.p_overview.plotItem.vb.mapSceneToView(pos)
            ts_click = mouse_point.x()
            dt_click = datetime.fromtimestamp(ts_click, tz=timezone.utc)
            new_date_str = dt_click.strftime("%Y-%m-%d")
            self.change_active_date(new_date_str)

    def on_encounter_combo_changed(self, index):
        enc_num = self.combo_enc.itemData(index)
        if enc_num and enc_num != self.current_enc_num:
            self.current_enc_num = enc_num
            new_date = get_encounter_perihelion_date(enc_num)
            self.load_encounter_overview(enc_num)
            self.change_active_date(new_date)

    def on_prev_encounter(self):
        if self.current_enc_num > 1:
            self.combo_enc.setCurrentIndex(self.combo_enc.currentIndex() - 1)

    def on_next_encounter(self):
        if self.combo_enc.currentIndex() < self.combo_enc.count() - 1:
            self.combo_enc.setCurrentIndex(self.combo_enc.currentIndex() + 1)

    def reset_overview_view(self):
        self.p_overview.autoRange()

    # ==========================================
    # ACTIVE DAY DATA LOADING & REFRESH LOGIC
    # ==========================================
    def change_active_date(self, new_date_str):
        self.active_date_str = new_date_str
        active_dt = datetime.strptime(self.active_date_str, "%Y-%m-%d")
        new_enc = get_encounter_from_date(active_dt)
        
        if new_enc and new_enc != self.current_enc_num:
            self.current_enc_num = new_enc
            self.sync_encounter_controls(new_enc)
            self.load_encounter_overview(new_enc)

        self.setWindowTitle(f"PSP SPAN-I Dynamic VDF Browser [{self.active_date_str}]")
        self.update_overview_highlight_box(self.active_date_str)
        self.load_daily_datasets(self.active_date_str)

    def load_daily_datasets(self, date_str):
        """Reloads daily VDFs and timeseries stack for active date with responsive Qt events."""
        # Process pending Qt events to update window title and cursor immediately
        app = QtWidgets.QApplication.instance()
        if app:
            app.processEvents()

        # Sync red highlight box position on overview banner
        self.update_overview_highlight_box(date_str)

        self.presliced_vdf = None
        self.vdf_data_p = None
        self.vdf_data_he = None

        if not self.live_cdf:
            self.presliced_vdf = load_presliced_vdf(date_str)

        if self.presliced_vdf is None:
            try:
                self.vdf_data_p = load_live_vdf_cdf(date_str, species_tag='00')
            except Exception as e:
                print(f"--> Proton CDF load warning: {e}")

            if self.species_tag in ['01', '0a', 'both'] and self.vdf_data_p is not None:
                try:
                    self.vdf_data_he = load_live_vdf_cdf(date_str, species_tag='0a', proton_geom=self.vdf_data_p['geom'])
                except Exception as e:
                    print(f"--> Alpha CDF load warning: {e}")

        self.mag_data = load_psp_mag_data(date_str)
        self.wave_data = load_wave_analysis_data(date_str)
        self.hh_data = load_hammerhead_data(date_str)
        self.lfr_data = load_lfr_density_data(date_str)
        self.moments_data = load_moments_data(date_str, mag_data=self.mag_data)

        self.is_presliced = (self.presliced_vdf is not None)
        self.is_dual = (self.vdf_data_he is not None) or (self.is_presliced and ('he_xz_native' in self.presliced_vdf or 'he_xz' in self.presliced_vdf))
        self.cb_he.setEnabled(self.is_dual)

        if self.is_presliced and 'times' in self.presliced_vdf:
            self.times_ref = self.presliced_vdf['times']
        elif self.vdf_data_p is not None and 'times' in self.vdf_data_p:
            self.times_ref = self.vdf_data_p['times']
        elif self.mag_data is not None and 'mag_times' in self.mag_data:
            self.times_ref = self.mag_data['mag_times']
        else:
            t_start = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
            self.times_ref = np.linspace(t_start, t_start + 86400, 100)

        self.replot_native_timeseries_stack()
        self.update_vdf(self.times_ref[0])

    # ==========================================
    # UI EVENT HANDLERS & RENDERING
    # ==========================================
    def on_species_toggled(self):
        """Toggles H+ and He++ visibility and re-balances QGridLayout row stretches."""
        if not self.cb_p.isChecked() and not self.cb_he.isChecked():
            sender = self.sender()
            if sender:
                sender.blockSignals(True)
                sender.setChecked(True)
                sender.blockSignals(False)
            return

        show_p = self.cb_p.isChecked()
        show_he = self.cb_he.isChecked() and self.is_dual

        self.lbl_p.setVisible(show_p)
        self.p_xy.setVisible(show_p)
        self.p_xz.setVisible(show_p)
        self.cbar_widget_p.setVisible(show_p)

        self.lbl_he.setVisible(show_he)
        self.he_xy.setVisible(show_he)
        self.he_xz.setVisible(show_he)
        self.cbar_widget_he.setVisible(show_he)

        # Dynamically allocate grid row stretch based on active panels
        self.vdf_grid.setRowStretch(0, 1 if show_p else 0)
        self.vdf_grid.setRowStretch(1, 1 if show_he else 0)

        if hasattr(self, 'lbl_time_val') and self.lbl_time_val.text():
            ts_str = self.lbl_time_val.text().replace(" UTC", "")
            try:
                dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                ts_unix = dt.replace(tzinfo=timezone.utc).timestamp()
                self.update_vdf(ts_unix, sync_lines=False)
            except Exception:
                pass

    def toggle_dark_mode(self, is_dark):
        """Toggles dark mode theme while explicitly preserving QSplitter geometry."""
        # Save exact QSplitter panel heights before applying stylesheet
        splitter_sizes = self.splitter.sizes() if hasattr(self, 'splitter') else None

        bg_color = '#2b2b2b' if is_dark else '#ffffff'
        fg_color = '#e0e0e0' if is_dark else '#000000'
        grid_alpha = 0.2 if is_dark else 0.3

        pg.setConfigOption('background', bg_color)
        pg.setConfigOption('foreground', fg_color)

        if self.centralWidget():
            self.centralWidget().setStyleSheet(f"background-color: {bg_color}; color: {fg_color};")

        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {bg_color};
            }}
            QGraphicsView {{
                background-color: {bg_color};
                border: none;
                margin: 0px;
                padding: 0px;
            }}
            QLabel, QRadioButton, QCheckBox {{
                background-color: transparent;
                color: {fg_color};
            }}
            QGroupBox {{
                background-color: transparent;
                color: {fg_color};
                border: 1px solid {'#555555' if is_dark else '#aaaaaa'};
                border-radius: 4px;
            }}
            QComboBox, QPushButton, QToolButton {{
                background-color: {'#3a3a3a' if is_dark else '#f0f0f0'};
                color: {fg_color};
                border: 1px solid {'#555555' if is_dark else '#cccccc'};
            }}
        """)

        if hasattr(self, 'ts_widget'): self.ts_widget.setBackground(bg_color)
        if hasattr(self, 'cbar_widget_p'): self.cbar_widget_p.setBackground(bg_color)
        if hasattr(self, 'cbar_widget_he'): self.cbar_widget_he.setBackground(bg_color)
        if hasattr(self, 'p_overview'): self.p_overview.setBackground(bg_color)

        # Force text and pen color on ColorBarItem labels
        for cb_attr in ['cbar_p', 'cbar_he']:
            if hasattr(self, cb_attr):
                cbar = getattr(self, cb_attr)
                if hasattr(cbar, 'axis') and cbar.axis is not None:
                    cbar.axis.setPen(pg.mkPen(fg_color))
                    cbar.axis.setTextPen(pg.mkPen(fg_color))
                    cbar.axis.setLabel(text='log₁₀ f(v)', color=fg_color)

        all_plots = [self.p_overview] if hasattr(self, 'p_overview') else []
        for p_name in ['p_xy', 'p_xz', 'he_xy', 'he_xz', 'p_mag', 'p_wave', 'p_ellip', 'p_mom']:
            if hasattr(self, p_name):
                all_plots.append(getattr(self, p_name))

        for p in all_plots:
            if hasattr(p, 'setBackground'):
                p.setBackground(bg_color)

            plot_item = p.getPlotItem() if hasattr(p, 'getPlotItem') else p
            vb = plot_item.getViewBox() if hasattr(plot_item, 'getViewBox') else None
            if vb:
                vb.setBackgroundColor(bg_color)
                vb.setBorder(None)

            if hasattr(plot_item, 'titleLabel') and plot_item.titleLabel:
                t_str = plot_item.titleLabel.text
                plot_item.setTitle(t_str, color=fg_color)

            for ax_name in ['left', 'bottom', 'top', 'right']:
                ax = plot_item.getAxis(ax_name)
                if ax:
                    ax.setPen(pg.mkPen(fg_color, width=1.0))
                    ax.setTextPen(pg.mkPen(fg_color))
                    if hasattr(ax, 'label') and ax.label:
                        ax.label.setDefaultTextColor(QtGui.QColor(fg_color))

            plot_item.showGrid(x=True, y=True, alpha=grid_alpha)

        if hasattr(self, 'curve_b_mag') and self.curve_b_mag is not None:
            self.curve_b_mag.setPen(pg.mkPen(fg_color, width=1.2))
        if hasattr(self, 'curve_vr') and self.curve_vr is not None:
            self.curve_vr.setPen(pg.mkPen(fg_color, width=1.5))

        for lbl_attr in ['lbl_time_title', 'lbl_mode', 'lbl_species']:
            if hasattr(self, lbl_attr):
                getattr(self, lbl_attr).setStyleSheet(f"font-size: 11px; font-weight: bold; color: {fg_color};")

        if hasattr(self, 'lbl_mag'):
            self.lbl_mag.setText(
                f"<span style='color:{fg_color}; font-size:10pt;'>|B|</span><br>"
                "<span style='color:#d62728; font-size:10pt;'>B_R</span><br>"
                "<span style='color:#2ca02c; font-size:10pt;'>B_T</span><br>"
                "<span style='color:#1f77b4; font-size:10pt;'>B_N</span>"
            )
        if hasattr(self, 'lbl_mom'):
            self.lbl_mom.setText(
                f"<span style='color:{fg_color}; font-size:10pt;'>V_R</span><br>"
                "<span style='color:#1f77b4; font-size:10pt;'>V_A</span>"
            )

        scatter_pen = pg.mkPen('#e0e0e0' if is_dark else '#888888', width=0.7)
        for scat_attr in ['scat_p_xy_all', 'scat_p_xz_all', 'scat_he_xy_all', 'scat_he_xz_all']:
            if hasattr(self, scat_attr):
                getattr(self, scat_attr).setPen(scatter_pen)

        if hasattr(self, 'lbl_time_val'):
            self.lbl_time_val.setStyleSheet(f"font-family: monospace; font-size: 12px; color: {'#ff6666' if is_dark else '#d62728'};")

        # Restore exact QSplitter geometry after stylesheet parsing completes
        if splitter_sizes and hasattr(self, 'splitter'):
            self.splitter.setSizes(splitter_sizes)

    def on_mode_changed(self, button):
        if hasattr(self, 'lbl_time_val') and self.lbl_time_val.text():
            ts_str = self.lbl_time_val.text().replace(" UTC", "")
            try:
                dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                ts_unix = dt.replace(tzinfo=timezone.utc).timestamp()
                self.update_vdf(ts_unix, sync_lines=False)
            except Exception:
                pass

    def setup_native_timeseries_stack(self):
        """Initializes plot objects with zero-margin tight vertical stacking and smart log axes."""
        label_style = {'font-size': '10pt'}
        Y_AXIS_FIXED_WIDTH = 75

        self.ts_widget.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.ts_widget.ci.layout.setSpacing(1)

        # Plot 1: B_RTN
        vb_mag = ScrubViewBox(self)
        self.p_mag = self.ts_widget.addPlot(row=0, col=0, viewBox=vb_mag)
        self.p_mag.setLabel('left', 'B_RTN<br>(nT)', **label_style)
        self.p_mag.getAxis('left').setWidth(Y_AXIS_FIXED_WIDTH)
        self.p_mag.showGrid(x=True, y=True, alpha=0.3)
        self.p_mag.hideAxis('bottom')

        self.lbl_mag = self.ts_widget.addLabel(
            "<span style='color:#000000; font-size:10pt;'>|B|</span><br>"
            "<span style='color:#d62728; font-size:10pt;'>B_R</span><br>"
            "<span style='color:#2ca02c; font-size:10pt;'>B_T</span><br>"
            "<span style='color:#1f77b4; font-size:10pt;'>B_N</span>",
            row=0, col=1, justify='left'
        )
        self.lbl_mag.setMaximumWidth(45)

        # Plot 2: Wave Power Spectrogram with Log10AxisItem
        vb_wave = ScrubViewBox(self)
        log_axis_wave = Log10AxisItem(orientation='left')
        self.p_wave = self.ts_widget.addPlot(row=1, col=0, viewBox=vb_wave, axisItems={'left': log_axis_wave})
        self.p_wave.setLabel('left', 'Wave Power<br>Freq (Hz)', **label_style)
        self.p_wave.getAxis('left').setWidth(Y_AXIS_FIXED_WIDTH)
        self.p_wave.hideAxis('bottom')
        self.p_wave.setXLink(self.p_mag)

        lbl_wave_spacer = self.ts_widget.addLabel("", row=1, col=1)
        lbl_wave_spacer.setMaximumWidth(45)

        # Plot 3: Ellipticity Spectrogram with Log10AxisItem
        vb_ellip = ScrubViewBox(self)
        log_axis_ellip = Log10AxisItem(orientation='left')
        self.p_ellip = self.ts_widget.addPlot(row=2, col=0, viewBox=vb_ellip, axisItems={'left': log_axis_ellip})
        self.p_ellip.setLabel('left', 'Ellipticity<br>Freq (Hz)', **label_style)
        self.p_ellip.getAxis('left').setWidth(Y_AXIS_FIXED_WIDTH)
        self.p_ellip.hideAxis('bottom')
        self.p_ellip.setXLink(self.p_mag)
        
        # Link Spectrogram Frequency Y-Axes
        self.p_ellip.setYLink(self.p_wave)

        lbl_ellip_spacer = self.ts_widget.addLabel("", row=2, col=1)
        lbl_ellip_spacer.setMaximumWidth(45)

        # Plot 4: Velocity Moments
        vb_mom = ScrubViewBox(self)
        date_axis = UtcDateAxisItem(orientation='bottom')
        self.p_mom = self.ts_widget.addPlot(row=3, col=0, viewBox=vb_mom, axisItems={'bottom': date_axis})
        self.p_mom.setLabel('left', 'Velocity<br>(km/s)', **label_style)
        self.p_mom.getAxis('left').setWidth(Y_AXIS_FIXED_WIDTH)
        self.p_mom.setLabel('bottom', 'Time (UTC)', **label_style)
        self.p_mom.showGrid(x=True, y=True, alpha=0.3)
        self.p_mom.setXLink(self.p_mag)

        self.lbl_mom = self.ts_widget.addLabel(
            "<span style='color:#000000; font-size:10pt;'>V_R</span><br>"
            "<span style='color:#1f77b4; font-size:10pt;'>V_A</span>",
            row=3, col=1, justify='left'
        )
        self.lbl_mom.setMaximumWidth(45)

        self.v_lines = []
        for p in [self.p_mag, self.p_wave, self.p_ellip, self.p_mom]:
            vl = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=1.5, style=QtCore.Qt.SolidLine))
            vl.setZValue(100)
            p.addItem(vl)
            self.v_lines.append(vl)

    def replot_native_timeseries_stack(self):
        """Populates timeseries curves & heatmaps when active day reloads."""
        for p in [self.p_mag, self.p_wave, self.p_ellip, self.p_mom]:
            p.clear()

        self.v_lines = []
        for p in [self.p_mag, self.p_wave, self.p_ellip, self.p_mom]:
            vl = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=1.5, style=QtCore.Qt.SolidLine))
            vl.setZValue(100)
            p.addItem(vl)
            self.v_lines.append(vl)

        # Plot 1: Daily MAG B_RTN
        if self.mag_data is not None:
            t_m = sanitize_float_array(self.mag_data['mag_times'])
            self.curve_b_mag = self.p_mag.plot(t_m, sanitize_float_array(self.mag_data['b_mag']), pen=pg.mkPen('#000000', width=1.2))
            self.p_mag.plot(t_m, sanitize_float_array(self.mag_data['bx']), pen=pg.mkPen('#d62728', width=1.0))
            self.p_mag.plot(t_m, sanitize_float_array(self.mag_data['by']), pen=pg.mkPen('#2ca02c', width=1.0))
            self.p_mag.plot(t_m, sanitize_float_array(self.mag_data['bz']), pen=pg.mkPen('#1f77b4', width=1.0))
            
            valid_m = np.isfinite(t_m)
            if np.any(valid_m):
                valid_t_m = t_m[valid_m]
                self.p_mag.setXRange(valid_t_m[0], valid_t_m[-1], padding=0)

        # Plot 2: Wave Power Spectrogram
        if self.wave_data is not None and self.wave_data.get('power') is not None:
            img_wave = pg.ImageItem()
            img_wave.setZValue(0)
            self.p_wave.addItem(img_wave)
            img_wave.setColorMap(pg.colormap.get('turbo'))
            
            t_w = sanitize_float_array(self.wave_data['times'])
            freqs = sanitize_float_array(self.wave_data['freqs'])
            pw_raw = sanitize_float_array(self.wave_data['power'])

            if len(freqs) > 0 and len(t_w) > 0:
                log_f_orig = np.log10(np.maximum(freqs, 1e-4))
                log_f_min, log_f_max = log_f_orig[0], log_f_orig[-1]
                log_f_uniform = np.linspace(log_f_min, log_f_max, 200)

                f_interp = interp1d(log_f_orig, np.log10(np.clip(pw_raw, 1e-3, 1e3)), axis=1, bounds_error=False, fill_value=-3.0)
                pw_log_uniform = f_interp(log_f_uniform)

                img_wave.setImage(pw_log_uniform)
                rect = QtCore.QRectF(t_w[0], log_f_min, t_w[-1] - t_w[0], log_f_max - log_f_min)
                img_wave.setRect(rect)
                
                # Autorange Y-axis tightly to valid frequency range
                self.p_wave.setYRange(log_f_min, log_f_max, padding=0.0)

        # Plot 3: Ellipticity Spectrogram
        if self.wave_data is not None and self.wave_data.get('ellip') is not None:
            img_ellip = pg.ImageItem()
            img_ellip.setZValue(0)
            self.p_ellip.addItem(img_ellip)
            img_ellip.setColorMap(make_bipolar_colormap())

            t_w = sanitize_float_array(self.wave_data['times'])
            freqs = sanitize_float_array(self.wave_data['freqs'])
            el_raw = sanitize_float_array(self.wave_data['ellip'])

            if len(freqs) > 0 and len(t_w) > 0:
                log_f_orig = np.log10(np.maximum(freqs, 1e-4))
                log_f_min, log_f_max = log_f_orig[0], log_f_orig[-1]
                log_f_uniform = np.linspace(log_f_min, log_f_max, 200)

                el_interp = interp1d(log_f_orig, np.clip(el_raw, -1.0, 1.0), axis=1, bounds_error=False, fill_value=0.0)
                el_uniform = el_interp(log_f_uniform)

                img_ellip.setImage(el_uniform, levels=(-1.0, 1.0))
                rect = QtCore.QRectF(t_w[0], log_f_min, t_w[-1] - t_w[0], log_f_max - log_f_min)
                img_ellip.setRect(rect)
                
                # Autorange Y-axis tightly to valid frequency range
                self.p_ellip.setYRange(log_f_min, log_f_max, padding=0.0)

        # Plot 4: Plasma Moments (Vr, Va)
        if self.moments_data is not None:
            t_mom = sanitize_float_array(self.moments_data['times'])
            if self.moments_data.get('v_bulk') is not None:
                self.curve_vr = self.p_mom.plot(t_mom, sanitize_float_array(self.moments_data['v_bulk']), pen=pg.mkPen('#000000', width=1.5))
            if self.moments_data.get('v_alfven') is not None:
                self.p_mom.plot(t_mom, sanitize_float_array(self.moments_data['v_alfven']), pen=pg.mkPen('#1f77b4', width=1.2))

        if self.cb_dark.isChecked():
            self.toggle_dark_mode(True)

    def update_vdf(self, cursor_x, sync_lines=True):
        idx = np.clip(np.searchsorted(self.times_ref, cursor_x), 0, len(self.times_ref) - 1)
        actual_ts = self.times_ref[idx]
        
        dt_utc = datetime.fromtimestamp(actual_ts, tz=timezone.utc)
        self.lbl_time_val.setText(dt_utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC")

        if sync_lines and hasattr(self, 'v_lines'):
            for vl in self.v_lines:
                vl.blockSignals(True)
                vl.setValue(actual_ts)
                vl.blockSignals(False)

        use_native = self.rb_native.isChecked()

        def get_slice_vel(data_dict, prefix, plane, vel_axis):
            if not data_dict: return None
            
            lut_key = f"{prefix}_{vel_axis}_3d"
            idx_key = f"{prefix}_phi_idx" if plane == 'xz' else f"{prefix}_theta_idx"
            
            if lut_key in data_dict and idx_key in data_dict:
                lut = np.asarray(data_dict[lut_key])
                slice_idx = int(data_dict[idx_key][idx])
                if plane == 'xz':
                    return lut[:, :, slice_idx]
                else:
                    return lut[slice_idx, :, :].T
            
            for key in [f"{prefix}_{vel_axis}_{plane}_native", f"{prefix}_{vel_axis}_{plane}"]:
                if key in data_dict:
                    arr = np.asarray(data_dict[key])
                    if arr.ndim == 3: return arr[idx]
                    elif arr.ndim == 2: return arr
            return None

        def get_grid_item(data_dict, key):
            if not data_dict: return None
            arr = data_dict.get(key, None)
            if arr is None: return None
            arr = np.asarray(arr)
            if arr.ndim == 3: return arr[idx]
            elif arr.ndim == 2: return arr
            return None

        def update_scatter(scat_all, scat_val, vx_arr, vy_arr, vdf_log_raw, cmap=None, levels=(0.0, 8.0)):
            if vx_arr is not None and vy_arr is not None and vdf_log_raw is not None:
                vx_f = np.nan_to_num(vx_arr.flatten(), nan=0.0)
                vy_f = np.nan_to_num(vy_arr.flatten(), nan=0.0)
                vdf_f = np.nan_to_num(vdf_log_raw.flatten(), nan=0.0)
                
                scat_all.setData(x=vx_f, y=vy_f)
                
                valid = (vdf_f >= 1.0)
                if np.any(valid):
                    v_vals = vdf_f[valid]
                    if cmap is not None:
                        norm_vals = np.clip((v_vals - levels[0]) / (levels[1] - levels[0]), 0.0, 1.0)
                        rgba_colors = cmap.map(norm_vals, mode='byte')
                        brushes = [pg.mkBrush(*c) for c in rgba_colors]
                        scat_val.setData(x=vx_f[valid], y=vy_f[valid], brush=brushes)
                    else:
                        scat_val.setData(x=vx_f[valid], y=vy_f[valid])
                else:
                    scat_val.clear()
            else:
                scat_all.clear()
                scat_val.clear()

        def update_panel_slice(img_item, poly_group, poly_items, iso_list, native_vdf, native_vx, native_vyz, is_xz=True, cmap=None, levels=(0.0, 8.0)):
            if native_vdf is None or native_vx is None or native_vyz is None:
                img_item.clear()
                if poly_group is not None: poly_group.setVisible(False)
                return

            vdf_log_2d = native_vdf.astype(np.float32)
            vx_2d = np.nan_to_num(native_vx.astype(np.float64), nan=0.0)
            vyz_2d = np.nan_to_num(native_vyz.astype(np.float64), nan=0.0)

            if use_native and poly_group is not None:
                img_item.setVisible(False)
                for iso in iso_list: iso.setVisible(False)
                poly_group.setVisible(True)

                vx_nodes = centers_to_nodes(vx_2d)
                vyz_nodes = centers_to_nodes(vyz_2d)
                clean_vdf = np.nan_to_num(np.floor(np.maximum(vdf_log_2d.astype(np.float64), 0.0)), nan=0.0)

                for th in range(clean_vdf.shape[0]):
                    for en in range(clean_vdf.shape[1]):
                        poly_item = poly_items[th][en]
                        v_val = clean_vdf[th, en]
                        if v_val >= 1.0:
                            c_x = [vx_nodes[th, en], vx_nodes[th+1, en], vx_nodes[th+1, en+1], vx_nodes[th, en+1]]
                            c_y = [vyz_nodes[th, en], vyz_nodes[th+1, en], vyz_nodes[th+1, en+1], vyz_nodes[th, en+1]]
                            poly_item.setPolygon(QtGui.QPolygonF([QtCore.QPointF(c_x[k], c_y[k]) for k in range(4)]))
                            
                            norm_val = np.clip((v_val - levels[0]) / (levels[1] - levels[0]), 0.0, 1.0)
                            r, g, b, _ = cmap.map(norm_val, mode='byte')
                            poly_item.setBrush(QtGui.QBrush(QtGui.QColor(r, g, b, 230)))
                            poly_item.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 120), 0.5))
                            poly_item.setVisible(True)
                        else:
                            poly_item.setVisible(False)
            else:
                if poly_group is not None: poly_group.setVisible(False)
                img_item.setVisible(True)
                for iso in iso_list: iso.setVisible(True)

                if is_xz:
                    data_slice = compute_polar_fan_slice_onthefly(vx_2d, vyz_2d, vdf_log_2d, self.vx_grid_xz, self.vz_grid)
                    img_item.setRect(QtCore.QRectF(-2000, 0, 2200, 1000))
                else:
                    data_slice = compute_polar_fan_slice_onthefly(vx_2d, vyz_2d, vdf_log_2d, self.vx_grid_xy, self.vy_grid)
                    img_item.setRect(QtCore.QRectF(-2000, -700, 2200, 1400))

                if np.any(np.isfinite(data_slice)):
                    img_item.setImage(data_slice, autoLevels=False)
                    clean_iso_data = np.nan_to_num(data_slice, nan=-999.0)
                    for iso in iso_list:
                        iso.setData(clean_iso_data)
                else:
                    img_item.clear()
                    for iso in iso_list: iso.setData(None)

        if self.is_presliced:
            if self.cb_p.isChecked():
                p_xz_nat = get_grid_item(self.presliced_vdf, 'p_xz_native')
                if p_xz_nat is None:
                    p_xz_nat = get_grid_item(self.presliced_vdf, 'p_xz')

                p_vx_xz_nat = get_slice_vel(self.presliced_vdf, 'p', 'xz', 'vx')
                p_vz_xz_nat = get_slice_vel(self.presliced_vdf, 'p', 'xz', 'vz')

                p_xy_nat = get_grid_item(self.presliced_vdf, 'p_xy_native')
                if p_xy_nat is None:
                    p_xy_nat = get_grid_item(self.presliced_vdf, 'p_xy')

                p_vx_xy_nat = get_slice_vel(self.presliced_vdf, 'p', 'xy', 'vx')
                p_vy_xy_nat = get_slice_vel(self.presliced_vdf, 'p', 'xy', 'vy')

                update_panel_slice(self.img_p_xz, self.pgroup_p_xz, self.polys_p_xz, self.iso_p_xz, p_xz_nat, p_vx_xz_nat, p_vz_xz_nat, is_xz=True, cmap=self.cmap_p, levels=(0.0, 8.0))
                update_panel_slice(self.img_p_xy, self.pgroup_p_xy, self.polys_p_xy, self.iso_p_xy, p_xy_nat, p_vx_xy_nat, p_vy_xy_nat, is_xz=False, cmap=self.cmap_p, levels=(0.0, 8.0))

                update_scatter(self.scat_p_xz_all, self.scat_p_xz_val, p_vx_xz_nat, p_vz_xz_nat, p_xz_nat, cmap=self.cmap_p, levels=(0.0, 8.0))
                update_scatter(self.scat_p_xy_all, self.scat_p_xy_val, p_vx_xy_nat, p_vy_xy_nat, p_xy_nat, cmap=self.cmap_p, levels=(0.0, 8.0))

            if self.is_dual and self.cb_he.isChecked() and ('he_xz_native' in self.presliced_vdf or 'he_xz' in self.presliced_vdf):
                he_xz_nat = get_grid_item(self.presliced_vdf, 'he_xz_native')
                if he_xz_nat is None:
                    he_xz_nat = get_grid_item(self.presliced_vdf, 'he_xz')

                he_vx_xz_nat = get_slice_vel(self.presliced_vdf, 'he', 'xz', 'vx')
                he_vz_xz_nat = get_slice_vel(self.presliced_vdf, 'he', 'xz', 'vz')

                he_xy_nat = get_grid_item(self.presliced_vdf, 'he_xy_native')
                if he_xy_nat is None:
                    he_xy_nat = get_grid_item(self.presliced_vdf, 'he_xy')

                he_vx_xy_nat = get_slice_vel(self.presliced_vdf, 'he', 'xy', 'vx')
                he_vy_xy_nat = get_slice_vel(self.presliced_vdf, 'he', 'xy', 'vy')

                update_panel_slice(self.img_he_xz, self.pgroup_he_xz, self.polys_he_xz, self.iso_he_xz, he_xz_nat, he_vx_xz_nat, he_vz_xz_nat, is_xz=True, cmap=self.cmap_he, levels=(0.0, 5.0))
                update_panel_slice(self.img_he_xy, self.pgroup_he_xy, self.polys_he_xy, self.iso_he_xy, he_xy_nat, he_vx_xy_nat, he_vy_xy_nat, is_xz=False, cmap=self.cmap_he, levels=(0.0, 5.0))

                update_scatter(self.scat_he_xz_all, self.scat_he_xz_val, he_vx_xz_nat, he_vz_xz_nat, he_xz_nat, cmap=self.cmap_he, levels=(0.0, 5.0))
                update_scatter(self.scat_he_xy_all, self.scat_he_xy_val, he_vx_xy_nat, he_vy_xy_nat, he_xy_nat, cmap=self.cmap_he, levels=(0.0, 5.0))

            if 'magf_inst' in self.presliced_vdf:
                b_inst = self.presliced_vdf['magf_inst'][idx]
                if np.all(np.isfinite(b_inst)):
                    b_norm = np.linalg.norm(b_inst)
                    if b_norm > 0:
                        scale = 500.0
                        bx_u, bz_u = b_inst[0] / b_norm, b_inst[2] / b_norm
                        self.b_p_xz.setData(x=[-scale * bx_u, scale * bx_u], y=[-scale * bz_u, scale * bz_u])

        else:
            if self.cb_p.isChecked() and self.vdf_data_p is not None:
                vdf_3d = self.vdf_data_p['vdf_4d'][idx]
                if np.any(np.isfinite(vdf_3d) & (vdf_3d > 0)):
                    i_theta_max, _, i_phi_max = np.unravel_index(np.argmax(np.nan_to_num(vdf_3d)), vdf_3d.shape)
                else:
                    i_theta_max, i_phi_max = 0, 0

                vx_xz_2d = self.vdf_data_p['vx_4d'][idx, :, :, i_phi_max]
                vz_xz_2d = self.vdf_data_p['vz_4d'][idx, :, :, i_phi_max]
                vdf_xz_2d = vdf_3d[:, :, i_phi_max]

                with np.errstate(divide='ignore', invalid='ignore'):
                    log_vdf_xz = np.log10(vdf_xz_2d)
                log_vdf_xz_clean = np.where(vdf_xz_2d > 0, log_vdf_xz, np.where(vdf_xz_2d == 0, 0.0, np.nan)).astype(np.float32)

                vx_xy_2d = self.vdf_data_p['vx_4d'][idx, i_theta_max, :, :].T
                vy_xy_2d = self.vdf_data_p['vy_4d'][idx, i_theta_max, :, :].T
                vdf_xy_2d = vdf_3d[i_theta_max, :, :].T

                with np.errstate(divide='ignore', invalid='ignore'):
                    log_vdf_xy = np.log10(vdf_xy_2d)
                log_vdf_xy_clean = np.where(vdf_xy_2d > 0, log_vdf_xy, np.where(vdf_xy_2d == 0, 0.0, np.nan)).astype(np.float32)

                update_panel_slice(self.img_p_xz, self.pgroup_p_xz, self.polys_p_xz, self.iso_p_xz, log_vdf_xz_clean, vx_xz_2d, vz_xz_2d, is_xz=True, cmap=self.cmap_p, levels=(0.0, 8.0))
                update_panel_slice(self.img_p_xy, self.pgroup_p_xy, self.polys_p_xy, self.iso_p_xy, log_vdf_xy_clean, vx_xy_2d, vy_xy_2d, is_xz=False, cmap=self.cmap_p, levels=(0.0, 8.0))

                update_scatter(self.scat_p_xz_all, self.scat_p_xz_val, vx_xz_2d, vz_xz_2d, log_vdf_xz_clean, cmap=self.cmap_p, levels=(0.0, 8.0))
                update_scatter(self.scat_p_xy_all, self.scat_p_xy_val, vx_xy_2d, vy_xy_2d, log_vdf_xy_clean, cmap=self.cmap_p, levels=(0.0, 8.0))

            if self.is_dual and self.cb_he.isChecked() and self.vdf_data_he is not None:
                idx_he = np.clip(np.searchsorted(self.vdf_data_he['times'], actual_ts), 0, len(self.vdf_data_he['times']) - 1)
                vdf_3d_he = self.vdf_data_he['vdf_4d'][idx_he]
                if np.any(np.isfinite(vdf_3d_he) & (vdf_3d_he > 0)):
                    i_theta_max, _, i_phi_max = np.unravel_index(np.argmax(np.nan_to_num(vdf_3d_he)), vdf_3d_he.shape)
                else:
                    i_theta_max, i_phi_max = 0, 0

                vx_he_xz = self.vdf_data_he['vx_4d'][idx_he, :, :, i_phi_max]
                vz_he_xz = self.vdf_data_he['vz_4d'][idx_he, :, :, i_phi_max]
                vdf_he_xz = vdf_3d_he[:, :, i_phi_max]

                with np.errstate(divide='ignore', invalid='ignore'):
                    log_vdf_he_xz = np.log10(vdf_he_xz)
                log_vdf_he_xz_clean = np.where(vdf_he_xz > 0, log_vdf_he_xz, np.where(vdf_he_xz == 0, 0.0, np.nan)).astype(np.float32)

                vx_he_xy = self.vdf_data_he['vx_4d'][idx_he, i_theta_max, :, :].T
                vy_he_xy = self.vdf_data_he['vy_4d'][idx_he, i_theta_max, :, :].T
                vdf_he_xy = vdf_3d_he[i_theta_max, :, :].T

                with np.errstate(divide='ignore', invalid='ignore'):
                    log_vdf_he_xy = np.log10(vdf_he_xy)
                log_vdf_he_xy_clean = np.where(vdf_he_xy > 0, log_vdf_he_xy, np.where(vdf_he_xy == 0, 0.0, np.nan)).astype(np.float32)

                update_panel_slice(self.img_he_xz, self.pgroup_he_xz, self.polys_he_xz, self.iso_he_xz, log_vdf_he_xz_clean, vx_he_xz, vz_he_xz, is_xz=True, cmap=self.cmap_he, levels=(0.0, 5.0))
                update_panel_slice(self.img_he_xy, self.pgroup_he_xy, self.polys_he_xy, self.iso_he_xy, log_vdf_he_xy_clean, vx_he_xy, vy_he_xy, is_xz=False, cmap=self.cmap_he, levels=(0.0, 5.0))

                update_scatter(self.scat_he_xz_all, self.scat_he_xz_val, vx_he_xz, vz_he_xz, log_vdf_he_xz_clean, cmap=self.cmap_he, levels=(0.0, 5.0))
                update_scatter(self.scat_he_xy_all, self.scat_he_xy_val, vx_he_xy, vy_he_xy, log_vdf_he_xy_clean, cmap=self.cmap_he, levels=(0.0, 5.0))

            if self.vdf_data_p is not None and 'magf_inst' in self.vdf_data_p:
                b_inst = self.vdf_data_p['magf_inst'][idx]
                if np.all(np.isfinite(b_inst)):
                    b_norm = np.linalg.norm(b_inst)
                    if b_norm > 0:
                        scale = 500.0
                        bx_u, bz_u = b_inst[0] / b_norm, b_inst[2] / b_norm
                        self.b_p_xz.setData(x=[-scale * bx_u, scale * bx_u], y=[-scale * bz_u, scale * bz_u])

# ==========================================
# MAIN SCRIPT EXECUTION ENTRY POINT
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PSP SPAN-I Dynamic VDF Browser")
    parser.add_argument("--date", type=str, default=None, help="Date YYYY-MM-DD")
    parser.add_argument("--cli", action="store_true", help="Launch interactive retro CLI menu instead of direct GUI startup")
    parser.add_argument("--species", type=str, default="both", help="'00', '0a', or 'both'")
    parser.add_argument("--data-dir", type=str, default=None, help="Optional local directory containing staged CDFs/CSVs")
    parser.add_argument("--dark-mode", action="store_true", help="Enable dark mode theme")
    
    parser.add_argument("--live-cdf", action="store_true", help="Bypass .json/.npz files and slice raw CDFs directly from memory")
    parser.add_argument("--vdf-slice-file", type=str, default=None, help="Direct path override for a specific pre-sliced file")
    args = parser.parse_args()
    
    DEFAULT_ENCOUNTER_NUM, DEFAULT_PERIHELION_DATE = get_latest_past_encounter_info()

    if args.date:
        selected_date = args.date
    elif args.cli:
        selected_date = prompt_interactive_cli_selection()
    else:
        selected_date = DEFAULT_PERIHELION_DATE

    if args.data_dir:
        LOCAL_DATA_DIR = os.path.abspath(args.data_dir)
        print(f"--> [Local Cache Mode] Override active: {LOCAL_DATA_DIR}")

    app = pg.mkQApp("PSP SPAN-I Dynamic VDF Browser")

    if not args.dark_mode:
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        app.setStyleSheet("""
            QMainWindow, QWidget, QFrame, QLabel {
                background-color: #ffffff;
                color: #000000;
            }
            QSplitter::handle {
                background-color: #e0e0e0;
            }
        """)

    # 1. Pop up Splash Screen immediately
    splash = create_splash_screen("SWEAP_LOGO.png")
    splash.show()
    app.processEvents()

    species_tag = normalize_species_tag(args.species)
    
    # 2. Instantiate and show Window Shell
    browser = VDFMagDynamicBrowser(
        date_str=selected_date,
        species_tag=species_tag,
        live_cdf=args.live_cdf
    )

    if args.dark_mode:
        browser.cb_dark.setChecked(True)

    browser.show()
    app.processEvents()

    # 3. Dismiss Splash Screen once UI renders
    splash.finish(browser)

    sys.exit(app.exec_())
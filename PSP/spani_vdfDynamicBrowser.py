import os
import sys
import gzip
import json

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


# --- IMPORT MASTER CONFIGURATION ---
SCRIPT_DIR = "/home/kpaulson/MyDrive/Software/Python/githubScripts_python" if os.name != 'nt' else "G:/My Drive/Software/Python/githubScripts_python"
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

try:
    import config
    DRIVE_ROOT = config.get_drive_path()
    PSP_SHAREDDRIVE_ROOT = config.get_sharedDrivePSP_path()
    SWEAP_DIR = config.get_sweapCacheData()
    BERKELEY_DIR = config.get_berkeleyCacheData()
    ENCOUNTER_DATES = config.ENCOUNTER_DATES
except ImportError:
    DRIVE_ROOT, PSP_SHAREDDRIVE_ROOT, SWEAP_DIR, BERKELEY_DIR = "", "", "", ""
    ENCOUNTER_DATES = {}

AUTOPLOT_CACHE = os.path.join(DRIVE_ROOT, "Research", "Data", "AutoplotCache") if DRIVE_ROOT else ""
WAVE_ANALYSIS_ROOT = os.path.join(DRIVE_ROOT, "Research", "PSP", "WaveAnalysis", "WaveAnalysis_Files", "v1.4") if DRIVE_ROOT else ""
HAMMERHEAD_ROOT = os.path.join(DRIVE_ROOT, "Research", "PSP", "Hammerheads", "Hamstrings", "cdf", "v02") if DRIVE_ROOT else ""
JSON_MOMENTS_ROOT = os.path.join(DRIVE_ROOT, "Research", "PSP", "JSON") if DRIVE_ROOT else ""
JSON_TEAM_ROOT = os.path.join(DRIVE_ROOT, "Research", "PSP", "JSON", "team") if DRIVE_ROOT else "JSON/team"
SLICED_VDF_ROOT = os.path.join(DRIVE_ROOT, "Research", "PSP", "SPAN", "SPANi", "SPANi_slicedVDF") if DRIVE_ROOT else ""
LFR_DENSITY_FILE = os.path.join(AUTOPLOT_CACHE, "https", "w3sweap.cfa.harvard.edu", "data", "sci", "sweap", "spc", "LFR", "spp_fld_lfr_mission_density.cdf") if AUTOPLOT_CACHE else ""

LOCAL_DATA_DIR = None

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def get_encounter_from_date(target_dt):
    dict_source = {}
    if 'config' in sys.modules:
        dict_source = getattr(config, 'ENCOUNTER_DATERANGE', getattr(config, 'ENCOUNTER_DATES', {}))
    if not dict_source:
        dict_source = ENCOUNTER_DATES

    for enc, val in dict_source.items():
        try:
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                start_str, end_str = str(val[0])[:10], str(val[1])[:10]
            elif isinstance(val, str):
                start_str, end_str = val[:10], val[:10]
            else:
                continue

            start_dt = datetime.strptime(start_str, '%Y-%m-%d')
            end_dt = datetime.strptime(end_str, '%Y-%m-%d') + timedelta(days=1)
            if start_dt <= target_dt <= end_dt:
                return int(enc)
        except Exception:
            continue
    return None

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
# CACHING & FILE RESOLUTION
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
            rf"\\sshfs.r\kpaulson@fc.cfa.harvard.edu\mnt\sweaparc\psp\data\sci\sweap\spi\{lvl}\{subfolder}\{yyyy}\{mm}\\",
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
            rf"\\sshfs.r\kpaulson@fc.cfa.harvard.edu\mnt\sweaparc\psp\data\sci\fields\l2\{subfolder}\{yyyy}\{mm}\\",
        ]
        for cdir in candidate_dirs:
            if cdir and os.path.exists(cdir):
                matches = glob.glob(os.path.join(cdir, f"{prefix}_v*.cdf"))
                if matches:
                    matches.sort(reverse=True)
                    return get_cached_file(matches[0])
        raise FileNotFoundError(f"Could not locate MAG RTN CDF for {date_str}")

# ==========================================
# DATA LOADERS (JSON SLICES OR LIVE CDF)
# ==========================================
def load_presliced_vdf(date_str, use_test=False, test_hours=1.0, explicit_file=None):
    """Loads pre-sliced VDF dataset from JSON.GZ (Encounter path) or legacy NPZ files."""
    date_dt = datetime.strptime(date_str, "%Y-%m-%d")
    yyyy, mm = date_dt.strftime('%Y'), date_dt.strftime('%m')
    date_formatted = date_dt.strftime('%Y%m%d')
    enc_num = get_encounter_from_date(date_dt)

    slice_path = None
    if explicit_file and os.path.exists(explicit_file):
        slice_path = explicit_file
    else:
        if enc_num is not None and JSON_TEAM_ROOT:
            cand_gz = os.path.join(JSON_TEAM_ROOT, f"E{enc_num}", "psp_swp_spani_vdfs", f"psp_swp_spani_vdfSlice_{date_str}.json.gz")
            cand_json = os.path.join(JSON_TEAM_ROOT, f"E{enc_num}", "psp_swp_spani_vdfs", f"psp_swp_spani_vdfSlice_{date_str}.json")
            if os.path.exists(cand_gz):
                slice_path = cand_gz
            elif os.path.exists(cand_json):
                slice_path = cand_json

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
                    'times': data['times'],
                    'magf_inst': data['magf_inst']
                }
                for k in ['p_xz_native', 'p_xy_native', 'p_vx_xz_native', 'p_vz_xz_native', 'p_vx_xy_native', 'p_vy_xy_native',
                          'he_xz_native', 'he_xy_native', 'he_vx_xz_native', 'he_vz_xz_native', 'he_vx_xy_native', 'he_vy_xy_native']:
                    if k in data:
                        res[k] = data[k]
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
                    'times': np.array(data['times'], dtype=np.float64),
                    'magf_inst': np.array(data['magf_inst'], dtype=np.float32),
                    'p_xz_native': np.array(data['p_xz'], dtype=np.float32),
                    'p_xy_native': np.array(data['p_xy'], dtype=np.float32),
                }

                # Support both New 3D LUT Keys and Legacy 2D Keys
                for key_new, key_dest in [
                    ('p_vx_3d', 'p_vx_3d'), ('p_vy_3d', 'p_vy_3d'), ('p_vz_3d', 'p_vz_3d'),
                    ('p_phi_idx', 'p_phi_idx'), ('p_theta_idx', 'p_theta_idx'),
                    ('p_vx_xz', 'p_vx_xz_native'), ('p_vz_xz', 'p_vz_xz_native'),
                    ('p_vx_xy', 'p_vx_xy_native'), ('p_vy_xy', 'p_vy_xy_native')
                ]:
                    if key_new in data:
                        res[key_dest] = np.array(data[key_new])

                if 'he_xz' in data:
                    res.update({
                        'he_xz_native': np.array(data['he_xz'], dtype=np.float32),
                        'he_xy_native': np.array(data['he_xy'], dtype=np.float32),
                    })
                    for key_new, key_dest in [
                        ('he_vx_3d', 'he_vx_3d'), ('he_vy_3d', 'he_vy_3d'), ('he_vz_3d', 'he_vz_3d'),
                        ('he_phi_idx', 'he_phi_idx'), ('he_theta_idx', 'he_theta_idx'),
                        ('he_vx_xz', 'he_vx_xz_native'), ('he_vz_xz', 'he_vz_xz_native'),
                        ('he_vx_xy', 'he_vx_xy_native'), ('he_vy_xy', 'he_vy_xy_native')
                    ]:
                        if key_new in data:
                            res[key_dest] = np.array(data[key_new])

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
    date_dt = datetime.strptime(date_str, "%Y-%m-%d")
    yyyy, mm = date_dt.strftime('%Y'), date_dt.strftime('%m')
    t_fft_list, freqs, power_list, ellip_list = [], None, [], []
    for hh in ['0000', '0600', '1200', '1800']:
        fname = f"PSP_WaveAnalysis_{date_str}_{hh}_v1.4.cdf"
        fpath = None
        if LOCAL_DATA_DIR and os.path.exists(LOCAL_DATA_DIR):
            matches = glob.glob(os.path.join(LOCAL_DATA_DIR, "**", fname), recursive=True)
            if matches: fpath = matches[0]
        if not fpath:
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
        res = {'times': np.concatenate(t_fft_list), 'freqs': freqs, 'power': np.vstack(power_list)}
        if ellip_list:
            res['ellip'] = np.vstack(ellip_list)
        return res
    return None

def load_hammerhead_data(date_str):
    date_dt = datetime.strptime(date_str, "%Y-%m-%d")
    yyyy, mm, dd = date_dt.strftime('%Y'), date_dt.strftime('%m'), date_dt.strftime('%d')
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
    return None, None

def load_lfr_density_data(date_str):
    lfr_file = None
    if LOCAL_DATA_DIR and os.path.exists(LOCAL_DATA_DIR):
        matches = glob.glob(os.path.join(LOCAL_DATA_DIR, "**", "*lfr*mission_density*.cdf"), recursive=True)
        if matches: lfr_file = matches[0]
    if not lfr_file: lfr_file = LFR_DENSITY_FILE

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
    return None, None

def load_moments_data(date_str, mag_data=None):
    date_dt = datetime.strptime(date_str, "%Y-%m-%d")
    enc_num = get_encounter_from_date(date_dt)
    
    if enc_num is not None and JSON_MOMENTS_ROOT:
        json_pattern = os.path.join(JSON_MOMENTS_ROOT, f"E{enc_num:02d}", f"psp_spani_enc_{enc_num:02d}_*.json.gz")
        matches = glob.glob(json_pattern)
        if matches:
            try:
                print(f"--> Loading High-Res JSON Plasma Moments for Encounter {enc_num:02d}")
                with gzip.open(matches[0], 'rt', encoding='utf-8') as f:
                    jdata = json.load(f)
                times_str = jdata.get('times', [])
                dt_series = pd.to_datetime(times_str, utc=True)
                times_unix = (dt_series - pd.Timestamp("1970-01-01", tz='utc')).total_seconds().values
                
                t_start = pd.Timestamp(date_str, tz='utc').timestamp()
                mask = (times_unix >= t_start) & (times_unix < t_start + 86400)
                if np.any(mask):
                    v_bulk = np.array(jdata.get('vr', []))[mask]
                    v_alfven = np.array(jdata.get('va', []))[mask]
                    return {'times': times_unix[mask], 'v_bulk': v_bulk, 'v_alfven': v_alfven}
            except Exception as e:
                print(f"--> JSON moments load warning: {e}")

    if enc_num is not None:
        csv_path = resolve_merged_csv_path(enc_num)
        if csv_path and os.path.exists(csv_path):
            try:
                df = load_smart_csv(csv_path)
                if 'Times' in df.columns:
                    t_start = pd.Timestamp(date_str)
                    mask = (df['Times'] >= t_start) & (df['Times'] < t_start + pd.Timedelta(days=1))
                    df_win = df.loc[mask].copy()
                    if not df_win.empty:
                        times_unix = (df_win['Times'] - pd.Timestamp("1970-01-01")).dt.total_seconds().values
                        v_bulk = df_win['Vpr-Parker'].values if 'Vpr-Parker' in df_win.columns else None
                        dens = df_win['Np-Parker'].values if 'Np-Parker' in df_win.columns else None
                        v_alfven = None
                        if dens is not None and mag_data is not None:
                            b_interp = np.interp(times_unix, mag_data['mag_times'], mag_data['b_mag'], left=np.nan, right=np.nan)
                            v_alfven = 21.8 * b_interp / np.sqrt(np.where(dens > 0, 1.1 * dens, np.nan))
                        return {'times': times_unix, 'dens': dens, 'v_bulk': v_bulk, 'v_alfven': v_alfven}
            except Exception as e:
                print(f"--> Merged SWEAP CSV load warning: {e}")
    return None

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
        'mag_times': mag_unix,
        'bx': b_data[:, 0],
        'by': b_data[:, 1],
        'bz': b_data[:, 2],
        'b_mag': np.linalg.norm(b_data, axis=1)
    }

# ==========================================
# DESKTOP GUI BROWSER
# ==========================================
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
                    strns.append(dt.strftime("%Y-%m-%d"))
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

class VDFMagDynamicBrowser(QtWidgets.QMainWindow):
    def __init__(self, vdf_data_p, mag_data, vdf_data_he=None, wave_data=None, hh_data=None, lfr_data=None, moments_data=None, date_str="2020-01-29", presliced_vdf=None):
        super().__init__()
        self.vdf_data_p = vdf_data_p
        self.vdf_data_he = vdf_data_he
        self.mag_data = mag_data
        self.wave_data = wave_data
        self.hh_data = hh_data
        self.lfr_data = lfr_data
        self.moments_data = moments_data
        self.presliced_vdf = presliced_vdf
        self.is_presliced = (presliced_vdf is not None)
        self.is_dual = (vdf_data_he is not None) or (self.is_presliced and ('he_xz_native' in presliced_vdf or 'he_xz' in presliced_vdf))
        
        self.curve_b_mag = None
        self.curve_vr = None

        mode_label = f" [{self.presliced_vdf['filename']}]" if self.is_presliced else " [LIVE CDF SLICER]"
        self.setWindowTitle(f"PSP SPAN-I Dynamic VDF Browser ({date_str}){mode_label}")
        self.resize(1250, 1150)
        
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        
        main_layout.setContentsMargins(18, 12, 18, 12)
        main_layout.setSpacing(10)
        
        # Header Timestamp & Control Bar
        header_box = QtWidgets.QHBoxLayout()
        self.lbl_time_title = QtWidgets.QLabel("Active VDF Timestamp:")
        self.lbl_time_title.setStyleSheet("font-size: 12px; font-weight: bold;")
        
        if self.is_presliced and self.presliced_vdf is not None and 'times' in self.presliced_vdf:
            self.times_ref = self.presliced_vdf['times']
        elif self.vdf_data_p is not None and 'times' in self.vdf_data_p:
            self.times_ref = self.vdf_data_p['times']
        elif self.vdf_data_he is not None and 'times' in self.vdf_data_he:
            self.times_ref = self.vdf_data_he['times']
        elif self.mag_data is not None and 'mag_times' in self.mag_data:
            self.times_ref = self.mag_data['mag_times']
        elif self.moments_data is not None and 'times' in self.moments_data:
            self.times_ref = self.moments_data['times']
        else:
            t_start = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
            self.times_ref = np.linspace(t_start, t_start + 86400, 100)

        self.lbl_time_val = QtWidgets.QLabel("")
        self.lbl_time_val.setStyleSheet("font-family: monospace; font-size: 12px; color: #d62728;")
        
        header_box.addWidget(self.lbl_time_title)
        header_box.addWidget(self.lbl_time_val)
        header_box.addSpacing(25)

        # UI Mode Switcher Radio Buttons
        self.lbl_mode = QtWidgets.QLabel("Rendering Mode:")
        self.lbl_mode.setStyleSheet("font-size: 11px; font-weight: bold;")
        self.rb_cartesian = QtWidgets.QRadioButton("Cartesian Grid (100x100 + Contours)")
        self.rb_native = QtWidgets.QRadioButton("Native QuadMesh (8x32 Bins)")
        self.rb_cartesian.setChecked(True)

        self.bg_mode = QtWidgets.QButtonGroup(self)
        self.bg_mode.addButton(self.rb_cartesian, 0)
        self.bg_mode.addButton(self.rb_native, 1)
        self.bg_mode.buttonClicked.connect(self.on_mode_changed)

        header_box.addWidget(self.lbl_mode)
        header_box.addWidget(self.rb_cartesian)
        header_box.addWidget(self.rb_native)
        header_box.addSpacing(20)

        # Species Panel Toggle Checkboxes
        self.lbl_species = QtWidgets.QLabel("Species Panels:")
        self.lbl_species.setStyleSheet("font-size: 11px; font-weight: bold;")
        self.cb_p = QtWidgets.QCheckBox("H⁺")
        self.cb_p.setChecked(True)
        self.cb_p.toggled.connect(self.on_species_toggled)

        self.cb_he = QtWidgets.QCheckBox("He⁺⁺")
        self.cb_he.setChecked(self.is_dual)
        self.cb_he.setEnabled(self.is_dual)
        self.cb_he.toggled.connect(self.on_species_toggled)

        header_box.addWidget(self.lbl_species)
        header_box.addWidget(self.cb_p)
        header_box.addWidget(self.cb_he)
        header_box.addSpacing(20)

        # Live Soft Dark Mode Checkbox Toggle
        self.cb_dark = QtWidgets.QCheckBox("Dark Mode")
        self.cb_dark.toggled.connect(self.toggle_dark_mode)
        header_box.addWidget(self.cb_dark)

        header_box.addStretch()
        main_layout.addLayout(header_box)

        # TOP PANEL: PYQTGRAPH VDFs
        self.vdf_grid = QtWidgets.QGridLayout()
        self.vdf_grid.setContentsMargins(60, 5, 10, 10)
        self.vdf_grid.setHorizontalSpacing(8)
        self.vdf_grid.setVerticalSpacing(8)

        # On-the-fly Cartesian Grids
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

        # --- PROTONS (Row 0) ---
        self.lbl_p = QtWidgets.QLabel("H⁺")
        self.lbl_p.setStyleSheet("font-size: 22px; font-weight: bold; color: #1f77b4;")
        self.vdf_grid.addWidget(self.lbl_p, 0, 0)
        
        self.p_xy, self.img_p_xy, self.pgroup_p_xy, self.polys_p_xy, self.iso_p_xy, self.b_p_xy, self.scat_p_xy_all, self.scat_p_xy_val, self.cmap_p = make_panel("VDF SPAN-I θ-plane", 'Vy (km/s)', quant_cmap_p, CONTOUR_LEVELS_P, y_range=(-700, 700))
        self.p_xz, self.img_p_xz, self.pgroup_p_xz, self.polys_p_xz, self.iso_p_xz, self.b_p_xz, self.scat_p_xz_all, self.scat_p_xz_val, _ = make_panel("VDF SPAN-I φ-plane", 'Vz (km/s)', quant_cmap_p, CONTOUR_LEVELS_P, y_range=(0, 1000))
        self.vdf_grid.addWidget(self.p_xy, 0, 1)
        self.vdf_grid.addWidget(self.p_xz, 0, 2)

        self.cbar_widget_p = pg.GraphicsLayoutWidget()
        self.cbar_widget_p.setFixedWidth(95)
        self.cbar_p = pg.ColorBarItem(values=(0.0, 8.0), label='log₁₀ f(v)')
        self.cbar_p.setColorMap(quant_cmap_p)
        self.cbar_p.setImageItem([self.img_p_xy, self.img_p_xz])
        self.cbar_widget_p.addItem(self.cbar_p)
        self.vdf_grid.addWidget(self.cbar_widget_p, 0, 3)

        # --- ALPHAS (Row 1) ---
        if self.is_dual:
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
            self.cbar_widget_he.setFixedWidth(95)
            self.cbar_he = pg.ColorBarItem(values=(0.0, 5.0), label='log₁₀ f(v)')
            self.cbar_he.setColorMap(quant_cmap_he)
            self.cbar_he.setImageItem([self.img_he_xy, self.img_he_xz])
            self.cbar_widget_he.addItem(self.cbar_he)
            self.vdf_grid.addWidget(self.cbar_widget_he, 1, 3)

        self.p_xz.setXLink(self.p_xy)
        main_layout.addLayout(self.vdf_grid, stretch=4)

        # BOTTOM PANEL: NATIVE PYQTGRAPH TIME SERIES STACK
        self.ts_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.ts_widget, stretch=4)

        self.setup_native_timeseries_stack(date_str)
        self.update_vdf(self.times_ref[0])

    def on_species_toggled(self):
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

        if self.is_dual:
            self.lbl_he.setVisible(show_he)
            self.he_xy.setVisible(show_he)
            self.he_xz.setVisible(show_he)
            self.cbar_widget_he.setVisible(show_he)

        if hasattr(self, 'lbl_time_val'):
            ts_str = self.lbl_time_val.text().replace(" UTC", "")
            try:
                dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                ts_unix = dt.replace(tzinfo=timezone.utc).timestamp()
                self.update_vdf(ts_unix, sync_lines=False)
            except Exception:
                pass

    def toggle_dark_mode(self, is_dark):
        bg_color = '#2b2b2b' if is_dark else '#ffffff'
        fg_color = '#e0e0e0' if is_dark else '#000000'
        grid_alpha = 0.2 if is_dark else 0.3

        pg.setConfigOption('background', bg_color)
        pg.setConfigOption('foreground', fg_color)

        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {bg_color};
            }}
            QWidget {{
                background-color: {bg_color};
                color: {fg_color};
            }}
            QGraphicsView {{
                background-color: {bg_color};
                border: none;
            }}
            QLabel, QRadioButton, QCheckBox {{
                background-color: transparent;
                color: {fg_color};
            }}
            QSplitter::handle {{
                background-color: {'#444444' if is_dark else '#e0e0e0'};
            }}
        """)

        if hasattr(self, 'ts_widget'): self.ts_widget.setBackground(bg_color)
        if hasattr(self, 'cbar_widget_p'): self.cbar_widget_p.setBackground(bg_color)
        if hasattr(self, 'cbar_widget_he'): self.cbar_widget_he.setBackground(bg_color)

        for cb_attr in ['cbar_p', 'cbar_he']:
            if hasattr(self, cb_attr):
                cbar = getattr(self, cb_attr)
                if hasattr(cbar, 'axis') and cbar.axis is not None:
                    cbar.axis.setPen(pg.mkPen(fg_color))
                    cbar.axis.setTextPen(pg.mkPen(fg_color))

        all_plots = []
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

    def on_mode_changed(self, button):
        if hasattr(self, 'lbl_time_val'):
            ts_str = self.lbl_time_val.text().replace(" UTC", "")
            try:
                dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                ts_unix = dt.replace(tzinfo=timezone.utc).timestamp()
                self.update_vdf(ts_unix, sync_lines=False)
            except Exception:
                pass

    def setup_native_timeseries_stack(self, date_str):
        label_style = {'font-size': '10pt'}
        Y_AXIS_FIXED_WIDTH = 75

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

        if self.mag_data is not None:
            t_m = self.mag_data['mag_times']
            self.curve_b_mag = self.p_mag.plot(t_m, self.mag_data['b_mag'], pen=pg.mkPen('#000000', width=1.2))
            self.p_mag.plot(t_m, self.mag_data['bx'], pen=pg.mkPen('#d62728', width=1.0))
            self.p_mag.plot(t_m, self.mag_data['by'], pen=pg.mkPen('#2ca02c', width=1.0))
            self.p_mag.plot(t_m, self.mag_data['bz'], pen=pg.mkPen('#1f77b4', width=1.0))

        vb_wave = ScrubViewBox(self)
        self.p_wave = self.ts_widget.addPlot(row=1, col=0, viewBox=vb_wave)
        self.p_wave.setLabel('left', 'Wave Power<br>Freq (Hz)', **label_style)
        self.p_wave.getAxis('left').setWidth(Y_AXIS_FIXED_WIDTH)
        self.p_wave.setLogMode(x=False, y=False)
        self.p_wave.hideAxis('bottom')
        self.p_wave.setXLink(self.p_mag)

        lbl_wave_spacer = self.ts_widget.addLabel("", row=1, col=1)
        lbl_wave_spacer.setMaximumWidth(45)

        if self.wave_data is not None and self.wave_data.get('power') is not None:
            img_wave = pg.ImageItem()
            self.p_wave.addItem(img_wave)
            img_wave.setColorMap(pg.colormap.get('turbo'))
            
            t_w = self.wave_data['times']
            freqs = self.wave_data['freqs']
            pw_raw = self.wave_data['power']

            log_f_orig = np.log10(np.maximum(freqs, 1e-4))
            log_f_min, log_f_max = log_f_orig[0], log_f_orig[-1]
            log_f_uniform = np.linspace(log_f_min, log_f_max, 200)

            f_interp = interp1d(log_f_orig, np.log10(np.clip(pw_raw, 1e-3, 1e3)), axis=1, bounds_error=False, fill_value=-3.0)
            pw_log_uniform = f_interp(log_f_uniform)

            img_wave.setImage(pw_log_uniform)
            rect = QtCore.QRectF(t_w[0], log_f_min, t_w[-1] - t_w[0], log_f_max - log_f_min)
            img_wave.setRect(rect)

            axis_wave = self.p_wave.getAxis('left')
            ticks = []
            for exp in range(-2, 4):
                if log_f_min <= exp <= log_f_max:
                    lbl = "0.1" if exp == -1 else ("1" if exp == 0 else ("10" if exp == 1 else f"10^{exp}"))
                    ticks.append((exp, lbl))
            axis_wave.setTicks([ticks])

        vb_ellip = ScrubViewBox(self)
        self.p_ellip = self.ts_widget.addPlot(row=2, col=0, viewBox=vb_ellip)
        self.p_ellip.setLabel('left', 'Ellipticity<br>Freq (Hz)', **label_style)
        self.p_ellip.getAxis('left').setWidth(Y_AXIS_FIXED_WIDTH)
        self.p_ellip.setLogMode(x=False, y=False)
        self.p_ellip.hideAxis('bottom')
        self.p_ellip.setXLink(self.p_mag)

        lbl_ellip_spacer = self.ts_widget.addLabel("", row=2, col=1)
        lbl_ellip_spacer.setMaximumWidth(45)

        if self.wave_data is not None and self.wave_data.get('ellip') is not None:
            img_ellip = pg.ImageItem()
            self.p_ellip.addItem(img_ellip)
            img_ellip.setColorMap(make_bipolar_colormap())

            el_raw = self.wave_data['ellip']
            log_f_orig = np.log10(np.maximum(freqs, 1e-4))
            log_f_min, log_f_max = log_f_orig[0], log_f_orig[-1]
            log_f_uniform = np.linspace(log_f_min, log_f_max, 200)

            el_interp = interp1d(log_f_orig, np.clip(el_raw, -1.0, 1.0), axis=1, bounds_error=False, fill_value=0.0)
            el_uniform = el_interp(log_f_uniform)

            img_ellip.setImage(el_uniform, levels=(-1.0, 1.0))
            rect = QtCore.QRectF(t_w[0], log_f_min, t_w[-1] - t_w[0], log_f_max - log_f_min)
            img_ellip.setRect(rect)

            axis_ellip = self.p_ellip.getAxis('left')
            ticks = []
            for exp in range(-2, 4):
                if log_f_min <= exp <= log_f_max:
                    lbl = "0.1" if exp == -1 else ("1" if exp == 0 else ("10" if exp == 1 else f"10^{exp}"))
                    ticks.append((exp, lbl))
            axis_ellip.setTicks([ticks])

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

        if self.moments_data is not None:
            t_mom = self.moments_data['times']
            if self.moments_data.get('v_bulk') is not None:
                self.curve_vr = self.p_mom.plot(t_mom, self.moments_data['v_bulk'], pen=pg.mkPen('#000000', width=1.5))
            if self.moments_data.get('v_alfven') is not None:
                self.p_mom.plot(t_mom, self.moments_data['v_alfven'], pen=pg.mkPen('#1f77b4', width=1.2))

        self.v_lines = []
        for p in [self.p_mag, self.p_wave, self.p_ellip, self.p_mom]:
            vl = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=1.5, style=QtCore.Qt.SolidLine))
            p.addItem(vl)
            self.v_lines.append(vl)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PSP SPAN-I Dynamic VDF Browser")
    parser.add_argument("--date", type=str, default="2020-01-29", help="Date YYYY-MM-DD")
    parser.add_argument("--species", type=str, default="both", help="'00', '0a', or 'both'")
    parser.add_argument("--data-dir", type=str, default=None, help="Optional local directory containing staged CDFs/CSVs")
    parser.add_argument("--dark-mode", action="store_true", help="Enable dark mode theme")
    
    parser.add_argument("--live-cdf", action="store_true", help="Bypass .json/.npz files and slice raw CDFs on-the-fly directly from memory")
    parser.add_argument("--test", action="store_true", help="Explicitly load partial pre-sliced test files (_TEST_*.npz)")
    parser.add_argument("--test-hours", type=float, default=1.0, help="Target duration in hours for test slice lookup (default: 1.0)")
    parser.add_argument("--vdf-slice-file", type=str, default=None, help="Direct path override for a specific pre-sliced file")
    args = parser.parse_args()

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

    species_tag = normalize_species_tag(args.species)
    print(f"Loading data for {args.date}...")
    
    presliced_vdf = None
    vdf_data_p = None
    vdf_data_he = None

    if not args.live_cdf:
        presliced_vdf = load_presliced_vdf(
            args.date,
            use_test=args.test,
            test_hours=args.test_hours,
            explicit_file=args.vdf_slice_file
        )

    if presliced_vdf is None:
        try:
            vdf_data_p = load_live_vdf_cdf(args.date, species_tag='00')
        except Exception as e:
            print(f"--> Proton CDF load failed: {e}")

        if species_tag in ['01', '0a', 'both'] and vdf_data_p is not None:
            try:
                vdf_data_he = load_live_vdf_cdf(args.date, species_tag='0a', proton_geom=vdf_data_p['geom'])
            except Exception as e:
                print(f"--> Alpha CDF load warning: {e}")

    mag_file = None
    mag_data = None
    try:
        mag_file = resolve_psp_file(args.date, file_type='mag')
        mag_data = load_psp_mag_cdf(mag_file)
    except Exception as e:
        print(f"--> MAG CDF load warning: {e}")

    wave_data = load_wave_analysis_data(args.date)
    hh_data = load_hammerhead_data(args.date)
    lfr_data = load_lfr_density_data(args.date)
    moments_data = load_moments_data(args.date, mag_data=mag_data)

    browser = VDFMagDynamicBrowser(
        vdf_data_p=vdf_data_p, vdf_data_he=vdf_data_he, mag_data=mag_data, wave_data=wave_data,
        hh_data=hh_data, lfr_data=lfr_data, moments_data=moments_data,
        date_str=args.date, presliced_vdf=presliced_vdf
    )

    if args.dark_mode:
        browser.cb_dark.setChecked(True)

    browser.show()
    sys.exit(app.exec_())
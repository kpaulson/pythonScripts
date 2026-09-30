import argparse
import os
import glob
import sys
import gc
import warnings
import platform
import numpy as np
import pandas as pd
import cdflib
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import time
import concurrent.futures
from multiprocessing import Manager
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console

# --- Memory Resource Management ---
try:
    import resource
except ImportError:
    resource = None

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import config
from waveResonance.resonance_io import smart_cdf

# Import Spacecraft Potential Tools
try:
    from PSP.psp_spacecraftPotential_tools import (
        load_auth_credentials,
        get_or_fetch_vdc_files,
        load_vdc_potential
    )
except ImportError:
    from psp_spacecraftPotential_tools import (
        load_auth_credentials,
        get_or_fetch_vdc_files,
        load_vdc_potential
    )


def limit_memory(max_gb=8.0):
    """Prevents any single worker process from exceeding memory limits."""
    if resource is not None and platform.system() != 'Windows':
        max_bytes = int(max_gb * 1024 * 1024 * 1024)
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, hard))


def get_ram_usage_str():
    """Returns formatted peak RAM usage for the calling process."""
    if resource is not None and platform.system() != 'Windows':
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return f"{peak_kb / (1024 * 1024):.2f} GB"
    return "N/A"


def project_velocity_to_b0(v_vec, b_vec, enforce_antisunward=True):
    """
    Projects velocity parallel to B0.
    If enforce_antisunward=True, flips b_hat when B_radial is inward,
    ensuring v_parallel doesn't flip sign due to sector reversals or switchbacks.
    """
    b_mag = np.linalg.norm(b_vec, axis=-1, keepdims=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        b_hat = np.where(b_mag > 0, b_vec / b_mag, np.nan)
        
        if enforce_antisunward:
            # PSP +X points SUNWARD. Anti-sunward flow has Vx < 0.
            sunward_mask = (b_vec[:, 0] > 0)
            b_hat[sunward_mask] = -b_hat[sunward_mask]
        
    v_parallel = np.einsum('ti,ti->t', v_vec, b_hat)
    return v_parallel, np.squeeze(b_mag)

    
def load_lfr_data(lfr_file):
    """Loads electron density and core temperature from mission LFR CDF."""
    with smart_cdf(lfr_file) as cdf:
        ne = cdf.varget('electronDensity')
        Te = cdf.varget('electronTemperatureCore')
        lfr_unix = cdflib.cdfepoch.unixtime(cdf.varget('Epoch'))
    return lfr_unix, ne, Te
    
def load_mission_ephemeris(ephem_file):
    """Loads mission-long heliocentric position CDF and returns a 1D interpolator for r_au."""
    if not os.path.exists(ephem_file):
        print(f"    [!] Ephemeris CDF not found at: {ephem_file}")
        return None

    try:
        with smart_cdf(ephem_file) as cdf_e:
            e_time_raw = cdf_e.varget('epoch')
            rad_au = cdf_e.varget('RAD_AU')

            # Convert CDF Epochs to Unix Time
            e_unix = cdflib.cdfepoch.unixtime(e_time_raw)

            # Clean fill values
            valid = np.isfinite(rad_au) & (rad_au > 0) & (rad_au < 2.0)
            if not np.any(valid):
                return None

            return interp1d(e_unix[valid], rad_au[valid], bounds_error=False, fill_value="extrapolate")
    except Exception as e:
        print(f"    [!] Error loading ephemeris CDF: {e}")
        return None


def load_daily_vdc_data(y, m, d, vdc_dir=None, auth=None):
    """
    Discovers local/cached single-ended wf_vdc CDFs for a target day or 
    fetches missing ones remotely, then extracts spacecraft potential (V_sc) 
    and psp_fld_l2_quality_flags.
    """
    vdc_files = get_or_fetch_vdc_files(y, m, d, vdc_root=vdc_dir, auth=auth)
    if not vdc_files:
        return np.array([]), np.array([]), (np.array([]), np.array([]))
    
    vdc_unix, v_avg_dc = load_vdc_potential(vdc_files, verbose=False)
    
    qf_times = []
    qf_vals = []
    for vf in vdc_files:
        try:
            with smart_cdf(vf) as cdf:
                # OLD: if 'psp_fld_l2_quality_flags' in cdf.varnames():
                zvars = cdf.cdf_info().zVariables
                if 'psp_fld_l2_quality_flags' in zvars:
                    qf = cdf.varget('psp_fld_l2_quality_flags')
                    qf_t = cdflib.cdfepoch.unixtime(cdf.varget('Epoch'))
                    qf_times.append(qf_t)
                    qf_vals.append(qf)
        except Exception:
            pass
            
    if qf_times:
        qf_unix = np.concatenate(qf_times)
        qf_flags = np.concatenate(qf_vals)
        sort_idx = np.argsort(qf_unix)
        return vdc_unix, v_avg_dc, (qf_unix[sort_idx], qf_flags[sort_idx])
        
    return vdc_unix, v_avg_dc, (np.array([]), np.array([]))


def extract_wave_properties(freqs, power_b, power_e, mask_rh, theta_2d=None):
    """Extracts frequency moments and power-weighted wave normal angles."""
    n_times = power_b.shape[0]
    f_peak = np.full(n_times, np.nan)
    f_weighted = np.full(n_times, np.nan)
    f_bandwidth = np.full(n_times, np.nan)
    p_b_sum = np.full(n_times, np.nan)
    p_e_sum = np.full(n_times, np.nan)
    theta_weighted = np.full(n_times, np.nan)
    
    for t in range(n_times):
        rh_idx = np.where(mask_rh[t, :] == 1)[0]
        if len(rh_idx) == 0:
            continue
            
        t_freqs = freqs[rh_idx]
        t_power_b = power_b[t, rh_idx]
        t_power_e = power_e[t, rh_idx]
        
        valid_p = np.isfinite(t_power_b) & np.isfinite(t_power_e)
        t_freqs = t_freqs[valid_p]
        t_power_b = t_power_b[valid_p]
        t_power_e = t_power_e[valid_p]
        
        if len(t_power_b) == 0 or np.sum(t_power_b) == 0:
            continue
            
        f_peak[t] = t_freqs[np.argmax(t_power_b)]
        pb_tot = np.sum(t_power_b)
        f_weighted[t] = np.sum(t_freqs * t_power_b) / pb_tot
        
        variance = np.sum(t_power_b * (t_freqs - f_weighted[t])**2) / pb_tot
        f_bandwidth[t] = np.sqrt(max(0.0, variance))
        p_b_sum[t] = pb_tot
        p_e_sum[t] = np.sum(t_power_e)

        if theta_2d is not None:
            t_theta = theta_2d[t, rh_idx][valid_p]
            valid_th = np.isfinite(t_theta)
            if np.any(valid_th):
                theta_weighted[t] = np.sum(t_theta[valid_th] * t_power_b[valid_th]) / np.sum(t_power_b[valid_th])
        
    return f_peak, f_weighted, f_bandwidth, p_b_sum, p_e_sum, theta_weighted


def interp_with_gap_limit(x, y, x_new, max_gap=120):
    """Interpolates y(x) onto x_new, masking gaps larger than max_gap seconds."""
    f = interp1d(x, y, bounds_error=False, fill_value=np.nan)
    y_new = f(x_new)
    
    gaps = np.diff(x)
    large_gap_indices = np.where(gaps > max_gap)[0]
    
    for idx in large_gap_indices:
        gap_start = x[idx]
        gap_end = x[idx + 1]
        y_new[(x_new > gap_start) & (x_new < gap_end)] = np.nan
        
    return y_new


def interp_nearest_flags_with_gap_limit(x, y, x_new, max_gap=120):
    """Performs nearest-neighbor interpolation for discrete bitwise flags."""
    if len(x) == 0:
        return np.full(len(x_new), 0, dtype=int)
        
    f = interp1d(x, y, kind='nearest', bounds_error=False, fill_value=0)
    y_new = f(x_new).astype(int)
    
    gaps = np.diff(x)
    large_gap_indices = np.where(gaps > max_gap)[0]
    
    for idx in large_gap_indices:
        gap_start = x[idx]
        gap_end = x[idx + 1]
        y_new[(x_new > gap_start) & (x_new < gap_end)] = 1025
        
    return y_new
    
def compute_in_situ_f_RC(v_sc_array, r_au_array=None):
    """Computes dynamic preamplifier RC corner frequency for wave masking."""
    T_ph_eV = 1.5
    I_ph0_1AU = 10e-6
    C_stray = 36e-12

    v_sc_clean = np.clip(np.nan_to_num(v_sc_array, nan=0.0), 0.0, 15.0)
    r_au_clean = np.clip(np.nan_to_num(r_au_array, nan=1.0), 0.04, 1.2) if r_au_array is not None else np.ones_like(v_sc_clean)

    I_ph0_r = I_ph0_1AU / (r_au_clean ** 2)
    R_sh = (T_ph_eV / I_ph0_r) * np.exp(v_sc_clean / T_ph_eV)
    f_RC = 1.0 / (2.0 * np.pi * R_sh * C_stray)
    return np.maximum(f_RC, 1e-3)
    
def fetch_literature_event_vdc(args, auth_creds):
    """
    Queries true CDF-derived V_sc potential for Encounter 1 (Nov 3-4, 2018) literature 
    events using load_daily_vdc_data() and exports literature_events_context.csv.
    """
    print("\n[*] Fetching true Encounter 1 in-situ V_sc for literature comparison events...")

    # Literature event targets (Mozer+ 2020 & Karbashewski+ 2023)
    lit_events = [
        # Mozer et al. (2020) Events [Nov 4, 2018]
        {"paper": "Mozer+2020", "event": "1Hz_Turbulence", "date": "2018-11-04", "iso_time": "2018-11-04T05:32:25", "f_sc_Hz": 1.0,  "L_eff_m": 1.20, "L_eff_err": 0.50, "delta_B_nT": 10.0},
        {"paper": "Mozer+2020", "event": "3Hz_MHD",        "date": "2018-11-04", "iso_time": "2018-11-04T13:58:49", "f_sc_Hz": 3.2,  "L_eff_m": 0.98, "L_eff_err": 0.31, "delta_B_nT": 1.4},
        {"paper": "Mozer+2020", "event": "4Hz_AIC",        "date": "2018-11-04", "iso_time": "2018-11-04T18:36:00", "f_sc_Hz": 4.3,  "L_eff_m": 2.78, "L_eff_err": 0.51, "delta_B_nT": 3.0},
        {"paper": "Mozer+2020", "event": "4Hz_Whistler",   "date": "2018-11-04", "iso_time": "2018-11-04T18:36:00", "f_sc_Hz": 4.5,  "L_eff_m": 1.68, "L_eff_err": 0.51, "delta_B_nT": 3.0},
        {"paper": "Mozer+2020", "event": "10Hz_Whistler",  "date": "2018-11-04", "iso_time": "2018-11-04T12:00:00", "f_sc_Hz": 10.5, "L_eff_m": 1.51, "L_eff_err": 0.50, "delta_B_nT": 1.5},
        {"paper": "Mozer+2020", "event": "20Hz_Whistler",  "date": "2018-11-04", "iso_time": "2018-11-04T12:00:00", "f_sc_Hz": 20.0, "L_eff_m": 3.72, "L_eff_err": 0.71, "delta_B_nT": 0.3},
        {"paper": "Mozer+2020", "event": "35Hz_Whistler",  "date": "2018-11-04", "iso_time": "2018-11-04T12:00:00", "f_sc_Hz": 35.0, "L_eff_m": 3.75, "L_eff_err": 0.70, "delta_B_nT": 0.25},
        {"paper": "Mozer+2020", "event": "59Hz_Whistler",  "date": "2018-11-04", "iso_time": "2018-11-04T15:23:06", "f_sc_Hz": 59.0, "L_eff_m": 3.56, "L_eff_err": 0.51, "delta_B_nT": 0.22},

        # Karbashewski et al. (2023) Events [Nov 3, 2018 Window]
        {"paper": "Karbashewski+2023", "event": "Burst_10:19:15", "date": "2018-11-03", "iso_time": "2018-11-03T10:19:15", "f_sc_Hz": 120.0, "L_eff_m": 23.5, "L_eff_err": 5.0, "delta_B_nT": 0.5},
        {"paper": "Karbashewski+2023", "event": "Burst_10:30:30", "date": "2018-11-03", "iso_time": "2018-11-03T10:30:30", "f_sc_Hz": 100.0, "L_eff_m": 17.8, "L_eff_err": 4.0, "delta_B_nT": 0.5},
        {"paper": "Karbashewski+2023", "event": "Burst_10:34:18", "date": "2018-11-03", "iso_time": "2018-11-03T10:34:18", "f_sc_Hz": 150.0, "L_eff_m": 22.0, "L_eff_err": 4.0, "delta_B_nT": 0.5},
        {"paper": "Karbashewski+2023", "event": "Burst_10:43:47", "date": "2018-11-03", "iso_time": "2018-11-03T10:43:47", "f_sc_Hz": 180.0, "L_eff_m": 16.0, "L_eff_err": 4.0, "delta_B_nT": 0.5},
    ]

    # Load potential data for Enc 1 dates (2018-11-03 & 2018-11-04) using existing fetch helper
    vdc_cache = {}
    for date_str in ["2018-11-03", "2018-11-04"]:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        v_unix, v_dc, _ = load_daily_vdc_data(
            dt.strftime('%Y'), dt.strftime('%m'), dt.strftime('%d'), 
            vdc_dir=args.vdc_dir, auth=auth_creds
        )
        if len(v_unix) > 0:
            vdc_cache[date_str] = (v_unix, v_dc)
            print(f"    [+] Loaded {len(v_unix):,} V_sc records for {date_str}")
        else:
            print(f"    [!] Could not load V_sc CDF for {date_str}")

    # Interpolate true in-situ V_sc for each event epoch
    results = []
    for ev in lit_events:
        d_str = ev["date"]
        target_epoch = pd.to_datetime(ev["iso_time"]).timestamp()
        
        v_sc_val = np.nan
        if d_str in vdc_cache:
            v_u, v_d = vdc_cache[d_str]
            # Use interp1d with 120s max gap limit to get exact in-situ V_sc
            v_sc_val = float(interp_with_gap_limit(v_u, v_d, np.array([target_epoch]), max_gap=120)[0])

        ev_res = ev.copy()
        ev_res["V_sc_volts"] = v_sc_val
        results.append(ev_res)

    df_out = pd.DataFrame(results)
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "literature_events_context.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"    [+] Saved literature event context file: {out_csv}\n")
    return df_out


def process_day(current_date, args, lfr_tuple, f_interp_r_au, auth_creds, slot_queue, status_queue):
    """Worker task: Extracts calibration data for a single day."""
    limit_memory(8.0)
    
    slot_id = slot_queue.get()
    y, m, d = current_date.strftime('%Y'), current_date.strftime('%m'), current_date.strftime('%d')
    date_str = f"{y}-{m}-{d}"
    
    DRIVE_ROOT = config.get_drive_path()
    v_config = VERSION_PAIRS[args.version]
    v_analysis = v_config["v_analysis"]
    v_wave = v_config["v_wave"]

    ANALYSIS_IN_ROOT_V1 = f'{DRIVE_ROOT}/Research/PSP/WaveAnalysis/WaveAnalysis_Files/v{v_analysis}/'
    ANALYSIS_IN_ROOT_V2 = f'{DRIVE_ROOT}/Research/PSP/WaveAnalysis/WaveAnalysis_Files/v{v_wave}/'

    status_queue.put({
        'slot_id': slot_id, 'day': date_str, 'chunk': 'Initializing', 
        'overlaps': 0, 'ram': get_ram_usage_str(), 'status': 'Starting'
    })

    # Locate Hamstring CDF
    ham_pattern = os.path.join(args.ham_dir, f"hamstring_{y}-{m}-{d}*.cdf")
    ham_files = sorted(glob.glob(ham_pattern))
    
    if not ham_files:
        status_queue.put({
            'slot_id': slot_id, 'day': date_str, 'chunk': 'None', 
            'overlaps': 0, 'ram': get_ram_usage_str(), 'status': 'No Hamstring File'
        })
        slot_queue.put(slot_id)
        return None, []
        
    ham_file = ham_files[-1]
    
    # Load single-ended wf_vdc Spacecraft Potential AND Quality Flags for target day
    vdc_unix, v_avg_dc, (qf_unix, qf_flags) = load_daily_vdc_data(y, m, d, vdc_dir=args.vdc_dir, auth=auth_creds)
    vdc_loaded = len(vdc_unix) > 0
    qf_loaded = len(qf_unix) > 0

    lfr_unix, ne_lfr, te_lfr = lfr_tuple
    lfr_loaded = len(lfr_unix) > 0
    
    v3_pattern = os.path.join(DRIVE_ROOT, f"Research/PSP/WaveAnalysis/WaveAnalysis_Files/v3.4/{y}/{m}/PSP_WaveResonance_1D_{y}-{m}-{d}_v3.4.cdf")
    v3_files = sorted(glob.glob(v3_pattern))
    
    t_b0_sc_unix, b0_sc_daily = np.array([]), None
    if v3_files:
        try:
            with smart_cdf(v3_files[-1]) as cdf_v3:
                zvars = cdf_v3.cdf_info().zVariables
                if 'B0_SC' in zvars:
                    t_b0_sc_unix = cdflib.cdfepoch.unixtime(cdf_v3.varget('Epoch'))
                    b0_sc_daily = cdf_v3.varget('B0_SC')  # [N_times, 3] -> (Bx_sc, By_sc, Bz_sc)
        except Exception as e:
            pass
    
    day_results = []
    missing_chunks = []
    total_overlaps = 0

    try:
        with smart_cdf(ham_file) as cdf_h:
            n_ham    = cdf_h.varget('n_ham')
            vx_ham   = cdf_h.varget('vx_inst_ham')
            vy_ham   = cdf_h.varget('vy_inst_ham')
            vz_ham   = cdf_h.varget('vz_inst_ham')
            tpar_ham = cdf_h.varget('Tpar_ham')
            bx_inst  = cdf_h.varget('Bx_inst')
            by_inst  = cdf_h.varget('By_inst')
            bz_inst  = cdf_h.varget('Bz_inst')
            ham_unix = cdflib.cdfepoch.unixtime(cdf_h.varget('Epoch'))
            
        v_inst_ham = np.column_stack((vx_ham, vy_ham, vz_ham))
        b0_inst    = np.column_stack((bx_inst, by_inst, bz_inst))
        v_para_bulk, B0_mag = project_velocity_to_b0(v_inst_ham, b0_inst)
        
        for hr in [0, 6, 12, 18]:
            hr_str = f"{hr:02d}:00"
            iso_chunk_target = f"{y}-{m}-{d}T{hr:02d}:00"
            
            status_queue.put({
                'slot_id': slot_id, 'day': date_str, 'chunk': hr_str, 
                'overlaps': total_overlaps, 'ram': get_ram_usage_str(), 'status': 'Processing'
            })
            
            dt_start = datetime(int(y), int(m), int(d), hr, 0, 0)
            dt_end = dt_start + timedelta(hours=6)
            t_chunk_start = dt_start.timestamp()
            t_chunk_end = dt_end.timestamp()

            ham_in_chunk = (ham_unix >= t_chunk_start) & (ham_unix < t_chunk_end) & np.isfinite(n_ham) & (n_ham > 0.0001)
            has_hammerheads = np.any(ham_in_chunk)

            wave_v1_file = os.path.join(ANALYSIS_IN_ROOT_V1, y, m, f"PSP_WaveAnalysis_{y}-{m}-{d}_{hr:02d}00_v{v_analysis}.cdf")
            wave_v2_file = os.path.join(ANALYSIS_IN_ROOT_V2, y, m, f"PSP_WaveResonance_2D_{y}-{m}-{d}_{hr:02d}00_v{v_wave}.cdf")
            
            wave_files_exist = os.path.exists(wave_v1_file) and os.path.exists(wave_v2_file)

            if has_hammerheads and not wave_files_exist:
                missing_chunks.append(iso_chunk_target)
                continue

            if not wave_files_exist:
                continue
                
            with smart_cdf(wave_v1_file) as cdf_v1, smart_cdf(wave_v2_file) as cdf_v2:
                v2_vars = cdf_v2.cdf_info().zVariables
                v1_vars = cdf_v1.cdf_info().zVariables

                # 1. READ wave_unix FIRST BEFORE REFERENCING IT
                wave_unix = cdflib.cdfepoch.unixtime(cdf_v2.varget('Epoch'))

                # =========================================================================
                # INTERPOLATE DAILY B0_SC ONTO CHUNK wave_unix
                # =========================================================================
                if b0_sc_daily is not None and len(t_b0_sc_unix) > 0:
                    bx_interp = interp_with_gap_limit(t_b0_sc_unix, b0_sc_daily[:, 0], wave_unix, max_gap=args.gap_limit)
                    by_interp = interp_with_gap_limit(t_b0_sc_unix, b0_sc_daily[:, 1], wave_unix, max_gap=args.gap_limit)
                    bz_interp = interp_with_gap_limit(t_b0_sc_unix, b0_sc_daily[:, 2], wave_unix, max_gap=args.gap_limit)
                    
                    b0_sc_chunk = np.column_stack((bx_interp, by_interp, bz_interp))
                    bx_sc = b0_sc_chunk[:, 0]               # Sunward component (+X_sc)
                    b0_sc_mag = np.linalg.norm(b0_sc_chunk, axis=-1)
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        cos_theta_bx = np.clip(np.abs(bx_sc) / np.maximum(b0_sc_mag, 1e-3), 0.0, 1.0)
                        theta_BX_raw = np.degrees(np.arccos(cos_theta_bx))
                else:
                    theta_BX_raw = np.full(len(wave_unix), np.nan)

                try:
                    freqs = cdf_v2.varget('frequencies')
                except:
                    freqs = cdf_v2.varget('Frequencies')
                    
                power_b = cdf_v1.varget('wave_power_B')
                power_e = cdf_v1.varget('wave_power_E') 

                # --- EXTRACT REAL V_sw AND waveNormal FROM WaveResonance 2D CDF ---
                if 'v_sw_RTN' in v2_vars:
                    v_sw_rtn = cdf_v2.varget('v_sw_RTN')
                    v_sw_kms_sync = np.linalg.norm(v_sw_rtn, axis=-1)
                else:
                    v_sw_kms_sync = np.full(len(wave_unix), np.nan)

                if 'wavenumbers' in v2_vars:
                    k_bundled = cdf_v2.varget('wavenumbers') # [time, freq, 2]
                    k_para_2d = k_bundled[..., 0]
                    k_perp_2d = k_bundled[..., 1]
                    theta_kB_2d = np.degrees(np.arctan2(np.abs(k_perp_2d), np.abs(k_para_2d)))
                elif 'wave_normal' in v1_vars:
                    theta_kB_2d = cdf_v1.varget('wave_normal')
                else:
                    theta_kB_2d = None
                
                power_b[power_b < -1e30] = np.nan
                power_e[power_e < -1e30] = np.nan
                
                try:
                    masks_pol = cdf_v2.varget('masks_polarization')
                    mask_rh = masks_pol[..., 1]
                except (ValueError, KeyError):
                    mask_rh = cdf_v2.varget('mask_RH')
            
            # Sync particle & field vectors onto wave epoch timeline
            v_para_sync = interp_with_gap_limit(ham_unix, v_para_bulk, wave_unix, max_gap=args.gap_limit)
            n_ham_sync  = interp_with_gap_limit(ham_unix, n_ham, wave_unix, max_gap=args.gap_limit)
            tpar_sync   = interp_with_gap_limit(ham_unix, tpar_ham, wave_unix, max_gap=args.gap_limit)
            b0_sync     = interp_with_gap_limit(ham_unix, B0_mag, wave_unix, max_gap=args.gap_limit)
            
            # Sync Spacecraft DC Potential (V_sc)
            if vdc_loaded:
                v_sc_sync = interp_with_gap_limit(vdc_unix, v_avg_dc, wave_unix, max_gap=args.gap_limit)
            else:
                v_sc_sync = np.full(len(wave_unix), np.nan)

            # Sync Bitwise Quality Flags
            if qf_loaded:
                qf_sync = interp_nearest_flags_with_gap_limit(qf_unix, qf_flags, wave_unix, max_gap=args.gap_limit)
            else:
                qf_sync = np.zeros(len(wave_unix), dtype=int)

            if lfr_loaded:
                f_interp_ne = interp1d(lfr_unix, ne_lfr, bounds_error=False, fill_value=np.nan)
                f_interp_te = interp1d(lfr_unix, te_lfr, bounds_error=False, fill_value=np.nan)
                ne_sync = f_interp_ne(wave_unix)
                te_sync = f_interp_te(wave_unix)
            else:
                ne_sync = np.full(len(wave_unix), np.nan)
                te_sync = np.full(len(wave_unix), np.nan)
            
            # BITWISE QUALITY FLAG MASKING (Bit 0 = Bias Sweep, Bit 10 = Anomalous Bias)
            bias_sweep_active = (qf_sync & 1) != 0
            anomalous_bias_active = (qf_sync & 1024) != 0
            clean_bias_operations = ~bias_sweep_active & ~anomalous_bias_active

            active_wave_mask = (
                mask_rh.any(axis=1) & 
                np.isfinite(n_ham_sync) & 
                (n_ham_sync > 0.0001) & 
                clean_bias_operations
            )
            valid_idx = np.where(active_wave_mask)[0]

            if len(valid_idx) == 0:
                continue
                
            t_valid      = wave_unix[valid_idx]
            p_b_valid    = power_b[valid_idx, :]
            p_e_valid    = power_e[valid_idx, :] 
            m_rh_valid   = mask_rh[valid_idx, :]
            v_p_valid    = v_para_sync[valid_idx]
            n_h_valid    = n_ham_sync[valid_idx]
            tp_valid     = tpar_sync[valid_idx]
            b0_valid     = b0_sync[valid_idx]
            v_sc_valid   = v_sc_sync[valid_idx]
            qf_valid     = qf_sync[valid_idx]
            ne_valid     = ne_sync[valid_idx]
            te_valid     = te_sync[valid_idx]
            v_sw_valid   = v_sw_kms_sync[valid_idx]
            th_kB_valid  = theta_kB_2d[valid_idx, :] if theta_kB_2d is not None else None
            
            fp, fw, fb, pb_sum, pe_sum, th_weighted = extract_wave_properties(
                freqs, p_b_valid, p_e_valid, m_rh_valid, theta_2d=th_kB_valid
            )
            
            Omega_cp = (1.602e-19 * (b0_valid * 1e-9)) / 1.673e-27
            v_para_sc_ms = v_p_valid * 1e3
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                omega_weighted = fw * 2 * np.pi
                k_para_weighted = (omega_weighted + Omega_cp) / v_para_sc_ms
                V_ph_sc_weighted = (omega_weighted / k_para_weighted) / 1e3
                
                B_meas = np.sqrt(pb_sum) # nT
                E_meas = np.sqrt(pe_sum) # mV/m
                
                V_measured = (E_meas * 1e-3) * 3.5                      # Volts
                E_theory = np.abs(V_ph_sc_weighted) * 1e3 * (B_meas * 1e-9) # V/m
                L_eff = V_measured / E_theory                           # Meters
                
                debye_length = 7.43 * np.sqrt(te_valid / ne_valid)
                v_th_para = np.sqrt((2 * 1.602e-19 * tp_valid) / 1.673e-27)
            
            iso_time_str = pd.to_datetime(t_valid, unit='s', utc=True).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            
            # Interpolate heliocentric distance [AU] onto valid wave timestamps
            r_au_valid = f_interp_r_au(t_valid) if f_interp_r_au is not None else np.full(len(t_valid), np.nan)
            
            # Slice Spacecraft Sunward B-field angle for active resonant points
            theta_BX_valid = theta_BX_raw[valid_idx]
            
            # =========================================================================
            # COMPUTE BROADBAND RECTIFYING B-FIELD POWER (f > f_RC)
            # =========================================================================
            r_au_chunk = f_interp_r_au(wave_unix) if f_interp_r_au is not None else np.ones_like(wave_unix)
            f_rc_chunk = compute_in_situ_f_RC(v_sc_sync, r_au_chunk)
            
            # Mask 2D power_b for frequencies exceeding local f_RC
            f_mask = freqs[None, :] > f_rc_chunk[:, None]
            b_high_freq_all = np.sqrt(np.nansum(np.where(f_mask, power_b, 0.0), axis=-1))
            b_high_freq_valid = b_high_freq_all[valid_idx]
            # =========================================================================

            df_chunk = pd.DataFrame({
                'iso_time': iso_time_str,
                'epoch': t_valid, 
                'r_au': r_au_valid,                  # Heliocentric Distance [AU]
                'theta_BX_deg': theta_BX_valid,      # Spacecraft Sunward B-field Angle [deg]
                'n_ham': n_h_valid, 
                'Tpar_ham_eV': tp_valid,
                'v_th_para_ms': v_th_para, 
                'v_para_sc_ms': v_para_sc_ms, 
                'v_sw_kms': v_sw_valid,             # Real In-Situ Bulk Solar Wind Speed [km/s]
                'theta_kB_deg': th_weighted,         # Real Wave Normal Angle [deg]
                'B0_mag_nT': b0_valid,
                'B_wave_nT': B_meas,
                'B_high_freq_nT': b_high_freq_valid, # Total Broadband Rectifying Magnetic Power (f > f_RC) [nT]
                'V_wave_volts': V_measured,
                'V_sc_volts': v_sc_valid,
                'quality_flags': qf_valid,
                'f_peak_Hz': fp, 
                'f_weighted_Hz': fw, 
                'f_bandwidth_Hz': fb,
                'k_para_weighted': k_para_weighted, 
                'V_ph_sc_weighted_kms': V_ph_sc_weighted,
                'L_eff_m': L_eff, 
                'ne_lfr': ne_valid, 
                'Te_lfr': te_valid, 
                'debye_length_m': debye_length
            })
            
            df_chunk = df_chunk.dropna(subset=['f_peak_Hz'])
            if not df_chunk.empty:
                day_results.append(df_chunk)
                total_overlaps += len(df_chunk)
            
    except Exception as e:
        status_queue.put({
            'slot_id': slot_id, 'day': date_str, 'chunk': 'Error', 
            'overlaps': total_overlaps, 'ram': get_ram_usage_str(), 'status': f'Failed: {e}'
        })
        slot_queue.put(slot_id)
        return None, missing_chunks
        
    finally:
        gc.collect()

    status_queue.put({
        'slot_id': slot_id, 'day': date_str, 'chunk': 'Done', 
        'overlaps': total_overlaps, 'ram': get_ram_usage_str(), 'status': 'Complete'
    })
    slot_queue.put(slot_id)

    if day_results:
        return pd.concat(day_results, ignore_index=True), missing_chunks
    return None, missing_chunks


def make_dashboard(worker_states, completed_days, total_days):
    """Renders a dynamic live multi-row UI panel."""
    table = Table(expand=True)
    table.add_column("Worker Slot", justify="center", style="cyan bold", no_wrap=True)
    table.add_column("Target Day", justify="center", style="green")
    table.add_column("Active Chunk", justify="center", style="yellow")
    table.add_column("Matches Extracted", justify="right", style="magenta")
    table.add_column("Peak RAM", justify="right", style="blue")
    table.add_column("Status", justify="left", style="white")

    for slot_id in sorted(worker_states.keys()):
        state = worker_states[slot_id]
        table.add_row(
            f"Worker {slot_id + 1}",
            state.get('day', 'Idle'),
            state.get('chunk', '-'),
            f"{state.get('overlaps', 0):,}",
            state.get('ram', '-'),
            state.get('status', 'Idle')
        )
        
    title_str = f"Hammerhead Calibration Pipeline | Overall Progress: {completed_days}/{total_days} Days ({completed_days/total_days*100:.1f}%)"
    return Panel(table, title=title_str, border_style="bold blue")


def main():
    args = setup()
    date_parts = args.date.split(' through ')
    start_date = datetime.strptime(date_parts[0].strip(), '%Y-%m-%d')
    end_date   = datetime.strptime(date_parts[1].strip(), '%Y-%m-%d')

    date_tag = start_date.strftime('%Y%m%d') if start_date == end_date else f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

    print(f"[*] Commencing Multiprocess Calibration Extraction ({args.workers} Workers): {start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}")
    
    auth_creds = load_auth_credentials()
    if auth_creds:
        print(f"[*] SSL Remote Credentials loaded for user: '{auth_creds[0]}'")
        
    # Automatically extracts true Enc 1 (2018) V_sc for literature comparison
    fetch_literature_event_vdc(args, auth_creds)

    lfr_file = os.path.join(args.lfr_dir, "spp_fld_lfr_mission_density.cdf")
    lfr_unix, ne_lfr, te_lfr = np.array([]), np.array([]), np.array([])
    
    if os.path.exists(lfr_file):
        print(f"[*] Loading consolidated mission LFR QTN data: {os.path.basename(lfr_file)}")
        try:
            lfr_unix, ne_lfr, te_lfr = load_lfr_data(lfr_file)
            print(f"    [+] Loaded {len(lfr_unix)} LFR QTN timestamps successfully.")
        except Exception as e:
            print(f"    [!] Error loading LFR file: {e}")

    lfr_tuple = (lfr_unix, ne_lfr, te_lfr)
    
    # --- Load Mission Ephemeris for Heliocentric Distance r_au ---
    ephem_file = os.path.join(
        config.get_drive_path(), 
        'Research/Data/AutoplotCache/https/cdaweb.gsfc.nasa.gov/sp_phys/data/psp/ephemeris/helio1hr/psp_helio1hr_position_20180813_v01.cdf'
    )
    f_interp_r_au = load_mission_ephemeris(ephem_file)
    if f_interp_r_au is not None:
        print(f"[*] Mission Ephemeris loaded successfully from: {os.path.basename(ephem_file)}")

    day_list = []
    curr = start_date
    while curr <= end_date:
        day_list.append(curr)
        curr += timedelta(days=1)

    total_days = len(day_list)
    completed_days = 0
    all_results = []
    all_missing_chunks = []

    manager = Manager()
    slot_queue = manager.Queue()
    status_queue = manager.Queue()
    
    for i in range(args.workers):
        slot_queue.put(i)
        
    worker_states = {i: {'day': 'Idle', 'chunk': '-', 'overlaps': 0, 'ram': '-', 'status': 'Idle'} for i in range(args.workers)}

    with Live(make_dashboard(worker_states, 0, total_days), refresh_per_second=4) as live:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_day, day_dt, args, lfr_tuple, f_interp_r_au, auth_creds, slot_queue, status_queue): day_dt 
                for day_dt in day_list
            }
            
            completed_futures = set()
            while completed_days < total_days:
                while not status_queue.empty():
                    msg = status_queue.get()
                    slot = msg['slot_id']
                    worker_states[slot].update(msg)

                done = [f for f in futures if f.done() and f not in completed_futures]
                for f in done:
                    completed_futures.add(f)
                    completed_days += 1
                    try:
                        res_df, missing_list = f.result()
                        if res_df is not None and not res_df.empty:
                            all_results.append(res_df)
                        if missing_list:
                            all_missing_chunks.extend(missing_list)
                    except Exception:
                        pass
                
                live.update(make_dashboard(worker_states, completed_days, total_days))
                time.sleep(0.1)

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Write Missing WaveResonance Chunks Audit File ---
    if all_missing_chunks:
        sorted_missing = sorted(list(set(all_missing_chunks)))
        missing_txt_path = os.path.join(args.out_dir, f"hammerhead_missing_waveResonance_{date_tag}.txt")
        with open(missing_txt_path, "w") as f_out:
            for iso_chunk in sorted_missing:
                f_out.write(f"{iso_chunk}\n")
        print(f"\n[!] Audit: Found {len(sorted_missing)} hammerhead windows missing WaveResonance CDFs.")
        print(f"    [+] Saved missing chunk reprocessing list to: {missing_txt_path}")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        
        out_filename = args.out if args.out else f"hammerhead_cal_{date_tag}.csv"
        out_path = os.path.join(args.out_dir, out_filename)
        
        final_df.to_csv(out_path, index=False)
        print(f"\n[+] Processing complete. Calibrations saved to: {out_path}")
        print(f"[*] Total Resonant Matches Extracted: {len(final_df):,}")

        if args.plot:
            print("\n[*] Triggering Diagnostic Plotter...")
            try:
                from plot_hammerhead_diagnostics import generate_diagnostic_plots
                generate_diagnostic_plots(out_path, gap_limit=args.gap_limit)
            except Exception as e:
                print(f"    [!] Plotting failed: {e}")
    else:
        print("\n[!] No matching wave/particle resonant overlaps found in the date range.")


VERSION_PAIRS = {
    "1.4": {"v_analysis": "1.4", "v_wave": "2.4"},
    "1.5": {"v_analysis": "1.5", "v_wave": "2.5"},
}


def setup():
    parser = argparse.ArgumentParser(description="Hammerhead Leff Calibration Data Extractor")
    
    DRIVE_ROOT = config.get_drive_path()
    default_ham_dir = f"{DRIVE_ROOT}/Research/PSP/Hammerheads/Hamstrings/cdf/v02/"
    default_out_dir = f"{DRIVE_ROOT}/Research/PSP/Hammerheads/resonanceFiles/"
    default_lfr_dir = f"{DRIVE_ROOT}/Research/PSP/FIELDS/LFR_Density/"
    
    default_spp_cache = os.path.join(
        DRIVE_ROOT, 
        'Research/Data/AutoplotCache/http/research.ssl.berkeley.edu/data/spp/data/sci'
    )
    default_vdc_dir = os.path.join(default_spp_cache, 'fields/l2/dfb_wf_vdc')
    
    parser.add_argument('--date', type=str, required=True, help="Format: 'YYYY-MM-DD through YYYY-MM-DD'")
    parser.add_argument('-v', '--version', type=str, default="1.4", 
                        choices=list(VERSION_PAIRS.keys()),
                        help="Pipeline calibration version pair")
    parser.add_argument('--workers', type=int, default=4, help="Number of concurrent worker processes")
    parser.add_argument('--ham_dir', type=str, default=default_ham_dir, help="Directory containing hamstring CDFs")
    parser.add_argument('--lfr_dir', type=str, default=default_lfr_dir, help="Directory containing LFR QTN CDFs")
    parser.add_argument('--vdc_dir', type=str, default=default_vdc_dir, help="Directory containing DFB wf_vdc potential CDFs")
    parser.add_argument('--out_dir', type=str, default=default_out_dir, help="Directory to save output CSV")
    parser.add_argument('--out', type=str, default=None, help="Output CSV filename")
    parser.add_argument('--gap_limit', type=int, default=20, help="Maximum gap (in seconds) allowed for interpolation")
    parser.add_argument('--plot', action='store_true', help="Automatically run diagnostic plots on completion")
    
    return parser.parse_args()


if __name__ == "__main__":
    main()
import argparse
import os
import glob
import warnings
from datetime import datetime, timedelta, timezone
import numpy as np
import sys
import gc
import scipy.ndimage as ndimage
import concurrent.futures
import resource
import time
from multiprocessing import Manager
from contextlib import redirect_stdout
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

# Look up one level to find config and shared modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import config
import resonance_io
from core_waves import kinematics, resonances, spice_manager

# Global paths from config
DRIVE_ROOT           = config.get_drive_path()
PSP_SHAREDDRIVE_ROOT = config.get_sharedDrivePSP_path()
SCI_ROOT             = config.get_berkeleyCacheData()

MAG_4SA_ROOT  = f'{SCI_ROOT}/fields/l2/mag_SC_4_Sa_per_Cyc/'
CSV_ROOT      = f'{PSP_SHAREDDRIVE_ROOT}/15min_DATA_Internal/_CURRENT_20260604/' 
TMP_ROOT      = f'{DRIVE_ROOT}/tmp/'
LFR_DENSITY_FILE = f'{DRIVE_ROOT}/Research/PSP/FIELDS/LFR_Density/spp_fld_lfr_mission_density.cdf'

class QueueLogger:
    """Redirects print statements from workers directly to the Rich console queue."""
    def __init__(self, queue):
        self.queue = queue
    def write(self, msg):
        msg_str = msg.strip()
        if msg_str:
            self.queue.put(msg_str)
    def flush(self):
        pass

def limit_memory(max_gb=8.0):
    """Prevents any single process worker from exceeding memory limits."""
    max_bytes = int(max_gb * 1024 * 1024 * 1024)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, hard))
          
def get_masked_weighted_avg(data_2d, mask_2d, power_2d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        weights = np.where(mask_2d & np.isfinite(data_2d) & np.isfinite(power_2d), power_2d, 0.0)
        weighted_sum = np.nansum(data_2d * weights, axis=1)
        sum_weights = np.nansum(weights, axis=1)
        return np.where(sum_weights > 0, weighted_sum / sum_weights, np.nan)

def smooth_mask(arr, size=(2, 20)):
    valid_mask = np.isfinite(arr).astype(float)
    arr_filled = np.where(np.isfinite(arr), arr, 0.0)
    s_arr = ndimage.uniform_filter(arr_filled, size=size, mode='nearest')
    s_weights = ndimage.uniform_filter(valid_mask, size=size, mode='nearest')
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(s_weights > 0, s_arr / s_weights, np.nan)

def get_encounter_from_date(target_dt, padding_days=1):
    for enc, (start_str, end_str) in config.ORBIT_DATES.items():
        start_dt = datetime.strptime(start_str, '%Y-%m-%d') - timedelta(days=padding_days)
        end_dt = datetime.strptime(end_str, '%Y-%m-%d') + timedelta(days=1 + padding_days) 
        if start_dt <= target_dt <= end_dt:
            return enc
    return None

def make_dashboard(worker_states, completed, total):
    """Renders a dynamic live multi-row UI panel for Wave Resonance."""
    table = Table(expand=True)
    table.add_column("Worker", justify="center", style="cyan bold", no_wrap=True)
    table.add_column("Date", justify="center", style="green")
    table.add_column("Chunk", justify="center", style="yellow")
    table.add_column("Status", justify="left", style="white")
    table.add_column("Peak RAM", justify="right", style="magenta")

    for slot_id in sorted(worker_states.keys()):
        state = worker_states[slot_id]
        table.add_row(
            f"Worker {slot_id + 1}",
            state.get('date', 'Idle'),
            state.get('chunk', '-'),
            state.get('status', 'Idle'),
            state.get('ram', '-')
        )
        
    pct = (completed / total * 100) if total > 0 else 0.0
    title_str = f"Parker Solar Probe Wave Resonance Pipeline | Progress: {completed}/{total} Days ({pct:.1f}%)"
    return Panel(table, title=title_str, border_style="bold magenta")
    
def set_terminal_tab_title(title_text):
    """Writes OSC escape sequences directly to the controlling terminal device,
    completely bypassing Python stdout and Rich console redirection."""
    try:
        if os.name == 'posix':
            with open('/dev/tty', 'w') as tty:
                tty.write(f"\033]0;{title_text}\007")
                tty.flush()
        elif os.name == 'nt':
            with open('CON', 'w') as tty:
                tty.write(f"\033]0;{title_text}\007")
                tty.flush()
    except Exception:
        pass  # Gracefully ignore if running in a non-interactive environment
        
def get_psp_leff(f_Hz, V_sc_volts, delta_B_nT):
    """
    Definitive PSP FIELDS Antenna Effective Length Correction.
    Returns L_eff in meters. STILL IN PROGRESS, KEEP AS PLACEHOLDER
    """
    L_phys = 3.5
    A = 19.14
    V0, alpha = 5.21, 0.904
    f0, beta  = 0.408, 0.010
    B0, gamma = 4.047, 1.139

    v_norm = max(V_sc_volts / V0, 1e-5)
    f_norm = max(f_Hz / f0, 1e-5)
    b_norm = max(delta_B_nT / B0, 0.0)

    v_term = (v_norm * np.exp(1.0 - v_norm)) ** alpha
    f_term = (f_norm * np.exp(1.0 - f_norm)) ** beta
    damping = 1.0 / (1.0 + (b_norm ** gamma))

    return L_phys + A * v_term * f_term * damping
    
def get_psp_leff_with_uncertainty(f_Hz, V_sc_volts, delta_B_nT, V_raw_volts=None, sigma_V_raw=0.01):
    """
    Computes dynamic effective antenna length L_eff, its 1-sigma uncertainty,
    and (optionally) calibrated electric field with full propagated error.
    
    Parameters:
        f_Hz          : Wave frequency (Hz)
        V_sc_volts    : Spacecraft potential (V)
        delta_B_nT    : Wave magnetic fluctuation amplitude (nT)
        V_raw_volts   : Optional raw antenna voltage (V)
        sigma_V_raw   : Relative uncertainty in raw voltage (default 1%)
        
    Returns:
        L_eff (m), sigma_L_eff (m)  [and E_cal (V/m), sigma_E (V/m) if V_raw is provided]
    """
    # Converged Model 5 Parameters
    L_phys = 3.5
    A = 19.140
    V0, alpha = 5.210, 0.904
    f0, beta  = 0.408, 0.010
    B0, gamma = 4.047, 1.139

    v_norm = np.maximum(V_sc_volts / V0, 1e-5)
    f_norm = np.maximum(f_Hz / f0, 1e-5)
    b_norm = np.maximum(delta_B_nT / B0, 0.0)

    v_term = (v_norm * np.exp(1.0 - v_norm)) ** alpha
    f_term = (f_norm * np.exp(1.0 - f_norm)) ** beta
    damping = 1.0 / (1.0 + (b_norm ** gamma))

    # Photoelectron expansion term
    L_expansion = A * v_term * f_term * damping
    L_eff = L_phys + L_expansion

    # Model Uncertainty Formulation:
    # 0.5m baseline hardware uncertainty + 15% relative error on expansion term
    sigma_base = 0.5
    epsilon_rel = 0.15
    sigma_L_eff = np.sqrt(sigma_base**2 + (epsilon_rel * L_expansion)**2)

    if V_raw_volts is None:
        return L_eff, sigma_L_eff

    # Propagate into calibrated Electric Field
    E_cal = V_raw_volts / L_eff
    rel_err_V = sigma_V_raw  # Fractional error (e.g. 0.01 = 1%)
    rel_err_L = sigma_L_eff / L_eff
    
    sigma_E = E_cal * np.sqrt(rel_err_V**2 + rel_err_L**2)

    return L_eff, sigma_L_eff, E_cal, sigma_E

def process_day(day_tuple, slot_queue, status_queue, log_queue):
    """Worker task: Processes 1 full day (all 4 chunks) and writes 2D & 1D CDF files."""
    current_date, args = day_tuple
    limit_memory(8.0) # Cap worker at 8 GB RAM
    
    slot_id = slot_queue.get() # Acquire dashboard worker slot
    y, m, d = current_date.strftime('%Y'), current_date.strftime('%m'), current_date.strftime('%d')
    date_str = f"{y}-{m}-{d}"
    
    v_minor = args.v_in.split('.')[-1]
    v_2d = f"2.{v_minor}"
    v_3d = f"3.{v_minor}"
    
    ANALYSIS_IN_ROOT     = f'{DRIVE_ROOT}/Research/PSP/WaveAnalysis/WaveAnalysis_Files/v{args.v_in}/'
    ANALYSIS_OUT_ROOT_2D = f'{DRIVE_ROOT}/Research/PSP/WaveAnalysis/WaveAnalysis_Files/v{v_2d}/'
    ANALYSIS_OUT_ROOT_1D = f'{DRIVE_ROOT}/Research/PSP/WaveAnalysis/WaveAnalysis_Files/v{v_3d}/'

    def update_ui(status_text, chunk_str="-"):
        """Sends progress state to the main rendering thread."""
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ram_str = f"{peak_kb / (1024 * 1024):.2f} GB"
        status_queue.put({
            'slot_id': slot_id, 'date': date_str, 'chunk': chunk_str,
            'status': status_text, 'ram': ram_str
        })

    # Route print statements to devnull or queue depending on --verbose flag
    out_stream = QueueLogger(log_queue) if args.verbose else open(os.devnull, 'w')

    try:
        with redirect_stdout(out_stream):
            update_ui("Checking Overwrite", "-")
            
            # --- 1. DAILY OVERWRITE CHECK ---
            v_out_dir_1d = os.path.join(ANALYSIS_OUT_ROOT_1D, y, m)
            out_file_1d = os.path.join(v_out_dir_1d, f"PSP_WaveResonance_1D_{y}-{m}-{d}_v{v_3d}.cdf")
            
            if not args.overwrite and os.path.exists(out_file_1d):
                if args.verbose:
                    print(f"[*] Skipping Day: {date_str} (1D File Exists)")
                update_ui("Skipped (1D Exists)", "-")
                return True
                
            if args.verbose:
                print(f"[=========== Processing Day: {date_str} ===========]")
            
            daily_d_out = {'Time': [], 'B0': [], 'V_sw': [], 'N_p': [], 'N_alpha': [], 'Density_Flag': []}
            daily_physics_out = {'V_A': [], 'int_LH': [], 'int_RH': [], 'int_Sn_pos': [], 'int_Sn_neg': []}
            
            avg_keys = [f'avg_{var}_{prefix}' for prefix in ['LH', 'RH'] for var in [
                'k_parallel', 'k_perp', 'V_ph', 'f_plasma', 'E_res_p', 'E_res_alpha', 
                'E_res_landau_p', 'E_res_anom_p', 'E_res_anom_alpha']]
            for k in avg_keys:
                daily_physics_out[k] = []
                
            chunks_processed = 0

            for hr in [0, 6, 12, 18]:
                start_dt = current_date.replace(hour=hr, minute=0)
                end_dt   = start_dt + timedelta(hours=6)
                hr_str   = start_dt.strftime('%H')
                chunk_str = f"{hr_str}:00"

                v_out_dir_2d = os.path.join(ANALYSIS_OUT_ROOT_2D, y, m)
                out_file_2d = os.path.join(v_out_dir_2d, f"PSP_WaveResonance_2D_{y}-{m}-{d}_{hr_str}00_v{v_2d}.cdf")

                update_ui("Resolving Files", chunk_str)
                
                wave_file = os.path.join(ANALYSIS_IN_ROOT, y, m, f"PSP_WaveAnalysis_{y}-{m}-{d}_{hr_str}00_v{args.v_in}.cdf")
                
                # --- 1. RESOLVE PLASMA SOURCE PATHS EARLY ---
                mag_file_pattern = os.path.join(MAG_4SA_ROOT, y, m, f"psp_fld_l2_mag_SC_4_Sa_per_Cyc_{y}{m}{d}_v*.cdf")
                matching_mag_files = sorted(glob.glob(mag_file_pattern))
                mag_file = matching_mag_files[-1] if matching_mag_files else None
                
                enc_num = get_encounter_from_date(start_dt)
                csv_path = os.path.join(CSV_ROOT, f"E{enc_num:02d}.csv") if enc_num is not None else None

                # --- 2D BYPASS MODE: Fast-track 1D generation from existing 2D CDFs ---
                if args.from_2d:
                    if not os.path.exists(out_file_2d):
                        if args.verbose: print(f"  [!] Skipping {chunk_str}: 2D file not found ({out_file_2d})")
                        continue

                    if not mag_file or enc_num is None or not csv_path or not os.path.exists(csv_path):
                        if args.verbose: print(f"  [!] Skipping {chunk_str}: Required mag/CSV plasma source missing")
                        continue

                    update_ui("Extracting 2D CDF", chunk_str)
                    d_out, physics_out_2d = resonance_io.read_2d_waveResonance_cdf(out_file_2d)
                    
                    # Fetch background plasma data onto the 2D Epoch time grid
                    update_ui("Fetching Plasma Data", chunk_str)
                    d_plasma = resonance_io.fetch_plasma_data(
                        d_out['Time'], mag_file, csv_path, start_dt, end_dt, LFR_DENSITY_FILE
                    )
                    d_out.update(d_plasma)
                        
                    # Calculate unmasked V_A continuously across the whole time grid
                    B0_mag = np.linalg.norm(d_out['B0'], axis=-1)
                    V_A = kinematics.calculate_alfven_velocity(B0_mag, d_out['N_p'], d_out['N_alpha'])
                    
                    # Compute 1D power-weighted averages from extracted 2D data
                    avgs_1d = {}
                    for prefix, mask in [('LH', physics_out_2d['mask_LH']), ('RH', physics_out_2d['mask_RH'])]:
                        power_weights = d_out.get('Power_B', np.ones_like(physics_out_2d['V_ph']))
                        avgs_1d[f'avg_k_parallel_{prefix}']     = get_masked_weighted_avg(physics_out_2d['k_parallel'], mask, power_weights)
                        avgs_1d[f'avg_k_perp_{prefix}']         = get_masked_weighted_avg(physics_out_2d['k_perp'], mask, power_weights)
                        avgs_1d[f'avg_V_ph_{prefix}']           = get_masked_weighted_avg(physics_out_2d['V_ph'], mask, power_weights)
                        avgs_1d[f'avg_f_plasma_{prefix}']       = get_masked_weighted_avg(physics_out_2d['f_plasma'], mask, power_weights)
                        avgs_1d[f'avg_E_res_p_{prefix}']        = get_masked_weighted_avg(physics_out_2d['E_res_p'], mask, power_weights)
                        avgs_1d[f'avg_E_res_alpha_{prefix}']    = get_masked_weighted_avg(physics_out_2d['E_res_alpha'], mask, power_weights)
                        avgs_1d[f'avg_E_res_landau_p_{prefix}'] = get_masked_weighted_avg(physics_out_2d['E_res_landau_p'], mask, power_weights)
                        avgs_1d[f'avg_E_res_anom_p_{prefix}']   = get_masked_weighted_avg(physics_out_2d['E_res_anom_p'], mask, power_weights)
                        avgs_1d[f'avg_E_res_anom_alpha_{prefix}']= get_masked_weighted_avg(physics_out_2d['E_res_anom_alpha'], mask, power_weights)

                    # Accumulate into daily 1D arrays
                    daily_d_out['Time'].append(d_out['Time'])
                    daily_d_out['B0'].append(d_out['B0'])
                    daily_d_out['V_sw'].append(d_out['V_sw'])
                    daily_d_out['N_p'].append(d_out['N_p'])
                    daily_d_out['N_alpha'].append(d_out['N_alpha'])
                    daily_d_out['Density_Flag'].append(d_out['Density_Flag'])
                    
                    daily_physics_out['V_A'].append(V_A)  # Continuous, unmasked V_A
                    daily_physics_out['int_LH'].append(d_out.get('int_LH', np.full(len(d_out['Time']), np.nan)))
                    daily_physics_out['int_RH'].append(d_out.get('int_RH', np.full(len(d_out['Time']), np.nan)))
                    daily_physics_out['int_Sn_pos'].append(d_out.get('int_Sn_pos', np.full(len(d_out['Time']), np.nan)))
                    daily_physics_out['int_Sn_neg'].append(d_out.get('int_Sn_neg', np.full(len(d_out['Time']), np.nan)))
                    
                    for k in avg_keys:
                        daily_physics_out[k].append(avgs_1d[k])
                    
                    chunks_processed += 1
                    continue
                
                #mag_file_pattern = os.path.join(MAG_4SA_ROOT, y, m, f"psp_fld_l2_mag_SC_4_Sa_per_Cyc_{y}{m}{d}_v*.cdf")
                #matching_mag_files = sorted(glob.glob(mag_file_pattern))
                
                #if not matching_mag_files:
                #    if args.verbose: print(f"  [!] Skipping {chunk_str}: Mag file missing")
                #    continue
                    
                #mag_file = matching_mag_files[-1] 
                
                enc_num = get_encounter_from_date(start_dt)
                if enc_num is None:
                    continue
                    
                csv_path = os.path.join(CSV_ROOT, f"E{enc_num:02d}.csv")
                if not os.path.exists(csv_path) or not os.path.exists(wave_file):
                    continue

                v_out_dir_2d = os.path.join(ANALYSIS_OUT_ROOT_2D, y, m)
                out_file_2d = os.path.join(v_out_dir_2d, f"PSP_WaveResonance_2D_{y}-{m}-{d}_{hr_str}00_v{v_2d}.cdf")

                # 1. Fetch CDF/CSV Data
                update_ui("Fetching Data", chunk_str)
                d_out = resonance_io.fetch_and_align_data(
                    wave_file, mag_file, csv_path, start_dt, end_dt, LFR_DENSITY_FILE
                )
                
                power_e_clean = np.where(d_out['Power_E'] > 0, d_out['Power_E'], np.nan)
                if np.all(np.isnan(power_e_clean)):
                    if args.verbose: print(f"  [!] Skipping {chunk_str}: Power_E is 100% invalid")
                    continue
                
                # 2. Fetch SPICE Data
                cmat, V_sc_RTN = spice_manager.get_spice_data(d_out['Time'])
                
                # 3. Process Wave Vector and Doppler Corrections
                update_ui("Physics & Resonances", chunk_str)
                L_baseline = d_out.get('efieldAntenna_baselineLength', 3.5)
                L_eff_dynamic = np.full_like(d_out['f_sc'], L_baseline)
                
                k_mag, V_ph = kinematics.get_k_magnitude_from_powers(
                    d_out['f_sc'], d_out['Power_E'], d_out['Power_B'], L_baseline, L_eff_dynamic
                )
                f_plasma, k_parallel, k_perp = kinematics.calculate_plasma_frame_physics(
                    d_out['f_sc'], d_out['k_n'], d_out['k_p'], d_out['k_q'], k_mag, 
                    d_out['V_sw'], V_sc_RTN, d_out['B0'], cmat
                )
                
                # 4. Process Resonant Energies
                B0_mag = np.linalg.norm(d_out['B0'], axis=-1)
                E_res_p, E_res_alpha, E_res_landau_p, E_res_anom_p, E_res_anom_alpha = resonances.calculate_resonant_energies(
                    f_plasma, k_parallel, B0_mag
                )
                
                # 5. Process Auxiliary Metrics
                with np.errstate(divide='ignore', invalid='ignore'):
                    compressibility = d_out['Power_B_para'] / (d_out['Power_B_para'] + d_out['Power_B'])
                V_A = kinematics.calculate_alfven_velocity(B0_mag, d_out['N_p'], d_out['N_alpha'])
                
                # 6. Process Integrated Parameters & Masks
                int_LH, int_RH, int_Sn_pos, int_Sn_neg, mask_LH, mask_RH, mask_Sn_pos, mask_Sn_neg = kinematics.calculate_integrated_powers(
                    d_out['f_sc'], d_out['Power_B'], d_out['S_n'], 
                    d_out['coherency_B'], d_out['ellipticity_B'], k_parallel, k_mag
                )
                
                # 7. Apply Coherency Mask
                coh_smoothed = smooth_mask(d_out['coherency_B'])
                valid_wave_mask = coh_smoothed >= 0.5
                
                E_res_p          = np.where(valid_wave_mask, E_res_p, np.nan)
                E_res_alpha      = np.where(valid_wave_mask, E_res_alpha, np.nan)
                E_res_landau_p   = np.where(valid_wave_mask, E_res_landau_p, np.nan)
                E_res_anom_p     = np.where(valid_wave_mask, E_res_anom_p, np.nan)
                E_res_anom_alpha = np.where(valid_wave_mask, E_res_anom_alpha, np.nan)
                
                # 7.5 Calculate 1D Power-Weighted Averages
                avgs_1d = {}
                for prefix, mask in [('LH', mask_LH), ('RH', mask_RH)]:
                    avgs_1d[f'avg_k_parallel_{prefix}']     = get_masked_weighted_avg(k_parallel, mask, d_out['Power_B'])
                    avgs_1d[f'avg_k_perp_{prefix}']         = get_masked_weighted_avg(k_perp, mask, d_out['Power_B'])
                    avgs_1d[f'avg_V_ph_{prefix}']           = get_masked_weighted_avg(V_ph, mask, d_out['Power_B'])
                    avgs_1d[f'avg_f_plasma_{prefix}']       = get_masked_weighted_avg(f_plasma, mask, d_out['Power_B'])
                    avgs_1d[f'avg_E_res_p_{prefix}']        = get_masked_weighted_avg(E_res_p, mask, d_out['Power_B'])
                    avgs_1d[f'avg_E_res_alpha_{prefix}']    = get_masked_weighted_avg(E_res_alpha, mask, d_out['Power_B'])
                    avgs_1d[f'avg_E_res_landau_p_{prefix}'] = get_masked_weighted_avg(E_res_landau_p, mask, d_out['Power_B'])
                    avgs_1d[f'avg_E_res_anom_p_{prefix}']   = get_masked_weighted_avg(E_res_anom_p, mask, d_out['Power_B'])
                    avgs_1d[f'avg_E_res_anom_alpha_{prefix}']= get_masked_weighted_avg(E_res_anom_alpha, mask, d_out['Power_B'])

                # 8. Output 2D Spectrogram to CDF
                if not args.overwrite and os.path.exists(out_file_2d):
                    if args.verbose: print(f"  [*] 2D File exists. Skipping write: {os.path.basename(out_file_2d)}")
                else:
                    update_ui("Writing 2D CDF", chunk_str)
                    physics_out_2d = {
                        'f_plasma': f_plasma, 'k_parallel': k_parallel, 'k_perp': k_perp, 'V_ph': V_ph,
                        'compressibility': compressibility,
                        'E_res_p': E_res_p, 'E_res_alpha': E_res_alpha, 'E_res_landau_p': E_res_landau_p, 
                        'E_res_anom_p': E_res_anom_p, 'E_res_anom_alpha': E_res_anom_alpha,
                        'mask_LH': mask_LH.astype(np.int8), 'mask_RH': mask_RH.astype(np.int8), 
                        'mask_Sn_pos': mask_Sn_pos.astype(np.int8), 'mask_Sn_neg': mask_Sn_neg.astype(np.int8),
                        'L_eff_dynamic': L_eff_dynamic
                    }
                    resonance_io.write_2d_waveResonance_cdf(out_file_2d, d_out, physics_out_2d, v_2d, os.path.basename(wave_file), os.path.basename(mag_file))
                
                # 9. Accumulate 1D data into Daily Lists
                daily_d_out['Time'].append(d_out['Time'])
                daily_d_out['B0'].append(d_out['B0'])
                daily_d_out['V_sw'].append(d_out['V_sw'])
                daily_d_out['N_p'].append(d_out['N_p'])
                daily_d_out['N_alpha'].append(d_out['N_alpha'])
                daily_d_out['Density_Flag'].append(d_out['Density_Flag'])
                
                daily_physics_out['V_A'].append(V_A)
                daily_physics_out['int_LH'].append(int_LH)
                daily_physics_out['int_RH'].append(int_RH)
                daily_physics_out['int_Sn_pos'].append(int_Sn_pos)
                daily_physics_out['int_Sn_neg'].append(int_Sn_neg)
                
                for k in avg_keys:
                    daily_physics_out[k].append(avgs_1d[k])
                
                chunks_processed += 1

            # Write daily 1D merged file
            if chunks_processed > 0:
                update_ui("Writing 1D CDF", "Daily Merge")
                if args.verbose:
                    print(f"  -> Writing Merged Daily 1D File for {date_str}...")
                
                merged_d_out = {k: np.concatenate(v, axis=0) for k, v in daily_d_out.items()}
                merged_physics_out = {k: np.concatenate(v, axis=0) for k, v in daily_physics_out.items()}
                
                resonance_io.write_1d_waveResonance_cdf(out_file_1d, merged_d_out, merged_physics_out, v_3d, "Multiple (Daily Merged)", "Multiple (Daily Merged)")
                update_ui("Complete", "Done")
            else:
                update_ui("No Valid Data", "Done")

            return True

    except Exception as e:
        update_ui(f"Failed: {e}", "-")
        log_queue.put(f"  [!] Fatal Failure on {date_str}: {e}")
        return False
        
    finally:
        if not args.verbose:
            out_stream.close()
        gc.collect()
        slot_queue.put(slot_id) # Release worker slot back to pool

def main():
    args = setup()
    
    try:
        date_parts = args.date.split(' through ')
        start_date = datetime.strptime(date_parts[0].strip(), '%Y-%m-%d')
        end_date   = datetime.strptime(date_parts[1].strip(), '%Y-%m-%d')
    except Exception as e:
        print(f"[!] Error parsing --date string: {e}")
        return

    print(f"[*] Commencing Multiprocess Batch Loop: {start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}")
    
    # --- PRE-FLIGHT SPICE DOWNLOAD ---
    print("[*] Checking & downloading SPICE kernels for the entire batch window...")
    
    spice_days = []
    curr_dt = start_date
    while curr_dt <= end_date:
        spice_days.append(curr_dt)
        curr_dt += timedelta(days=1)

    failed_spice_dates = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold magenta]{task.description}"),
        BarColumn(bar_width=40, complete_style="magenta", finished_style="green"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        transient=True  # Clears progress bar when done
    ) as progress:
        
        spice_task = progress.add_task("Syncing SPICE...", total=len(spice_days))
        
        for day_dt in spice_days:
            date_str = day_dt.strftime('%Y-%m-%d')
            progress.update(spice_task, description=f"Syncing SPICE: {date_str}")
            
            try:
                # Force SPICE manager to verify/download kernels for this daily timestamp
                spice_manager.get_spice_data(np.array([day_dt.timestamp()]))
            except Exception as e:
                failed_spice_dates.append((date_str, str(e)))
                # Prints cleanly above the progress bar without disrupting it
                progress.console.print(f"[bold red][!] SPICE pre-fetch failed for {date_str}: {e}[/bold red]")
                
            progress.advance(spice_task)

    if failed_spice_dates:
        print(f"[!] Warning: {len(failed_spice_dates)} date(s) had SPICE sync issues (workers will retry on the fly).")
    else:
        print("[+] All SPICE kernels verified and cached locally.\n")

    # Build master list of daily tasks
    day_tasks = []
    curr = start_date
    while curr <= end_date:
        day_tasks.append((curr, args))
        curr += timedelta(days=1)

    total_days = len(day_tasks)
    completed_days = 0

    # Multiprocessing Queues and Worker Slots
    manager = Manager()
    slot_queue = manager.Queue()
    status_queue = manager.Queue()
    log_queue = manager.Queue()
    
    for i in range(args.workers):
        slot_queue.put(i)
        
    worker_states = {i: {'date': 'Idle', 'chunk': '-', 'status': 'Idle', 'ram': '-'} for i in range(args.workers)}

    # Run Process Pool + Live Dashboard Event Loop
    try:
        with Live(make_dashboard(worker_states, 0, total_days), refresh_per_second=4) as live:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                
                futures = {
                    executor.submit(process_day, task, slot_queue, status_queue, log_queue): task 
                    for task in day_tasks
                }
                
                completed_futures = set()
                last_completed = -1

                while completed_days < total_days:
                    # 1. Print worker logs safely above the dashboard table
                    while not log_queue.empty():
                        msg = log_queue.get()
                        live.console.print(msg)

                    # 2. Update dashboard table state
                    while not status_queue.empty():
                        msg = status_queue.get()
                        slot = msg['slot_id']
                        worker_states[slot].update(msg)

                    # 3. Harvest completed day tasks
                    done = [f for f in futures if f.done() and f not in completed_futures]
                    for f in done:
                        completed_futures.add(f)
                        completed_days += 1
                    
                    # --- TABBY LIVE TAB TITLE UPDATE (DIRECT TTY) ---
                    if completed_days != last_completed:
                        last_completed = completed_days
                        pct = int((completed_days / total_days) * 100)
                        set_terminal_tab_title(f"[{pct}%] WaveResonance ({completed_days}/{total_days} Days)")

                    live.update(make_dashboard(worker_states, completed_days, total_days))
                    time.sleep(0.1)

    except KeyboardInterrupt:
        set_terminal_tab_title("[ABORTED] WaveResonance")
        print("\n\n[!] KeyboardInterrupt detected! Force-killing worker pool and releasing memory...")
        executor.shutdown(wait=False, cancel_futures=True)
        manager.shutdown()
        sys.exit(1)

    # --- TABBY COMPLETION NOTIFICATION & TERMINAL BELL ---
    try:
        if os.name == 'posix':
            with open('/dev/tty', 'w') as tty:
                tty.write("\033]0;[DONE] WaveResonance Complete!\007") # Set Tab Title
                tty.write("\033]9;Parker Solar Probe WaveResonance processing complete!\007") # Tabby Notification
                tty.flush()
    except Exception:
        pass

    print("\n[+] Multiprocessing waveResonance execution complete.")

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True, help="Format: 'YYYY-MM-DD through YYYY-MM-DD'")
    parser.add_argument('-v_in', type=str, default="1.4", help="Input WaveAnalysis version")
    parser.add_argument('--workers', type=int, default=2, help="Number of days to process concurrently")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed step-by-step progress.")
    parser.add_argument('--overwrite', '-o', action='store_true', help="Overwrite existing CDF files")
    parser.add_argument('--from_2d', action='store_true', help="Reprocess 1D v3.x files directly from 2D v2.x CDFs without re-running heavy 2D calculations")
    return parser.parse_args()

if __name__ == "__main__":
    main()
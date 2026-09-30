import argparse
import os
import glob
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import cdflib
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import config


# =========================================================================
# EFFECTIVE LENGTH MODEL (Physics Bandpass + Damping)
# =========================================================================

def get_leff_bandpass_damping(f_Hz, V_sc_volts, delta_B_nT):
    """Physics Bandpass + Non-Linear Wave Damping."""
    L_base, A, V0, f0, B0, alpha, beta, gamma = 1.68, 22.6, 5.25, 0.43, 4.68, 0.87, 0.01, 0.88
    v_norm = np.maximum(V_sc_volts / np.maximum(V0, 1e-3), 1e-5)
    f_norm = np.maximum(f_Hz / np.maximum(f0, 1e-3), 1e-5)
    b_norm = np.maximum(delta_B_nT / np.maximum(B0, 1e-3), 0.0)
    v_term = (v_norm * np.exp(1.0 - v_norm)) ** alpha
    f_term = (f_norm * np.exp(1.0 - f_norm)) ** beta
    damping = 1.0 / (1.0 + (b_norm ** gamma))
    return L_base + A * v_term * f_term * damping


# =========================================================================
# INDEPENDENT PLASMA DATA LOADER (4 Sa/s MAG + SWEAP/LFR -> v_A)
# =========================================================================

def load_independent_plasma_data(start_dt, end_dt):
    """
    Reads 4 Sa/cycle L2 MAG CDFs and SWEAP/LFR density sources, downsampling B0
    to 1 Hz, and applies a rolling median filter to scrub fit glitches.
    """
    SCI_ROOT = config.get_berkeleyCacheData()
    PSP_SHAREDDRIVE_ROOT = config.get_sharedDrivePSP_path()
    MAG_4SA_ROOT = os.path.join(SCI_ROOT, 'fields', 'l2', 'mag_SC_4_Sa_per_Cyc')
    CSV_ROOT = os.path.join(PSP_SHAREDDRIVE_ROOT, '15min_DATA_Internal', '_CURRENT_20260604')
    LFR_DENSITY_FILE = os.path.join(config.get_drive_path(), 'Research', 'PSP', 'FIELDS', 'LFR_Density', 'spp_fld_lfr_mission_density.cdf')

    print("    [*] Loading 4 Sa/cycle L2 MAG B0 data...")
    b0_records = []
    curr = start_dt
    while curr <= end_dt:
        y, m, d = curr.strftime('%Y'), curr.strftime('%m'), curr.strftime('%d')
        mag_pattern = os.path.join(MAG_4SA_ROOT, y, m, f"psp_fld_l2_mag_SC_4_Sa_per_Cyc_{y}{m}{d}_v*.cdf")
        files = sorted(glob.glob(mag_pattern))
        if files:
            cdf_path = files[-1]
            try:
                with cdflib.CDF(cdf_path) as cdf_m:
                    zvars = cdf_m.cdf_info().zVariables
                    t_var = 'epoch_mag_SC_4_Sa_per_Cyc' if 'epoch_mag_SC_4_Sa_per_Cyc' in zvars else 'epoch_mag_SC_1min'
                    b_var = 'psp_fld_l2_mag_SC_4_Sa_per_Cyc' if 'psp_fld_l2_mag_SC_4_Sa_per_Cyc' in zvars else 'psp_fld_l2_mag_SC_1min'
                    mag_times = cdflib.cdfepoch.unixtime(cdf_m.varget(t_var))
                    b0_raw = cdf_m.varget(b_var)
                    b0_raw = np.where(b0_raw < -1e20, np.nan, b0_raw)
                    b0_mag = np.linalg.norm(b0_raw, axis=-1)
                    
                    df_day_b = pd.DataFrame({'unix': mag_times, 'b0_mag': b0_mag})
                    df_day_b['datetime'] = pd.to_datetime(df_day_b['unix'], unit='s')
                    df_1hz = df_day_b.set_index('datetime').resample('1s').mean().reset_index()
                    df_1hz['unix'] = df_1hz['datetime'].astype('int64') / 1e9
                    b0_records.append(df_1hz[['unix', 'datetime', 'b0_mag']])
            except Exception as e:
                print(f"      [!] Error reading MAG file {os.path.basename(cdf_path)}: {e}")
        curr += timedelta(days=1)

    if not b0_records:
        print("    [!] Error: No MAG L2 files found in date range.")
        return None

    df_plasma = pd.concat(b0_records, ignore_index=True).sort_values('unix').reset_index(drop=True)

    print("    [*] Loading SWEAP CSV proton density...")
    csv_records = []
    curr = start_dt
    while curr <= end_dt:
        enc_num = None
        for enc, (s_str, e_str) in config.ORBIT_DATES.items():
            s_d = datetime.strptime(s_str, '%Y-%m-%d') - timedelta(days=1)
            e_d = datetime.strptime(e_str, '%Y-%m-%d') + timedelta(days=2)
            if s_d <= curr <= e_d:
                enc_num = enc
                break
        if enc_num is not None:
            csv_path = os.path.join(CSV_ROOT, f"E{enc_num:02d}.csv")
            if os.path.exists(csv_path):
                try:
                    df_csv = pd.read_csv(csv_path, index_col=0)
                    df_csv.columns = df_csv.columns.str.strip()
                    if 'Times' in df_csv.columns:
                        df_csv['Times'] = pd.to_datetime(df_csv['Times'], utc=True, errors='coerce').dt.tz_localize(None)
                        c_mask = (df_csv['Times'] >= start_dt - timedelta(hours=12)) & (df_csv['Times'] <= end_dt + timedelta(hours=12))
                        df_win = df_csv.loc[c_mask]
                        
                        np_col = None
                        for col in ['Np_Parker', 'Np', 'n_p', 'N_p', 'density', 'Np_spc', 'Np_spani']:
                            if col in df_win.columns and df_win[col].notna().any():
                                np_col = col
                                break
                        if np_col:
                            np_vals = pd.to_numeric(df_win[np_col], errors='coerce').values
                            np_vals = np.where((np_vals > 0) & (np_vals < 1e5), np_vals, np.nan)
                            unix_csv = (df_win['Times'] - pd.Timestamp("1970-01-01")).dt.total_seconds().values
                            csv_records.append(pd.DataFrame({'unix': unix_csv, 'n_p': np_vals}))
                except Exception as e:
                    print(f"      [!] Error reading SWEAP CSV {os.path.basename(csv_path)}: {e}")
        curr += timedelta(days=1)

    # Linearly interpolate n_p onto 1 Hz plasma grid
    if csv_records:
        df_np_raw = pd.concat(csv_records, ignore_index=True).sort_values('unix').reset_index(drop=True)
        df_np_clean = df_np_raw.dropna(subset=['n_p']).drop_duplicates(subset=['unix'])
        if len(df_np_clean) > 2:
            f_np = interp1d(df_np_clean['unix'], df_np_clean['n_p'], bounds_error=False, fill_value='extrapolate')
            n_p_interp = f_np(df_plasma['unix'])
            df_plasma['n_p'] = np.where((n_p_interp > 0) & (n_p_interp < 1e5), n_p_interp, np.nan)
        else:
            df_plasma['n_p'] = np.nan
    else:
        df_plasma['n_p'] = np.nan

    if os.path.exists(LFR_DENSITY_FILE) and df_plasma['n_p'].isna().any():
        print("    [*] Patching missing SWEAP density gaps with LFR Mission Density...")
        try:
            with cdflib.CDF(LFR_DENSITY_FILE) as cdf_lfr:
                lfr_unix = cdflib.cdfepoch.unixtime(cdf_lfr.varget('Epoch'))
                lfr_ne = cdf_lfr.varget('electronDensity')
                valid_lfr = np.isfinite(lfr_ne) & (lfr_ne > 0) & (lfr_ne < 1e5)
                if np.any(valid_lfr):
                    f_lfr = interp1d(lfr_unix[valid_lfr], lfr_ne[valid_lfr] / 1.08, bounds_error=False, fill_value=np.nan)
                    lfr_interp = f_lfr(df_plasma['unix'])
                    df_plasma['n_p'] = df_plasma['n_p'].fillna(pd.Series(lfr_interp))
        except Exception as e:
            print(f"      [!] LFR density patch notice: {e}")

    b0_val = df_plasma['b0_mag'].values
    np_val = df_plasma['n_p'].values
    v_A_raw = np.where((np_val > 0) & (b0_val > 0), 21.812 * b0_val / np.sqrt(np_val * 1.16), np.nan)
    
    # --- MEDIAN FILTERING & OUTLIER SCRUBBING ---
    v_A_series = pd.Series(v_A_raw)
    v_A_med = v_A_series.rolling(window=181, center=True, min_periods=1).median()
    
    dev_ratio = v_A_raw / v_A_med
    v_A_clean = np.where((dev_ratio > 0.65) & (dev_ratio < 1.5), v_A_raw, v_A_med)
    
    df_plasma['v_alfven'] = v_A_clean
    return df_plasma


# =========================================================================
# WAVE DATA LOADER (1D WaveResonance CDFs)
# =========================================================================

def load_wave_data(cdf_path):
    """Loads LH phase velocity, wave normal angle, and power from 1D WaveResonance CDF."""
    if not os.path.exists(cdf_path):
        return None

    try:
        cdf = cdflib.CDF(cdf_path)
        unix_times = cdflib.cdfepoch.unixtime(cdf.varget('Epoch'))
        dtimes = pd.to_datetime(unix_times, unit='s')
        zvars = cdf.cdf_info().zVariables

        v_ph_raw = cdf.varget('phase_velocity_avg')[:, 0]
        v_ph_lh = np.where((v_ph_raw > 0) & (v_ph_raw < 1e6) & np.isfinite(v_ph_raw), v_ph_raw, np.nan)

        power_raw = cdf.varget('integrated_wave_power')[:, 0]
        power_lh = np.where((power_raw >= 0) & (power_raw < 1e20) & np.isfinite(power_raw), power_raw, 0.0)

        f_plasma_raw = cdf.varget('plasma_frequency_avg')[:, 0]
        f_lh = np.abs(np.where((f_plasma_raw > -1e20) & np.isfinite(f_plasma_raw), f_plasma_raw, 1.0))
        f_lh = np.where(f_lh == 0, 1.0, f_lh)

        k_avg_lh = cdf.varget('wavenumbers_avg_LH') if 'wavenumbers_avg_LH' in zvars else np.full((len(unix_times), 2), np.nan)
        k_avg_lh = np.where(k_avg_lh < -1e20, np.nan, k_avg_lh)

        k_para = k_avg_lh[:, 0]
        k_perp = k_avg_lh[:, 1]

        valid_k = np.isfinite(k_para) & np.isfinite(k_perp)
        theta_kB_rad = np.where(valid_k, np.arctan2(np.abs(k_perp), np.abs(k_para)), np.nan)
        cos_theta = np.cos(theta_kB_rad)

        delta_B_nT = np.sqrt(np.maximum(power_lh, 0.0))
        v_sc_eval = np.full_like(f_lh, 3.5)
        L_phys = 3.5
        L_eff_bandpass = get_leff_bandpass_damping(f_lh, v_sc_eval, delta_B_nT)
        v_ph_corr = v_ph_lh * (L_phys / L_eff_bandpass)

        df_waves = pd.DataFrame({
            'unix': unix_times,
            'datetime': dtimes,
            'v_ph_base': v_ph_lh,
            'v_ph_corr': v_ph_corr,
            'power_lh': power_lh,
            'cos_theta': cos_theta
        })

        return df_waves[np.isfinite(v_ph_lh)].copy()

    except Exception as e:
        print(f"    [!] Error reading wave CDF {os.path.basename(cdf_path)}: {e}")
        return None


# =========================================================================
# PLOTTING ROUTINE WITH UPDATED LABELS
# =========================================================================

def generate_icw_phase_speed_plot(df_plasma, df_waves, start_dt, end_dt, out_dir):
    """Generates two-panel ICW phase speed shift plot with Leff Fit Correction labels."""
    if df_waves.empty or df_plasma.empty:
        print("    [!] Missing wave or plasma data for requested window.")
        return

    m_plasma = (df_plasma['datetime'] >= start_dt) & (df_plasma['datetime'] <= end_dt)
    m_waves = (df_waves['datetime'] >= start_dt) & (df_waves['datetime'] <= end_dt)

    df_p = df_plasma[m_plasma].sort_values('datetime').reset_index(drop=True)
    df_w = df_waves[m_waves].sort_values('datetime').reset_index(drop=True)

    if df_w.empty:
        print(f"    [!] No valid LH wave points in window: {start_dt} to {end_dt}")
        return

    # Interpolate continuous plasma v_A onto wave timestamps
    f_va = interp1d(df_p['unix'], df_p['v_alfven'], bounds_error=False, fill_value=np.nan)
    df_w['v_alfven'] = f_va(df_w['unix'])
    
    # Projected v_A cos(theta_kB) at wave points
    df_w['v_alfven_proj'] = np.where((df_w['cos_theta'] > 0.1736), df_w['v_alfven'] * df_w['cos_theta'], np.nan)

    # Continuous background line in Panel 1
    f_cos = interp1d(df_w['unix'], np.nan_to_num(df_w['cos_theta'], nan=1.0), bounds_error=False, fill_value=1.0)
    df_p['cos_theta_interp'] = f_cos(df_p['unix'])
    
    v_proj_raw = df_p['v_alfven'] * df_p['cos_theta_interp']
    v_proj_series = pd.Series(v_proj_raw)
    
    # 3-minute rolling median filter on background line
    df_p['v_alfven_proj'] = v_proj_series.rolling(window=181, center=True, min_periods=1).median()

    # Insert gap NaNs in plasma line when time delta > 120 sec
    time_diffs = df_p['datetime'].diff().dt.total_seconds()
    gap_indices = df_p.index[time_diffs > 120]
    dummy_rows = []
    for idx in gap_indices:
        dummy_rows.append({'datetime': df_p.loc[idx - 1, 'datetime'] + timedelta(seconds=1), 'v_alfven_proj': np.nan})
    if dummy_rows:
        df_p_masked = pd.concat([df_p, pd.DataFrame(dummy_rows)], ignore_index=True).sort_values('datetime').reset_index(drop=True)
    else:
        df_p_masked = df_p.copy()

    # Dynamic transparency based on sample density
    n_pts = len(df_w)
    if n_pts > 50000:
        alpha_base, alpha_corr = 0.04, 0.04
        ms_panel1, ms_panel2 = 1.0, 1.2
    elif n_pts > 10000:
        alpha_base, alpha_corr = 0.08, 0.08
        ms_panel1, ms_panel2 = 1.5, 2.0
    else:
        alpha_base, alpha_corr = 0.20, 0.20
        ms_panel1, ms_panel2 = 2.5, 3.5

    t_min_str = df_w['datetime'].min().strftime('%Y-%m-%d %H:%M')
    t_max_str = df_w['datetime'].max().strftime('%Y-%m-%d %H:%M')
    time_title = f"{t_min_str} to {t_max_str} UTC"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1.4]})
    fig.suptitle(f"LH ICW Phase Speed Shift ($V_\\mathrm{{ph}}$ vs. $v_A \\cos\\theta_{{kB}}$)\n[{time_title}]", 
                 fontweight='bold', fontsize=18, y=0.98)

    # --- PANEL 1: Timeseries ---
    ax1.plot(df_w['datetime'], df_w['v_ph_base'], '.', color='#ff7f0e', alpha=alpha_base, markersize=ms_panel1, zorder=2, label=r'Base $V_{\mathrm{ph}}$ ($L_{\mathrm{phys}}=3.5\mathrm{m}$)')
    ax1.plot(df_w['datetime'], df_w['v_ph_corr'], '.', color='#1f77b4', alpha=alpha_corr, markersize=ms_panel1, zorder=3, label=r'$L_{\mathrm{eff}}$ Fit Corrected $V_{\mathrm{ph}}$')
    ax1.plot(df_p_masked['datetime'], df_p_masked['v_alfven_proj'], '-', color='black', linewidth=1.8, zorder=5, label=r'Projected Alfvén Speed $v_A \cos\theta_{kB}$')

    ax1.set_yscale('log')
    ax1.set_ylabel("Phase Velocity (km/s)", fontweight='bold', fontsize=14)
    ax1.tick_params(axis='both', which='both', labelsize=12)
    
    # Tight Y-axis limits for Panel 1
    v_alf_valid = df_p_masked['v_alfven_proj'].dropna()
    y_min1 = max(10.0, v_alf_valid.min() * 0.7) if not v_alf_valid.empty else 10.0
    y_max1 = min(1e4, max(2000.0, np.percentile(df_w['v_ph_corr'].dropna(), 99.8) * 4.0))
    ax1.set_ylim(bottom=y_min1, top=y_max1)
    
    ax1.grid(True, which='both', alpha=0.3, linestyle=':')
    
    leg1 = ax1.legend(loc='upper right', fontsize=11, ncol=3, framealpha=0.9)
    for handle in leg1.legend_handles:
        if hasattr(handle, 'set_alpha'):
            handle.set_alpha(1.0)
            handle.set_markersize(7)

    # Smart date formatting
    span_hours = (df_w['datetime'].max() - df_w['datetime'].min()).total_seconds() / 3600.0
    if span_hours <= 36:
        formatter = mdates.DateFormatter('%H:%M')
        locator = mdates.HourLocator(interval=max(1, int(span_hours / 6)))
    elif span_hours <= 120:
        formatter = mdates.DateFormatter('%b %d\n%H:%M')
        locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    else:
        formatter = mdates.DateFormatter('%b %d')
        locator = mdates.AutoDateLocator(minticks=6, maxticks=10)

    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)
    ax1.set_xlabel("Time (UTC)", fontweight='bold', fontsize=14)

    # --- PANEL 2: Parity Plot ---
    df_parity = df_w.dropna(subset=['v_ph_base', 'v_ph_corr', 'v_alfven_proj']).copy()
    df_parity = df_parity[df_parity['v_alfven_proj'] > 0]

    if not df_parity.empty:
        ax2.plot(df_parity['v_alfven_proj'], df_parity['v_ph_base'], '.', color='#ff7f0e', alpha=alpha_base, markersize=ms_panel2, zorder=2, label=r'Uncorrected ($L_{\mathrm{phys}}=3.5\mathrm{m}$)')
        ax2.plot(df_parity['v_alfven_proj'], df_parity['v_ph_corr'], '.', color='#1f77b4', alpha=alpha_corr, markersize=ms_panel2, zorder=3, label=r'$L_{\mathrm{eff}}$ Fit Correction')

        x_min = max(10.0, df_parity['v_alfven_proj'].min() * 0.8)
        x_max = min(2000.0, max(500.0, df_parity['v_alfven_proj'].max() * 1.5))
        
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlim(left=x_min, right=x_max)

        # UNTIED Y-AXIS LIMITS
        y_min2 = max(10.0, df_parity['v_ph_corr'].min() * 0.7)
        y_max2 = min(1e4, max(2000.0, np.percentile(df_parity['v_ph_corr'].dropna(), 99.8) * 4.0))
        ax2.set_ylim(bottom=y_min2, top=y_max2)

        ref = np.logspace(np.log10(x_min), np.log10(x_max), 100)
        ax2.plot(ref, ref, 'k--', linewidth=1.8, zorder=4, label=r'1:1 Ideal Dispersion ($V_{\mathrm{ph}} = v_A \cos\theta_{kB}$)')

    ax2.set_xlabel(r"Theoretical Projected Alfvén Speed $v_A \cos\theta_{kB}$ (km/s)", fontweight='bold', fontsize=14)
    ax2.set_ylabel(r"Measured Phase Velocity $V_{\mathrm{ph}}$ (km/s)", fontweight='bold', fontsize=14)
    ax2.tick_params(axis='both', which='both', labelsize=12)
    ax2.grid(True, which='both', alpha=0.3, linestyle=':')
    
    leg2 = ax2.legend(loc='upper left', fontsize=12, framealpha=0.9)
    for handle in leg2.legend_handles:
        if hasattr(handle, 'set_alpha'):
            handle.set_alpha(1.0)
            handle.set_markersize(8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    date_tag = start_dt.strftime('%Y%m%d') if start_dt.date() == end_dt.date() else f"{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}"
    out_png = os.path.join(out_dir, f"icw_phase_speed_standalone_bandpass_{date_tag}.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    [+] Saved standalone ICW phase speed plot to: {out_png}")


# =========================================================================
# MAIN CLI EXECUTION
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Standalone ICW Phase Speed Correction (Leff Fit)")
    parser.add_argument('--date', type=str, required=True, help="Format: 'YYYY-MM-DD' or 'YYYY-MM-DD through YYYY-MM-DD'")
    parser.add_argument('--out_dir', type=str, default=None, help="Directory to save output PNG")
    args = parser.parse_args()

    parts = args.date.split(' through ')
    start_dt = datetime.strptime(parts[0].strip(), '%Y-%m-%d')
    end_dt = datetime.strptime(parts[1].strip(), '%Y-%m-%d') + timedelta(days=1) - timedelta(seconds=1) if len(parts) > 1 else start_dt + timedelta(days=1) - timedelta(seconds=1)

    out_dir = args.out_dir if args.out_dir else os.getcwd()

    # 1. Independently fetch continuous B0 and n_p to calculate v_A
    df_plasma = load_independent_plasma_data(start_dt, end_dt)

    # 2. Fetch wave phase velocities from 1D WaveResonance CDFs
    search_dir = os.path.join(config.get_drive_path(), "Research", "PSP", "WaveAnalysis", "WaveAnalysis_Files", "v3.4")
    all_cdf_files = glob.glob(os.path.join(search_dir, "**", "PSP_WaveResonance_1D_*.cdf"), recursive=True)

    wave_dfs = []
    curr = start_dt
    while curr <= end_dt:
        date_pattern = curr.strftime('%Y-%m-%d')
        day_files = [f for f in all_cdf_files if date_pattern in os.path.basename(f)]
        for f in day_files:
            print(f"    [*] Loading Wave CDF: {os.path.basename(f)}")
            df_w = load_wave_data(f)
            if df_w is not None and not df_w.empty:
                wave_dfs.append(df_w)
        curr += timedelta(days=1)

    if wave_dfs and df_plasma is not None:
        df_waves_all = pd.concat(wave_dfs, ignore_index=True)
        generate_icw_phase_speed_plot(df_plasma, df_waves_all, start_dt, end_dt, out_dir)
    else:
        print(f"    [!] Failed to match wave CDFs or plasma data for date(s): {args.date}")
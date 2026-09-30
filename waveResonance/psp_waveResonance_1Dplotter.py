"""
psp_waveResonance_1Dplotter.py
Reads and stacks 1D WaveResonance CDF files, generating publication-quality panels.
"""
import os
import argparse
import gc
import glob
import numpy as np
import pandas as pd
import cdflib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# =============================================================================
# USER CONFIGURATION: PATHS & SETTINGS
# =============================================================================
BASE_DATA_DIR = r"G:\My Drive\Research\PSP\WaveAnalysis\WaveAnalysis_Files\v3.4"

# Publication-ready Matplotlib parameters (AGU/GRL standards)
plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1.5,
    'axes.labelsize': 12,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'xtick.major.width': 1.5,
    'ytick.major.size': 6,
    'ytick.major.width': 1.5,
    'lines.linewidth': 1.2,
    'figure.autolayout': True
})

# High-contrast, colorblind-safe palette
COLORS = {
    'LH': '#0072B2', 'RH': '#D55E00', 
    'X': '#21918c', 'Y': '#3b528b', 'Z': '#5ec962', 
    'R': '#C44E52', 'T': '#55A868', 'N': '#4C72B0', # Muted RGB for RTN
    'V_A': '#000000'
}

AVAILABLE_PANELS = [
    'B0', 'velocity', 'wave_power', 'density', 
    'proton_res_lh', 'proton_res_rh', 'alpha_res_lh', 'alpha_res_rh'
]

# =============================================================================
# DATA LOADING & FILTERING
# =============================================================================
def load_concatenated_1d_cdfs(start_date_str, end_date_str, base_dir=BASE_DATA_DIR):
    """
    Discovers and stacks 1D WaveResonance CDFs across a specified date range.
    Uses vectorized masking to eliminate slow for-loops over high-cadence telemetry.
    """
    start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    target_files = []
    current_dt = start_dt
    while current_dt <= end_dt:
        year_str = current_dt.strftime('%Y')
        month_str = current_dt.strftime('%m')
        date_str = current_dt.strftime('%Y-%m-%d')
        
        search_dir = os.path.join(base_dir, year_str, month_str)
        search_pattern = os.path.join(search_dir, f"*WaveResonance_1D_{date_str}*.cdf")
        
        target_files.extend(glob.glob(search_pattern))
        current_dt += timedelta(days=1)
        
    if not target_files:
        raise FileNotFoundError(f"No 1D CDFs found in {base_dir} (or subdirectories) for the requested dates.")

    target_files.sort()
    stacked_data = {}
    
    for file_path in target_files:
        with cdflib.CDF(file_path) as cdf:
            unix_times_s = cdflib.cdfepoch.unixtime(cdf.varget('Epoch'))
            
            if 'datetime_utc' not in stacked_data:
                stacked_data['datetime_utc'] = []
            stacked_data['datetime_utc'].extend([datetime.utcfromtimestamp(ts) for ts in unix_times_s])
            
            for var in cdf.cdf_info().zVariables:
                if 'label' in var or 'index' in var or var == 'Epoch':
                    continue
                    
                raw_array = cdf.varget(var)
                # Apply vectorized masking for standard CDF fill values (-1e31)[cite: 2]
                masked_array = np.where(raw_array < -1e30, np.nan, raw_array)
                
                if var not in stacked_data:
                    stacked_data[var] = []
                stacked_data[var].append(masked_array)
                
    # Finalize arrays to ensure contiguous memory chunks for rapid plotting
    stacked_data['datetime_utc'] = np.array(stacked_data['datetime_utc'])
    for key in stacked_data:
        if key != 'datetime_utc':
            stacked_data[key] = np.concatenate(stacked_data[key], axis=0)
            
    return stacked_data

def apply_smoothing(data_array, window_size):
    """
    Applies a rolling median filter to aggressively smooth high-frequency 
    variance in resonance energies while preserving physical step-boundaries.
    """
    if window_size <= 1:
        return data_array
    return pd.DataFrame(data_array).rolling(window=window_size, center=True, min_periods=1).median().values

# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_custom_stack(data_dict, panels, smooth_pts=1, output_filename=None):
    """
    Generates a stacked multipanel plot mapped dynamically from user requests.
    Actively manages memory via garbage collection at function exit.
    """
    num_panels = len(panels)
    fig, axes = plt.subplots(num_panels, 1, figsize=(10, 2.5 * num_panels), sharex=True)
    if num_panels == 1:
        axes = [axes]
        
    time_array_utc = data_dict['datetime_utc']
    
    for ax, panel in zip(axes, panels):
        if panel == 'B0':
            b_field_nt = data_dict['B0_SC']
            ax.plot(time_array_utc, b_field_nt[:, 0], color=COLORS['X'], label=r'$B_x$')
            ax.plot(time_array_utc, b_field_nt[:, 1], color=COLORS['Y'], label=r'$B_y$')
            ax.plot(time_array_utc, b_field_nt[:, 2], color=COLORS['Z'], label=r'$B_z$')
            ax.set_ylabel(r'$B_0$ ($nT$)')
            ax.legend(loc='upper right', frameon=False, fontsize=10, ncol=3)
            
        elif panel == 'velocity':
            v_sw_rtn_kms = data_dict['v_sw_RTN']
            v_alfven_kms = data_dict['v_alfven']
            ax.plot(time_array_utc, v_sw_rtn_kms[:, 0], color=COLORS['R'], label=r'$V_R$')
            ax.plot(time_array_utc, v_sw_rtn_kms[:, 1], color=COLORS['T'], label=r'$V_T$')
            ax.plot(time_array_utc, v_sw_rtn_kms[:, 2], color=COLORS['N'], label=r'$V_N$')
            ax.plot(time_array_utc, v_alfven_kms, color=COLORS['V_A'], linestyle='--', label=r'$V_A$')
            ax.set_ylabel(r'Velocity ($km/s$)')
            ax.legend(loc='upper right', frameon=False, fontsize=10, ncol=4)
            
        elif panel == 'wave_power':
            power_nt2 = data_dict['integrated_wave_power'].copy()
            # Explicitly mask non-positive values to prevent matplotlib from plunging to -inf on log scales
            power_nt2[power_nt2 <= 0] = np.nan
            
            # Scatter plot is preferred here to avoid false connecting lines across missing telemetry
            ax.plot(time_array_utc, power_nt2[:, 0], color=COLORS['LH'], marker='.', markersize=2, linestyle='none', label='LH Power')
            ax.plot(time_array_utc, power_nt2[:, 1], color=COLORS['RH'], marker='.', markersize=2, linestyle='none', label='RH Power')
            ax.set_yscale('log')
            ax.set_ylabel(r'Power ($nT^2$)')
            ax.legend(loc='upper right', frameon=False, fontsize=10)
            
        elif panel == 'proton_res_lh':
            res_lh_ev = data_dict['resonances_proton_avg_LH'].copy()
            res_lh_ev[res_lh_ev <= 0] = np.nan
            res_lh_ev = apply_smoothing(res_lh_ev, smooth_pts)
            
            ax.plot(time_array_utc, res_lh_ev[:, 0], color=COLORS['LH'], label='Cyclotron (n=1)')
            ax.plot(time_array_utc, res_lh_ev[:, 1], color=COLORS['LH'], linestyle='--', label='Landau (n=0)')
            ax.plot(time_array_utc, res_lh_ev[:, 2], color=COLORS['LH'], linestyle=':', label='Anomalous (n=-1)')
            ax.set_yscale('log')
            ax.set_ylabel(r'LH $E_{res,p}$ ($eV$)')
            ax.legend(loc='upper right', frameon=False, fontsize=10)

        elif panel == 'proton_res_rh':
            res_rh_ev = data_dict['resonances_proton_avg_RH'].copy()
            res_rh_ev[res_rh_ev <= 0] = np.nan
            res_rh_ev = apply_smoothing(res_rh_ev, smooth_pts)
            
            ax.plot(time_array_utc, res_rh_ev[:, 0], color=COLORS['RH'], label='Cyclotron (n=1)')
            ax.plot(time_array_utc, res_rh_ev[:, 1], color=COLORS['RH'], linestyle='--', label='Landau (n=0)')
            ax.plot(time_array_utc, res_rh_ev[:, 2], color=COLORS['RH'], linestyle=':', label='Anomalous (n=-1)')
            ax.set_yscale('log')
            ax.set_ylabel(r'RH $E_{res,p}$ ($eV$)')
            ax.legend(loc='upper right', frameon=False, fontsize=10)
            
        elif panel == 'density':
            ion_density_cm3 = data_dict['n_p']
            alpha_density_cm3 = data_dict['n_alpha']
            ax.plot(time_array_utc, ion_density_cm3, color=COLORS['X'], label='Protons')
            ax.plot(time_array_utc, alpha_density_cm3, color=COLORS['Y'], label='Alphas')
            ax.set_yscale('log')
            ax.set_ylabel(r'Density ($cm^{-3}$)')
            ax.legend(loc='upper right', frameon=False, fontsize=10)

        ax.grid(True, which='major', linestyle='-', alpha=0.3)
        ax.minorticks_on()
        
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)
    axes[-1].set_xlabel('Time (UTC)')
    
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"[*] Plot saved to {output_filename}")
    else:
        plt.show()
        
    # Free memory explicitly to prevent RAM leaks during multi-orbit processing
    fig.clf()
    plt.close(fig)
    gc.collect()

# =============================================================================
# CLI ROUTING
# =============================================================================
if __name__ == '__main__':
    DEFAULT_DASHBOARD = ['B0', 'velocity', 'wave_power', 'proton_res_lh', 'proton_res_rh']
    
    parser = argparse.ArgumentParser(
        description="Visualize 1D Wave Resonance data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--date', type=str, required=True, 
                        help="Start and end date formatted as 'YYYY-MM-DD through YYYY-MM-DD' (or a single 'YYYY-MM-DD').")
    parser.add_argument('--panels', nargs='+', type=str, default=DEFAULT_DASHBOARD, choices=AVAILABLE_PANELS,
                        help=f"List of panels to plot.")
    parser.add_argument('--smooth', type=int, default=11, 
                        help="Rolling median window size for resonance panels (default: 11).")
    parser.add_argument('--out', type=str, default=None, help="Optional filename to save the plot.")
    parser.add_argument('--base-dir', type=str, default=BASE_DATA_DIR, 
                        help="Override the default base directory for CDF files.")
    
    args = parser.parse_args()
    
    # Parse the new "through" string formatting
    if ' through ' in args.date:
        d_start, d_end = args.date.split(' through ')
    else:
        d_start, d_end = args.date, args.date
        
    print(f"[*] Extracting data from {d_start} to {d_end}...")
    loaded_data = load_concatenated_1d_cdfs(d_start, d_end, base_dir=args.base_dir) 
    
    print(f"[*] Plotting panels: {', '.join(args.panels)} (Smoothing: {args.smooth} pts)")
    plot_custom_stack(loaded_data, args.panels, smooth_pts=args.smooth, output_filename=args.out)
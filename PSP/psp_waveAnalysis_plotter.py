import socket
import os
import glob
import cdflib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm, Normalize
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import argparse
import tempfile
import shutil
import contextlib

CUSTOM_TMP = '/home/kpaulson/tmp'
os.makedirs(CUSTOM_TMP, exist_ok=True)

@contextlib.contextmanager
def open_local_cdf(remote_path):
    """Copies a CDF to local disk for fast reading, then deletes it."""
    fd, temp_path = tempfile.mkstemp(suffix='.cdf', dir=CUSTOM_TMP)
    os.close(fd)
    try:
        shutil.copy2(remote_path, temp_path)
        with cdflib.CDF(temp_path) as cdf:
            yield cdf
    finally:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass

# Set global font sizes
plt.rcParams.update({'font.size': 12})

############################################################################
# 1. UTILITY FUNCTIONS
############################################################################
def us2000_to_datetime(cdf_epochs):
    base = datetime(2000, 1, 1, 12, 0)
    return np.array([base + timedelta(microseconds=t/1000.0) for t in cdf_epochs])
    
def tt2000_to_datetime64(epochs):
    """Converts TT2000 epochs to a numpy datetime64 array, turning fill values into NaT."""
    # Convert TT2000 to Unix seconds (fill values automatically become NaN)
    unix_secs = cdflib.cdfepoch.unixtime(epochs)
    valid = ~np.isnan(unix_secs)
    
    # Initialize an array of 'NaT' (Not a Time)
    dt64_array = np.full(len(epochs), np.datetime64('NaT', 'ms'), dtype='datetime64[ms]')
    
    # Convert valid seconds to milliseconds and cast to datetime64
    dt64_array[valid] = (unix_secs[valid] * 1000).astype('datetime64[ms]')
    return dt64_array

def ephem_epoch_to_datetime(epochs):
    # Standard converter for mission-long position files
    return cdflib.epochs.CDFepoch.to_datetime(epochs)

def get_latest_version_file(directory, pattern):
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)
    if not files: return None
    files.sort()
    return files[-1]

def get_analysis_file_path(dt):
    hour_bin = (dt.hour // 6) * 6
    folder = os.path.join(ANALYSIS_ROOT, dt.strftime('%Y'), dt.strftime('%m'))
    filename = f"PSP_WaveAnalysis_{dt.strftime('%Y-%m-%d')}_{hour_bin:02d}00_{VERSION}.cdf"
    return os.path.join(folder, filename)

def load_data(start_dt, end_dt):
    br_times, br_out = None, None
    ephem_out = {}
    
    curr = start_dt - timedelta(hours=start_dt.hour % 6)
    analysis_files = []
    while curr < end_dt:
        analysis_files.append(get_analysis_file_path(curr))
        curr += timedelta(hours=6)
    
    a_data = {}
    freq_axis = None
    for f in analysis_files:
        if not os.path.exists(f): continue
        with open_local_cdf(f) as cdf:
            info = cdf.cdf_info()
            all_vars = info.zVariables + info.rVariables
            for var in all_vars:
                if var.startswith('label_') or var == 'comp_index':
                    continue
                val = cdf.varget(var)
                if var == 'frequencies':
                    freq_axis = val
                    continue
                
                # --- DYNAMIC METADATA MASKING ---
                # If the array contains floating-point values, check the metadata for valid ranges
                if np.issubdtype(val.dtype, np.floating):
                    # Ensure the array is float64 so it can safely hold np.nan values
                    val = val.astype(np.float64)
                    try:
                        atts = cdf.varattsget(var)
                        if atts:
                            if 'VALIDMIN' in atts:
                                vmin = atts['VALIDMIN'][0] if isinstance(atts['VALIDMIN'], (list, np.ndarray)) else atts['VALIDMIN']
                                val[val < vmin] = np.nan
                            if 'VALIDMAX' in atts:
                                vmax = atts['VALIDMAX'][0] if isinstance(atts['VALIDMAX'], (list, np.ndarray)) else atts['VALIDMAX']
                                val[val > vmax] = np.nan
                    except Exception:
                        pass # If the variable lacks attributes, bypass masking gracefully
                
                if var not in a_data: a_data[var] = []
                a_data[var].append(val)
    
    if not a_data:
        raise ValueError(f"No WaveAnalysis data found for range {start_dt} to {end_dt}")

    for var in a_data: 
        a_data[var] = np.concatenate(a_data[var], axis=0)
        # --- CONVERT CDF FILL VALUES BACK TO NaNs ---
        if np.issubdtype(a_data[var].dtype, np.floating):
            a_data[var][a_data[var] <= -1e30] = np.nan
    a_data['frequencies'] = freq_axis

    t_b = tt2000_to_datetime64(a_data['mag_time'])
    t_fft = tt2000_to_datetime64(a_data['fft_time'])
    
    # RTN Br
    rtn_dir = os.path.join(RTN_ROOT, f"{start_dt.strftime('%Y')}/{start_dt.strftime('%m')}/")
    rtn_file = get_latest_version_file(rtn_dir, f"psp_fld_l2_mag_RTN_1min_{start_dt.strftime('%Y%m%d')}_v*.cdf")
    if rtn_file:
        with open_local_cdf(rtn_file) as cdf_r:
            br_times = tt2000_to_datetime64(cdf_r.varget('epoch_mag_RTN_1min'))
            rtn_full = cdf_r.varget('psp_fld_l2_mag_RTN_1min')
            br_out = rtn_full[:, 0] if rtn_full.ndim > 1 else rtn_full

    # Ephemeris
    ephem_file = os.path.join(EPHEM_ROOT, "psp_helio1hr_position_20180813_v01.cdf")
    if os.path.exists(ephem_file):
        with open_local_cdf(ephem_file) as cdf_e:
            e_time = ephem_epoch_to_datetime(cdf_e.varget('epoch'))
            e_time_num = mdates.date2num(e_time)
            for var in ['RAD_AU', 'HG_LAT', 'HG_LON']:
                data_raw = cdf_e.varget(var)
                mask = (data_raw > -1e10) & (data_raw < 1e10)
                if not np.any(mask): continue
                clean_time = e_time_num[mask]
                clean_val = data_raw[mask]
                if var == 'RAD_AU' and np.nanmedian(clean_val) > 1000:
                    clean_val /= 149597870.7 
                if var == 'HG_LON': clean_val %= 360
                ephem_out[var] = interp1d(clean_time, clean_val, bounds_error=False, fill_value="extrapolate")

    # --- v1.4+ COMPATIBILITY BRIDGE ---
    # B-Field
    if 'B_fieldAligned' in a_data:
        a_data['Bn'] = a_data['B_fieldAligned'][:, 0]
        a_data['Bp'] = a_data['B_fieldAligned'][:, 1]
        a_data['Bq'] = a_data['B_fieldAligned'][:, 2]
        
    # E-Field
    if 'E_fieldAligned' in a_data:
        a_data['En'] = a_data['E_fieldAligned'][:, 0]
        a_data['Ep'] = a_data['E_fieldAligned'][:, 1]
        a_data['Eq'] = a_data['E_fieldAligned'][:, 2]
        
    # Poynting Vector
    if 'S_fieldAligned' in a_data:
        a_data['Sn'] = a_data['S_fieldAligned'][:, :, 0]
        a_data['Sp'] = a_data['S_fieldAligned'][:, :, 1]
        a_data['Sq'] = a_data['S_fieldAligned'][:, :, 2]
        
    # Wave Vector
    if 'k_hat_fieldAligned' in a_data:
        a_data['kn'] = a_data['k_hat_fieldAligned'][:, :, 0]
        a_data['kp'] = a_data['k_hat_fieldAligned'][:, :, 1]
        a_data['kq'] = a_data['k_hat_fieldAligned'][:, :, 2]

    return t_b, t_fft, a_data, br_times, br_out, ephem_out

############################################################################
# 2. PLOTTING CORE
############################################################################
def generate_plot(start_dt, end_dt, mode='batch', use_log_y=False, force=False):
    if mode == 'batch':
        out_dir = os.path.join(PLOT_OUT_ROOT, start_dt.strftime('%Y/%m'))
    else:
        out_dir = ONE_OFF_OUT_ROOT
    
    fname = f"PSP_WaveAnalysis_{start_dt.strftime('%Y-%m-%d_%H%M')}.png"
    save_path = os.path.join(out_dir, fname)

    if mode == 'batch' and not force and os.path.exists(save_path):
        print(f"Skipping Batch: {fname} (exists)")
        return

    print(f"    Processing {mode.upper()}: {fname}...")
    t_b, t_fft, data, t_br, val_br, eph_funcs = load_data(start_dt, end_dt)
    freqs = data['frequencies']
    f_max = np.nanmax(freqs)

    fig = plt.figure(figsize=(14, 15))
    gs = gridspec.GridSpec(7, 1, height_ratios=[1.2, 1, 1, 1, 1, 1, 0.4])
    plt.subplots_adjust(hspace=0.08, left=0.12, right=0.82, top=0.94, bottom=0.12)
    
    # --- 1. B-Field Panel with Robust Auto-scaling ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_b, data['Bn'], color='black', lw=0.8, label=r'$B_{||}$')
    ax1.plot(t_b, data['Bp'], color='red',   lw=0.8, label=r'$B_{\perp T}$')
    ax1.plot(t_b, data['Bq'], color='blue',  lw=0.8, label=r'$B_{\perp N}$')
    
    if val_br is not None:
        ax1.plot(t_br, val_br, color='#cc5500', lw=2.0, label=r'$B_R$', alpha=0.6)
    
    # Find min/max specifically for the visible window, ignoring non-finite/fill values
    b_visible_mask = (t_b >= start_dt) & (t_b <= end_dt)
    # 1. Handle High-Cadence Data (Use percentiles to clip spikes)
    b_high_cadence = np.concatenate([data['Bn'][b_visible_mask], data['Bp'][b_visible_mask], data['Bq'][b_visible_mask]])
    b_hc_finite = b_high_cadence[(b_high_cadence > -1e10) & (b_high_cadence < 1e10)]
    
    if len(b_hc_finite) > 0:
        b_min, b_max = np.nanpercentile(b_hc_finite, [1, 99])
    else:
        b_min, b_max = -10, 10 # Fallback
        
    # 2. Handle Low-Cadence Br Data (Use absolute min/max because it's already averaged)
    if val_br is not None:
        br_visible_mask = (t_br >= start_dt) & (t_br <= end_dt)
        br_data = val_br[br_visible_mask]
        br_finite = br_data[(br_data > -1e10) & (br_data < 1e10)]
        
        if len(br_finite) > 0:
            b_min = min(b_min, np.nanmin(br_finite))
            b_max = max(b_max, np.nanmax(br_finite))

    # Apply limits with padding
    pad = (b_max - b_min) * 0.1
    ax1.set_ylim(b_min - pad, b_max + pad)

    ax1.set_ylabel('B (nT)', fontweight='bold', fontsize=12)
    ax1.set_xlim(start_dt, end_dt)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False, 
               fontsize=10, handlelength=1.0, labelspacing=0.3)
    ax1.set_title(f"PSP WaveAnalysis  {start_dt.strftime('%Y-%m-%d %H:%M')}  {VERSION}", fontsize=14)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # --- 2-6. Spectrograms ---
    spec_vars = ['B_power_perp', 'S_theta', 'ellipticity', 'coherency', 'wave_normal']
    cmaps = ['turbo', 'RdBu_r', 'RdBu_r', 'gray_r', 'gray']
    z_titles = ['PSD $B_{kperp}$'+'\n'+'($nT^2/Hz$)', r'$S_{\theta}$', 'Ellipticity'+'\n'+'RH        LH', 'Coherency', 'Wave Normal']
    
    limits = {
        'B_power_perp': LogNorm(1e-2, 1e3),
        'S_theta': Normalize(0, 180),
        'ellipticity': Normalize(-1, 1),
        'coherency': Normalize(0, 1),
        'wave_normal': Normalize(0, 90)
    }
    
    # Convert times to numbers and filter for pcolormesh
    t_fft_num = mdates.date2num(t_fft)
    valid_time_mask = np.isfinite(t_fft_num)
    valid_freq_mask = np.isfinite(freqs)
    
    t_clean = t_fft_num[valid_time_mask]
    f_clean = freqs[valid_freq_mask]

    for i, var in enumerate(spec_vars):
        ax = fig.add_subplot(gs[i+1], sharex=ax1)
        norm = limits.get(var, Normalize())
        
        # Correctly slicing the data array using the masks to match clean axes
        z_data = data[var][valid_time_mask, :][:, valid_freq_mask].T
        
        mesh = ax.pcolormesh(t_clean, f_clean, z_data, cmap=cmaps[i], norm=norm, shading='auto')
        ax.set_ylabel('Frequency!C(Hz)', fontsize=10)
        
        pos = ax.get_position()
        cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.012, pos.height])
        cb = fig.colorbar(mesh, cax=cax)
        cb.set_label(z_titles[i], fontsize=11, fontweight='bold', rotation=270, labelpad=25)
        cb.ax.tick_params(labelsize=9)

        if use_log_y:
            ax.set_yscale('log')
            ax.set_ylim(0.5, f_max)
        else:
            ax.set_ylim(0, f_max)
            
        if i < 4:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # --- 7. Ephemeris Footer ---
    ax_table = fig.add_subplot(gs[6])
    ax_table.axis('off')
    
    tick_times = []
    curr = start_dt.replace(minute=0, second=0)
    while curr <= end_dt:
        tick_times.append(curr); curr += timedelta(hours=1)
    
    tick_nums = mdates.date2num(tick_times)
    row_names = ['CARR_LON', 'CARR_LAT', 'SC_R_AU']
    eph_keys = ['HG_LON', 'HG_LAT', 'RAD_AU']
    label_x_offset = -0.18 

    for row_idx, (name, key) in enumerate(zip(row_names, eph_keys)):
        y_pos = 0.32 - (row_idx * 0.3) 
        ax_table.text(label_x_offset, y_pos, name, transform=ax_table.transAxes, 
                      fontweight='bold', fontsize=10, ha='left')
        
        if key in eph_funcs:
            vals = eph_funcs[key](tick_nums)
            for val, t_num in zip(vals, tick_nums):
                x_rel = (t_num - mdates.date2num(start_dt)) / (mdates.date2num(end_dt) - mdates.date2num(start_dt))
                if 0 <= x_rel <= 1:
                    ax_table.text(x_rel, y_pos, f"{val:.3f}", transform=ax_table.transAxes, 
                                  ha='center', fontsize=12)

    ax_table.text(0.5, -0.7, 'Time (UTC)', transform=ax_table.transAxes, 
                  ha='center', fontweight='bold', fontsize=11)
    ax_table.text(0.5, -0.9, f"{start_dt.strftime('%Y-%m-%d (%j) %H:%M')} to {end_dt.strftime('%H:%M')}", 
                  transform=ax_table.transAxes, ha='center', fontsize=10)

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight') 
    print(f"    Successfully saved: {save_path}")
    plt.close(fig)

############################################################################
# 3. WRAPPERS
############################################################################
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
psp_root = os.path.dirname(current_dir)
if psp_root not in sys.path:
    sys.path.insert(0, psp_root)

import config

server_list = ['fc.cfa.harvard.edu', 'machida']
host_name = socket.gethostname()
gdrive = config.get_drive_path()

VERSION = 'v1.4'

ANALYSIS_ROOT = os.path.join(gdrive, 'Research/PSP/WaveAnalysis/WaveAnalysis_Files', VERSION)
RTN_ROOT = os.path.join(gdrive, 'Research/Data/AutoplotCache/http/research.ssl.berkeley.edu/data/spp/data/sci/fields/l2/mag_RTN_1min/') 
EPHEM_ROOT = os.path.join(gdrive, 'Research/Data/AutoplotCache/https/cdaweb.gsfc.nasa.gov/sp_phys/data/psp/ephemeris/helio1hr/')
PLOT_OUT_ROOT = os.path.join(gdrive, 'Research/PSP/WaveAnalysis/Plots/Static', VERSION)
ONE_OFF_OUT_ROOT = os.path.join(gdrive, 'Research/PSP/WaveAnalysis/OneOff_Plots')

def run(start_str, end_str, mode='batch', use_log_y=False, force=False):
    t0 = datetime.strptime(start_str, '%Y-%m-%d %H:%M')
    t1 = datetime.strptime(end_str, '%Y-%m-%d %H:%M')
    
    if mode == 'fly':
        generate_plot(t0, t1, mode='fly', use_log_y=use_log_y, force=force)
    else:
        curr = t0
        while curr < t1:
            nxt = curr + timedelta(hours=6)
            try:
                generate_plot(curr, nxt, mode='batch', use_log_y=use_log_y, force=force)
            except ValueError as e:
                print(f"--- SKIP: {curr.strftime('%Y-%m-%d %H:%M')} | Reason: {e}")
            except Exception as e:
                print(f"!!! ERROR: {curr.strftime('%Y-%m-%d %H:%M')} failed with: {e}")
            curr = nxt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("start")
    parser.add_argument("end")
    parser.add_argument("--mode", default='fly')
    parser.add_argument("--logy", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--v_out", default="1.4", help="WaveAnalysis version directory")
    args = parser.parse_args()
    
    # Update global path roots on the fly with the passed version!
    VERSION = f"v{args.v_out}" if not args.v_out.startswith('v') else args.v_out
    ANALYSIS_ROOT = os.path.join(gdrive, 'Research/PSP/WaveAnalysis/WaveAnalysis_Files', VERSION)
    PLOT_OUT_ROOT = os.path.join(gdrive, 'Research/PSP/WaveAnalysis/Plots/Static', VERSION)
    
    run(args.start, args.end, mode=args.mode, use_log_y=args.logy, force=args.force)
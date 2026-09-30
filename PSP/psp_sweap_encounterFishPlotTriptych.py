import os
import sys
import glob
import socket
import argparse
import datetime
import tempfile
import shutil
from contextlib import contextmanager
import numpy as np
from spacepy import pycdf
import matplotlib.pyplot as plt
from matplotlib.dates import date2num

# Import your tools to dynamically find the encounter times
sys.path.insert(0, "/home/kpaulson/psp_code")
sys.path.insert(0, "G:/My Drive/Software/Python/githubScripts_python")
import spc_tools as spct

# --- RAMDISK / TEMP STORAGE CONFIG ---
# Fallback to standard temp directory if /dev/shm (Linux RAM disk) isn't available
CUSTOM_TMP = '/dev/shm/py-tmp-kpaulson' if os.path.exists('/dev/shm') else tempfile.gettempdir()

@contextmanager
def smart_cdf(filepath):
    """Safely copies a CDF to a local temp dir before reading to bypass rclone latency."""
    os.makedirs(CUSTOM_TMP, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(suffix='.cdf', dir=CUSTOM_TMP)
    os.close(fd) 
    try:
        shutil.copy2(filepath, temp_path)
        with pycdf.CDF(temp_path) as cdf:
            yield cdf
    finally:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass

def get_valid_root(path_options):
    """Returns the first base path in the list that exists on this machine."""
    for path in path_options:
        if os.path.exists(path):
            return path
    return path_options[0]

def get_spc_l3_data(start_dt, end_dt, base_path):
    """Loads SPC L3 moments using the fast-copy context manager."""
    spc_data = {'Epoch': [], 'np_moment': [], 'wp_moment': [], 'vp_moment_SC': []}
    
    current_dt = start_dt
    while current_dt <= end_dt:
        yyyy, mm, dd = f"{current_dt.year:04d}", f"{current_dt.month:02d}", f"{current_dt.day:02d}"
        glob_str = os.path.join(base_path, 'data', 'sci', 'sweap', 'spc', 'L3', yyyy, mm, f'spp_swp_spc_l3i_{yyyy}{mm}{dd}_v*.cdf')
        files = sorted(glob.glob(glob_str))
        
        if files:
            try:
                with smart_cdf(files[-1]) as dat:
                    for key in spc_data.keys():
                        spc_data[key].append(dat[key][...])
            except Exception as e:
                print(f"Error reading {files[-1]}: {e}")
                
        current_dt += datetime.timedelta(days=1)
        
    for key in spc_data.keys():
        spc_data[key] = np.concatenate(spc_data[key]) if spc_data[key] else np.array([])
            
    return spc_data

def get_span_l3_data(start_dt, end_dt, base_path):
    """Loads SPAN L3 moments using the fast-copy context manager."""
    span_data = {'Epoch': [], 'DENS': [], 'TEMP': [], 'VEL_SC': []}
    
    current_dt = start_dt
    while current_dt <= end_dt:
        yyyy, mm, dd = f"{current_dt.year:04d}", f"{current_dt.month:02d}", f"{current_dt.day:02d}"
        glob_str = os.path.join(base_path, 'data', 'sci', 'sweap', 'spi', 'L3', 'spi_sf00', yyyy, mm, f'psp_swp_spi_sf00_L3_mom_{yyyy}{mm}{dd}_v*.cdf')
        files = sorted(glob.glob(glob_str))
        
        if files:
            try:
                with smart_cdf(files[-1]) as dat:
                    for key in span_data.keys():
                        span_data[key].append(dat[key][...])
            except Exception as e:
                print(f"Error reading {files[-1]}: {e}")
                
        current_dt += datetime.timedelta(days=1)
        
    for key in span_data.keys():
        span_data[key] = np.concatenate(span_data[key]) if span_data[key] else np.array([])
            
    return span_data


def main(enc=None, verbose=False, noshow=False):
    if verbose:
        print(f"[*] Starting plot for Encounter {enc}")

    # --- 1. DYNAMIC MACHINE-AGNOSTIC PATHING ---
    hostName = socket.gethostname()
    
    # Let the script figure out where Google Drive lives on this specific OS
    drive_options = [
        os.path.expanduser('~/MyDrive'),              # Linux / WSL Rclone
        'G:/My Drive',                                # Windows Native Drive
        os.path.expanduser('~/Google Drive/My Drive') # Mac / Legacy Windows
    ]
    DRIVE_ROOT = get_valid_root(drive_options)
    
    shared_options = [
        os.path.expanduser('~/sharedDrive_PSP-SWEAP'),
        'G:/Shared drives/PSP_SWEAP',
        os.path.expanduser('~/Google Drive/Shared drives/PSP_SWEAP')
    ]
    PSP_SHAREDDRIVE_ROOT = get_valid_root(shared_options)
    
    # 1. Shared Ops Path
    ops_options = [
        '/home/kpaulson/sharedDrives_PSP-SWEAP/INSTRUMENT/SPC/Operations/',
        os.path.join(PSP_SHAREDDRIVE_ROOT, 'INSTRUMENT/SPC/Operations/')    
    ]
    sharedOpsDir = get_valid_root(ops_options)

    # 2. Science Data Path (Smarter Host Routing)
    if 'fc' in hostName or 'machida' in hostName:
        data_options = ['/psp/']
    else:
        # Laptop environment: prioritize the Autoplot caches!
        data_options = [
            os.path.join(DRIVE_ROOT, 'Research/Data/AutoplotCache/https/w3sweap.cfa.harvard.edu/'),
            os.path.join(DRIVE_ROOT, 'Research/Data/AutoplotCache/http/w3sweap.cfa.harvard.edu/'),
            os.path.join(DRIVE_ROOT, 'Research/Data/AutoplotCache/http/research.ssl.berkeley.edu/data/spp/'),
            '/psp/' 
        ]
    
    dataPath = get_valid_root(data_options)
    
    if args.verbose:
        print(f"[*] Ops Path Resolved To: {sharedOpsDir}")
        print(f"[*] Data Path Resolved To: {dataPath}")

    # --- 2. LOAD EPHEMERIS & MATHEMATICALLY FIND PERIHELIA ---
    print(f"Loading Ephemeris from {dataPath}...")
    ephem_file = os.path.join(sharedOpsDir, 'data', 'sci', 'sweap', 'spc', 'EPHEMERIS', 'psp_helio1hr_position_20180813_v01.cdf')
    
    with smart_cdf(ephem_file) as eph:
        eph_tm = eph['Epoch'][...]
        
        # Safely handle the two different versions of the ephemeris files
        if 'CARR_LON' in eph:
            eph_lon = eph['CARR_LON'][...] * (np.pi / 180.0) 
            eph_r = eph['SC_R_AU'][...]
        elif 'HG_LON' in eph:
            eph_lon = eph['HG_LON'][...] * (np.pi / 180.0)
            eph_r = eph['RAD_AU'][...]
        else:
            raise KeyError("Neither CARR_LON nor HG_LON found in ephemeris CDF!")

    print("Calculating perihelia from orbital mechanics...")
    from scipy.signal import find_peaks
    
    # We want minimums (perihelia), so we find the peaks of the NEGATIVE radial distance.
    # The ephemeris is 1-hour cadence. distance=1500 (62 days) ensures we only 
    # catch one minimum per orbit, since the shortest PSP orbit is ~88 days.
    peaks, _ = find_peaks(-eph_r, distance=1500)
    peri_dates = eph_tm[peaks]
    
    # --- ENCOUNTER SELECTION ---
    if args.enc is not None:
        plot_enc_num = args.enc
        next_enc_num = plot_enc_num + 1
        
        if plot_enc_num > len(peri_dates):
            print(f"Error: Encounter {plot_enc_num} exceeds available ephemeris data.")
            sys.exit(1)
            
        # Enc 1 is idx 0, Enc 28 is idx 27, etc.
        perihelion = peri_dates[plot_enc_num - 1] 
    else:
        # Find the most recently COMPLETED perihelion
        past_peris = np.where(peri_dates < datetime.datetime.now())[0]
        
        if len(past_peris) == 0:
            print("No past encounters found in ephemeris file!")
            sys.exit(1)
            
        last_enc_idx = past_peris[-1]
        plot_enc_num = last_enc_idx + 1  # The encounter to plot (e.g., 28)
        next_enc_num = plot_enc_num + 1  # The upcoming encounter we are prepping for (e.g., 29)
        
        perihelion = peri_dates[last_enc_idx]
    
    print(f"Generating Fish Plots for Encounter {plot_enc_num} (Perihelion: {perihelion.strftime('%Y-%m-%d')})")
    print(f"--> Saving into Pre-Encounter Review folder for Encounter {next_enc_num}")
    
    # Mirror the Autoplot time ranges (~48 days for SPC orbit track)
    spc_start = perihelion - datetime.timedelta(days=48)
    spc_end = perihelion + datetime.timedelta(days=48)

    # --- 3. DATA LOADING ---
    print("Loading SPC L3 Data...")
    spc_vars = get_spc_l3_data(spc_start, spc_end, dataPath)
    spc_tm = spc_vars['Epoch']
    spc_np = spc_vars['np_moment']
    spc_wp = spc_vars['wp_moment']
    
    if len(spc_vars['vp_moment_SC']) > 0:
        vp_spc = spc_vars['vp_moment_SC'].copy()
        vp_spc[vp_spc < -1e30] = np.nan
        spc_speed = np.linalg.norm(vp_spc, axis=1)
    else:
        spc_speed = np.array([])

    # --- DYNAMIC SPAN GAP-FILL LOGIC ---
    print("Detecting SPC thermal shutoff gap...")
    near_peri_mask = (spc_tm > perihelion - datetime.timedelta(days=5)) & (spc_tm < perihelion + datetime.timedelta(days=5))
    spc_tm_near = spc_tm[near_peri_mask]
    
    if len(spc_tm_near) > 1:
        time_diffs = np.diff(spc_tm_near)
        gap_idx = np.argmax(time_diffs)
        span_start = spc_tm_near[gap_idx]
        span_end = spc_tm_near[gap_idx + 1]
        print(f"--> SPC Shutoff Detected: {span_start.strftime('%Y-%m-%d %H:%M')} to {span_end.strftime('%Y-%m-%d %H:%M')}")
    else:
        print("--> Could not definitively find SPC gap. Falling back to +/- 5 days.")
        span_start = perihelion - datetime.timedelta(days=5)
        span_end = perihelion + datetime.timedelta(days=5)

    print("Loading SPAN L3 Data...")
    span_vars = get_span_l3_data(span_start, span_end, dataPath)
    
    # The loader returns full calendar days. We must strictly crop it to the exact SPC gap!
    span_tm_full = span_vars['Epoch']
    
    if len(span_tm_full) > 0:
        # Create a 2-hour buffer so the instrument handoff is visually obvious
        gap_buffer = datetime.timedelta(hours=2)
        valid_span = (span_tm_full > (span_start + gap_buffer)) & (span_tm_full < (span_end - gap_buffer))
        
        span_tm = span_tm_full[valid_span]
        span_np = span_vars['DENS'][valid_span]
        
        # Apply the mask to the temperature and velocity calculations as well
        valid_temp = span_vars['TEMP'][valid_span].copy()
        valid_temp[valid_temp <= 0] = np.nan
        span_wp = np.sqrt((valid_temp * 11604.525 * 1.3807e-16) / 1.6726e-24) / 100000.0
        
        vp_span = span_vars['VEL_SC'][valid_span].copy()
        vp_span[vp_span < -1e30] = np.nan
        span_speed = np.linalg.norm(vp_span, axis=1)
    else:
        span_tm = np.array([])
        span_np = np.array([])
        span_wp = np.array([])
        span_speed = np.array([])

    # --- 4. INTERPOLATE ORBITAL MECHANICS ---
    print("Synchronizing and wrapping coordinate systems...")
    eph_tm_num = date2num(eph_tm)
    spc_tm_num = date2num(spc_tm)
    span_tm_num = date2num(span_tm)

    eph_lon_unwrapped = np.unwrap(eph_lon)
    
    spc_lon = np.interp(spc_tm_num, eph_tm_num, eph_lon_unwrapped) % (2 * np.pi)
    spc_r = np.interp(spc_tm_num, eph_tm_num, eph_r)
    spc_x = spc_r * np.cos(spc_lon)
    spc_y = spc_r * np.sin(spc_lon)
    spc_dens_r2 = spc_np * (spc_r**2)
    
    if len(span_np) > 0:
        span_lon = np.interp(span_tm_num, eph_tm_num, eph_lon_unwrapped) % (2 * np.pi)
        span_r = np.interp(span_tm_num, eph_tm_num, eph_r)
        span_x = span_r * np.cos(span_lon)
        span_y = span_r * np.sin(span_lon)
        span_dens_r2 = span_np * (span_r**2)
    else:
        span_x, span_y, span_dens_r2 = [], [], []

    # --- 5. PLOTTING THE TRIPTYCH ---
    print("Drawing Fish Plots...")
    
    # Universally boost the font size across the entire plot
    plt.rcParams.update({
        'font.size': 16,          # General text (like colorbar tick numbers)
        'axes.labelsize': 20,     # X and Y axis labels
        'xtick.labelsize': 16,    # X tick numbers
        'ytick.labelsize': 16     # Y tick numbers
    })

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharex=True, sharey=True, constrained_layout=True, gridspec_kw={'wspace': 0.15})
    
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlim(-0.35, 0.35)
        ax.set_ylim(-0.15, 0.55)
        ax.set_xlabel('X$_{CARR}$ [AU] (rotating)')
        
        # --- FIX: Crop the background orbit line to just this specific encounter! ---
        eph_plot_mask = (eph_tm >= spc_start) & (eph_tm <= spc_end)
        eph_lon_plot = eph_lon[eph_plot_mask]
        eph_r_plot = eph_r[eph_plot_mask]
        
        eph_x = eph_r_plot * np.cos(eph_lon_plot)
        eph_y = eph_r_plot * np.sin(eph_lon_plot)
        
        # Mask the wrap-around jump in the background line so it doesn't cross the sun
        jump_mask = np.abs(np.diff(eph_lon_plot)) > np.pi
        eph_x[:-1][jump_mask] = np.nan 
        
        ax.plot(eph_x, eph_y, color='grey', linewidth=1.0, zorder=1)

    axes[0].set_ylabel('Y$_{CARR}$ [AU] (rotating)')

    def plot_panel(ax, z_spc, z_span, vmin, vmax, cmap, clabel):
        sc1 = ax.scatter(spc_x, spc_y, c=z_spc, cmap=cmap, vmin=vmin, vmax=vmax, s=20, zorder=3)
        if len(span_x) > 0:
            ax.scatter(span_x, span_y, color='black', s=60, zorder=2)
            sc2 = ax.scatter(span_x, span_y, c=z_span, cmap=cmap, vmin=vmin, vmax=vmax, s=20, zorder=4)
        cbar = fig.colorbar(sc1, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(clabel, rotation=270, labelpad=25, fontsize=20)

    cmap_choice = 'turbo' 
    plot_panel(axes[0], spc_dens_r2, span_dens_r2, 0, 20, cmap_choice, 'density*R$^2$ [cm$^{-3}$AU$^2$]')
    plot_panel(axes[1], spc_wp, span_wp, 0, 80, cmap_choice, 'thermal speed [km/s]')
    plot_panel(axes[2], spc_speed, span_speed, 0, 800, cmap_choice, 'speed [km/s]')


    # --- 6. AUTOMATED SAVING ---
    # Save into the upcoming encounter's review folder
    out_dir = os.path.join(sharedOpsDir, 'encounterPlanning', f'encounter{next_enc_num}', 'preEncounterReview')
    os.makedirs(out_dir, exist_ok=True)
    
    today_str = datetime.datetime.now().strftime('%Y%m%d')
    # Name the file based on the data it ACTUALLY contains (the past encounter)
    out_file = os.path.join(out_dir, f'spc_span_fishpPlotTriptych_enc{plot_enc_num}.png')
    
    print(f'Saving plot to: {out_file}')
    fig.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    if not args.noshow:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automated Fish Plots')
    parser.add_argument('-ns', '--noshow', action='store_true', help='Bypass plt.show()')
    parser.add_argument('-e', '--enc', type=int, default=None, help='Specific encounter')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print detailed logs')
    
    args = parser.parse_args()
    
    # Pass the parsed flags directly into your clean main function
    main(enc=args.enc, verbose=args.verbose, noshow=args.noshow)
import os
import re
import urllib.request
import datetime
import numpy as np
import spiceypy
import socket
import sys
import platform

# Look up one level to find config file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import config

# Prevent infinite hangs if NASA SPDF is unresponsive
socket.setdefaulttimeout(15)

# --- CACHE LOCATION ROUTER ---
hostname = socket.gethostname().lower()

# 1. Dedicated permanent cache for CfA unix nodes (prevents /tmp purges)
if any(node in hostname for node in ['fc.cfa', 'machida', 'fc', 'machida.cfa']):
    LOCAL_CACHE_PATH = '/home/kpaulson/PSP/ezephem_cache/'

# 2. Windows fallback
elif platform.system() == 'Windows':
    LOCAL_CACHE_PATH = os.path.join(os.environ.get('TEMP', 'C:\\Temp'), 'ezephem_cache')

# 3. Standard Linux / RAM-disk fallback
else:
    tmp_base = config.get_tmpMirror()
    if not tmp_base or not os.path.exists(os.path.dirname(tmp_base)):
        tmp_base = '/tmp'
    LOCAL_CACHE_PATH = os.path.join(tmp_base, 'ezephem_cache')

os.makedirs(LOCAL_CACHE_PATH, exist_ok=True)
# -----------------------------

def get_all_kernels_http(date_list):
    """
    Downloads and caches necessary SPICE kernels from the NASA SPDF HTTPS server.
    Optimized to strictly check the local cache FIRST to prevent NASA SPDF throttling.
    """
    cache_path = LOCAL_CACHE_PATH
    base_url = 'https://spdf.gsfc.nasa.gov/pub/data/psp/ephemeris/spice/'
    
    dirs = {
        'lsk': 'leap_second_kernel/',
        'sclk': 'operations_sclk_kernel/',
        'frame': 'frame_kernel/',
        'recon_ephem': 'reconstructed_ephemeris/',
        'ah': 'attitude_history/',
        'plan_ephem': 'planetary_ephemeris/',
        'pck': 'planetary_constant_kernel/'
    }
    
    kernel_paths = []
    
    # 1. Snapshot the local cache once to avoid repeated disk I/O
    local_files = os.listdir(cache_path)
    
    def get_cached_or_download(subdir, ext_pattern, specific_match=None):
        """Checks local cache first. Only scrapes NASA if the file is missing."""
        # Check Local First
        matches = [f for f in local_files if re.search(ext_pattern, f)]
        if specific_match:
            matches = [f for f in matches if specific_match in f]
            
        if matches:
            return os.path.join(cache_path, sorted(matches)[-1])
            
        # Fallback to NASA SPDF
        print(f"      [*] Kernel missing locally. Scraping NASA SPDF for {ext_pattern}...")
        url = base_url + subdir
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                html = response.read().decode('utf-8')
                
            remote_matches = sorted(list(set(re.findall(rf'href="([^"]+{ext_pattern})"', html))))
            if specific_match:
                remote_matches = [f for f in remote_matches if specific_match in f]
            
            if remote_matches:
                filename = remote_matches[-1]
                local_path = os.path.join(cache_path, filename)
                print(f"      Downloading {filename}...")
                urllib.request.urlretrieve(url + filename, local_path)
                local_files.append(filename) # Update local knowledge
                return local_path
        except Exception as e:
            print(f"      [!] Failed to download from {url}: {e}")
            
        return None

    def is_date_covered_by_bsp(target_dt):
        """Checks if a target date is already covered by a locally cached Recon BSP."""
        for f in local_files:
            if f.startswith('spp_recon_') and f.endswith('.bsp'):
                parts = f.split('_')
                if len(parts) >= 4:
                    try:
                        start_d = datetime.datetime.strptime(parts[2], '%Y%m%d')
                        end_d   = datetime.datetime.strptime(parts[3], '%Y%m%d')
                        if start_d <= target_dt <= end_d:
                            return os.path.join(cache_path, f)
                    except: pass
        return None

    # --- STATIC KERNELS (Only download once ever) ---
    k = get_cached_or_download(dirs['lsk'], r'\.tls')
    if k: kernel_paths.append(k)
    
    k = get_cached_or_download(dirs['sclk'], r'\.tsc')
    if k: kernel_paths.append(k)
    
    k = get_cached_or_download(dirs['pck'], r'\.tpc')
    if k: kernel_paths.append(k)
    
    k = get_cached_or_download(dirs['plan_ephem'], r'\.bsp')
    if k: kernel_paths.append(k)
    
    k1 = get_cached_or_download(dirs['frame'], r'\.tf', specific_match='spp_v')
    k2 = get_cached_or_download(dirs['frame'], r'\.tf', specific_match='spp_dyn')
    if k1: kernel_paths.append(k1)
    if k2: kernel_paths.append(k2)
        
    # --- TIME-DEPENDENT KERNELS (Check specific dates) ---
    for dt in date_list:
        year_str = dt.strftime('%Y')
        doy_str  = dt.strftime('%Y_%j') 
        
        # Attitude History (.bc)
        ah_year_subdir = dirs['ah'] + year_str + '/'
        k = get_cached_or_download(ah_year_subdir, r'\.bc', specific_match=doy_str)
        if k: kernel_paths.append(k)
            
        # Reconstructed Orbit Ephemeris (.bsp)
        bsp_path = is_date_covered_by_bsp(dt)
        if bsp_path:
            kernel_paths.append(bsp_path)
        else:
            # If not covered locally, scrape NASA for the missing time window
            print(f"      [*] Orbit BSP missing for {dt.strftime('%Y-%m-%d')}. Scraping NASA SPDF...")
            url = base_url + dirs['recon_ephem']
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req) as response:
                    html = response.read().decode('utf-8')
                remote_bsps = sorted(list(set(re.findall(r'href="(spp_recon_[^"]+\.bsp)"', html))))
                
                # Find the remote BSP that bounds our target date
                for bsp in remote_bsps:
                    parts = bsp.split('_')
                    if len(parts) >= 4:
                        try:
                            start_d = datetime.datetime.strptime(parts[2], '%Y%m%d')
                            end_d   = datetime.datetime.strptime(parts[3], '%Y%m%d')
                            if start_d <= dt <= end_d:
                                local_path = os.path.join(cache_path, bsp)
                                print(f"      Downloading {bsp}...")
                                urllib.request.urlretrieve(url + bsp, local_path)
                                local_files.append(bsp)
                                kernel_paths.append(local_path)
                                break
                        except: pass
            except Exception as e:
                print(f"      [!] Failed to scrape Recon BSPs: {e}")
                
    return list(set(kernel_paths)) # Remove duplicates


def get_spice_data(unix_times):
    """
    Returns both the SC->RTN rotation matrices and the Spacecraft Velocity.
    """
    dt_array = [datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).replace(tzinfo=None) for ts in unix_times]
    dt_dates = [datetime.datetime(i.year, i.month, i.day) for i in dt_array]
    date_list = np.unique(dt_dates)
    
    kernel_paths = get_all_kernels_http(date_list)
    
    if not kernel_paths:
        raise ValueError("Failed to download or locate SPICE kernels.")
    
    spiceypy.kclear()
    for file in kernel_paths:        
        if os.path.exists(file):
            spiceypy.furnsh(file)

    et_array = np.zeros(len(dt_array))
    for i, dt in enumerate(dt_array):
        utc_str = dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
        et_array[i] = spiceypy.utc2et(utc_str)
        
    cmat = np.zeros((len(et_array), 3, 3))
    v_sc_rtn = np.zeros((len(et_array), 3))
    
    for i, et in enumerate(et_array):
        try:
            cmat[i, :, :] = spiceypy.pxform('SPP_SPACECRAFT', 'SPP_RTN', et)
            state, _ = spiceypy.spkezr('-96', et, 'SPP_RTN', 'NONE', 'SUN')
            v_sc_rtn[i, :] = state[3:6]
        except spiceypy.support_types.SpiceyError as e:
            cmat[i, :, :] = np.nan
            v_sc_rtn[i, :] = np.nan
            
    spiceypy.kclear()
    return cmat, v_sc_rtn
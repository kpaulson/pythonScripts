import os
import sys
import glob
import argparse
import requests
from datetime import datetime, timedelta
from tqdm import tqdm

# --- MASTER CONFIG IMPORT (OPTIONAL FALLBACK) ---
SCRIPT_DIR = "/home/kpaulson/MyDrive/Software/Python/githubScripts_python" if os.name != 'nt' else "G:/My Drive/Software/Python/githubScripts_python"
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

try:
    import config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

def download_file(url, target_path):
    """Downloads a file via HTTP with a progress bar if not already present."""
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        print(f"  [Existing] {os.path.basename(target_path)}")
        return True

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    temp_path = target_path + ".tmp"

    try:
        resp = requests.get(url, stream=True, timeout=15)
        if resp.status_code != 200:
            return False
        total_size = int(resp.headers.get('content-length', 0))
        chunk_size = 1024 * 1024

        with open(temp_path, 'wb') as f, tqdm(
            desc=f"  [Downloading] {os.path.basename(target_path)}",
            total=total_size, unit='iB', unit_scale=True, unit_divisor=1024, leave=False
        ) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))

        os.replace(temp_path, target_path)
        print(f"  [Saved] {os.path.basename(target_path)}")
        return True
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"  [Failed] {url}: {e}")
        return False

def fetch_cdaweb_or_sweap(date_str, file_type, species_tag, out_dir):
    """Probes remote NASA/SWEAP servers for CDF files and downloads them."""
    date_dt = datetime.strptime(date_str, "%Y-%m-%d")
    yyyy, mm, dd = date_dt.strftime('%Y'), date_dt.strftime('%m'), date_dt.strftime('%d')
    formatted = f"{yyyy}{mm}{dd}"

    urls = []
    if file_type == 'span':
        lvl = 'L2B' if species_tag == '0a' else 'L2'
        subfolder = 'spi_sf0a' if species_tag == '0a' else f"spi_sf{species_tag}"
        file_prefix = f"psp_swp_spi_sf{species_tag}_L2B_mom_{formatted}" if species_tag == '0a' else f"psp_swp_spi_sf{species_tag}_L2_8Dx32Ex8A_{formatted}"
        
        for ver in range(9, -1, -1):
            ver_str = f"{ver:02d}"
            urls.append(f"https://w3sweap.cfa.harvard.edu/data/sci/sweap/spi/{lvl}/{subfolder}/{yyyy}/{mm}/{file_prefix}_v{ver_str}.cdf")
            urls.append(f"https://spdf.gsfc.nasa.gov/pub/data/psp/sweap/spi/{lvl}/{subfolder}/{yyyy}/{mm}/{file_prefix}_v{ver_str}.cdf")

    elif file_type == 'mag':
        prefix = f"psp_fld_l2_mag_SC_4_Sa_per_Cyc_{formatted}"
        for ver in range(9, -1, -1):
            ver_str = f"{ver:02d}"
            urls.append(f"http://research.ssl.berkeley.edu/data/spp/data/sci/fields/l2/mag_SC_4_Sa_per_Cyc/{yyyy}/{mm}/{prefix}_v{ver_str}.cdf")
            urls.append(f"https://spdf.gsfc.nasa.gov/pub/data/psp/fields/l2/mag_SC_4_Sa_per_Cyc/{yyyy}/{mm}/{prefix}_v{ver_str}.cdf")

    for url in urls:
        target_path = os.path.join(out_dir, os.path.basename(url))
        if download_file(url, target_path):
            return target_path
    return None

def main():
    parser = argparse.ArgumentParser(description="PSP SPAN-I Pre-fetcher & Local Cache Builder")
    parser.add_argument("--start-date", type=str, required=True, help="Start Date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="End Date YYYY-MM-DD (defaults to start-date)")
    parser.add_argument("--out-dir", type=str, default="./psp_data_cache", help="Local directory to store staged dataset")
    parser.add_argument("--species", type=str, default="both", help="Species: '00', '0a', or 'both'")
    args = parser.parse_args()

    t_start = datetime.strptime(args.start_date, "%Y-%m-%d")
    t_end = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else t_start
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n==================================================")
    print(f"  PSP Data Staging Tool")
    print(f"  Range: {args.start_date} to {t_end.strftime('%Y-%m-%d')}")
    print(f"  Target Local Cache: {out_dir}")
    print(f"==================================================\n")

    curr_dt = t_start
    while curr_dt <= t_end:
        date_str = curr_dt.strftime("%Y-%m-%d")
        print(f"--> Processing Date: {date_str}")

        # 1. MAG Data
        fetch_cdaweb_or_sweap(date_str, 'mag', None, out_dir)

        # 2. SPAN Protons / Alphas
        if args.species in ['00', 'both']:
            fetch_cdaweb_or_sweap(date_str, 'span', '00', out_dir)
        if args.species in ['0a', 'both']:
            fetch_cdaweb_or_sweap(date_str, 'span', '0a', out_dir)

        # 3. WaveAnalysis, Hammerhead, LFR, & CSVs (Copy from config if available)
        if HAS_CONFIG:
            yyyy, mm, dd = curr_dt.strftime('%Y'), curr_dt.strftime('%m'), curr_dt.strftime('%d')
            drive_root = config.get_drive_path()
            
            # WaveAnalysis
            wave_folder = os.path.join(drive_root, "Research", "PSP", "WaveAnalysis", "WaveAnalysis_Files", "v1.4", yyyy, mm)
            for hh in ['0000', '0600', '1200', '1800']:
                w_file = os.path.join(wave_folder, f"PSP_WaveAnalysis_{date_str}_{hh}_v1.4.cdf")
                if os.path.exists(w_file):
                    shutil.copy2(w_file, os.path.join(out_dir, os.path.basename(w_file)))
                    print(f"  [Staged] {os.path.basename(w_file)}")

            # Hammerhead
            ham_dir = os.path.join(drive_root, "Research", "PSP", "Hammerheads", "Hamstrings", "cdf", "v02")
            for h_match in glob.glob(os.path.join(ham_dir, f"hamstring_{yyyy}-{mm}-{dd}_v*.cdf")):
                shutil.copy2(h_match, os.path.join(out_dir, os.path.basename(h_match)))
                print(f"  [Staged] {os.path.basename(h_match)}")

            # Merged CSVs
            enc = None
            for e, (s_s, e_s) in config.ENCOUNTER_DATES.items():
                if datetime.strptime(s_s, '%Y-%m-%d') <= curr_dt <= datetime.strptime(e_s, '%Y-%m-%d') + timedelta(days=1):
                    enc = e
                    break
            if enc:
                csv_file = os.path.join(drive_root, '15min_DATA_Internal', f"E{enc:02d}.csv")
                if os.path.exists(csv_file):
                    shutil.copy2(csv_file, os.path.join(out_dir, os.path.basename(csv_file)))
                    print(f"  [Staged] {os.path.basename(csv_file)}")

        curr_dt += timedelta(days=1)

    print(f"\n[Complete] All files staged in: {out_dir}\n")

if __name__ == '__main__':
    main()
"""
spani_preprocess_vdf_slices.py
==============================
Optimized SPAN-I VDF Slice Pre-Processor with 3D Velocity LUT + 1D Index Arrays.
Outputs compact daily .json.gz files (~8-10 MB) directly to:
Research/PSP/JSON/team/E{enc}/psp_swp_spani_vdfs/psp_swp_spani_vdfSlice_YYYY-MM-DD.json.gz
"""

import os
import sys
import glob
import json
import gzip
import shutil
import tempfile
import argparse
import gc
from datetime import datetime, timedelta, timezone
import numpy as np
import cdflib
from tqdm import tqdm

# --- MASTER CONFIG IMPORT ---
SCRIPT_DIR = "/home/kpaulson/MyDrive/Software/Python/githubScripts_python" if os.name != 'nt' else "G:/My Drive/Software/Python/githubScripts_python"
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

try:
    import config
    DRIVE_ROOT = config.get_drive_path()
    SWEAP_DIR = config.get_sweapCacheData()
    ENCOUNTER_DATES = config.ENCOUNTER_DATES
    AUTOPLOT_CACHE = os.path.join(DRIVE_ROOT, "Research", "Data", "AutoplotCache")
except ImportError:
    DRIVE_ROOT, SWEAP_DIR, AUTOPLOT_CACHE = "", "", ""
    ENCOUNTER_DATES = {}

JSON_TEAM_ROOT = os.path.join(DRIVE_ROOT, "Research", "PSP", "JSON", "team") if DRIVE_ROOT else "JSON/team"

def get_encounter_from_date(target_dt):
    """Dynamically resolves Encounter number from config.py. Returns None if unmapped."""
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

    print(f"[!] Warning: Date {target_dt.strftime('%Y-%m-%d')} does not map to any Encounter in config.py.")
    return None

def prepare_4d_grid(grid_array, num_records):
    grid_flat = grid_array.flatten()
    elements = 2048
    if grid_flat.size == num_records * elements:
        return grid_flat.reshape((num_records, 8, 32, 8))
    return np.tile(grid_flat[:elements].reshape((8, 32, 8)), (num_records, 1, 1, 1))

def resolve_vdf_file(date_str, species_tag='00'):
    date_dt = datetime.strptime(date_str, "%Y-%m-%d")
    yyyy, mm = date_dt.strftime('%Y'), date_dt.strftime('%m')
    date_formatted = date_dt.strftime('%Y%m%d')

    file_prefix = f"psp_swp_spi_sf0a_L2B_mom_{date_formatted}" if species_tag == '0a' else f"psp_swp_spi_sf{species_tag}_L2_8Dx32Ex8A_{date_formatted}"
    lvl, subfolder = ('L2B', 'spi_sf0a') if species_tag == '0a' else ('L2', f"spi_sf{species_tag}")

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
                return matches[0]
    return None

def process_day(date_str, overwrite=False):
    cdf_p = None
    cdf_he = None
    try:
        date_dt = datetime.strptime(date_str, "%Y-%m-%d")
        enc_num = get_encounter_from_date(date_dt)

        if enc_num is None:
            print(f"[!] Skipping {date_str}: Encounter number could not be determined.")
            return None

        out_dir = os.path.join(JSON_TEAM_ROOT, f"E{enc_num}", "psp_swp_spani_vdfs")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"psp_swp_spani_vdfSlice_{date_str}.json.gz")

        if not overwrite and os.path.exists(out_file):
            print(f"[*] Pre-processed slice file already exists: {out_file}")
            return out_file

        file_p = resolve_vdf_file(date_str, '00')
        if not file_p:
            print(f"[!] Proton CDF not found for {date_str}, skipping.")
            return None

        print(f"\n--> Processing SPAN-I VDF slices for {date_str} (Encounter {enc_num})...")
        cdf_p = cdflib.CDF(file_p)
        times_all = cdflib.cdfepoch.unixtime(cdf_p.varget('Epoch'))
        num_records = len(times_all)

        if 'EFLUX' in cdf_p.cdf_info().zVariables:
            eflux_p_raw = cdf_p.varget('EFLUX')[:num_records]
        else:
            eflux_p_raw = cdf_p.varget('DATA')[:num_records] * 0.97e8

        theta_p = prepare_4d_grid(cdf_p.varget('THETA'), num_records)
        phi_p = prepare_4d_grid(cdf_p.varget('PHI'), num_records)
        energy_p = prepare_4d_grid(cdf_p.varget('ENERGY'), num_records)

        try: magf_p = cdf_p.varget('MAGF_INST')[:num_records]
        except Exception: magf_p = np.full((num_records, 3), np.nan)

        mass_p, charge_p = 0.010438870, 1.0

        eflux_p_4d = eflux_p_raw.reshape((num_records, 8, 32, 8))
        eflux_clean = np.where((eflux_p_4d < 0) | (~np.isfinite(eflux_p_4d)), np.nan, eflux_p_4d)
        energy_clean = np.where((energy_p <= 0) | (~np.isfinite(energy_p)), np.nan, energy_p)

        vdf_p = (eflux_clean / energy_clean) * (mass_p ** 2) / ((2e-5) * energy_clean)
        
        energy_geom_p = np.nan_to_num(energy_p, nan=100.0)
        theta_geom_p = np.nan_to_num(theta_p, nan=0.0)
        phi_geom_p = np.nan_to_num(phi_p, nan=0.0)

        vel_p_geom = np.sqrt(2 * charge_p * np.maximum(energy_geom_p, 1.0) / mass_p)
        rad_th_p, rad_ph_p = np.radians(theta_geom_p), np.radians(phi_geom_p)
        
        vx_p = vel_p_geom * np.cos(rad_ph_p) * np.cos(rad_th_p)
        vy_p = vel_p_geom * np.sin(rad_th_p)
        vz_p = vel_p_geom * np.sin(rad_ph_p) * np.cos(rad_th_p)

        p_xz = np.full((num_records, 8, 32), np.nan, dtype=np.float32)
        p_xy = np.full((num_records, 8, 32), np.nan, dtype=np.float32)
        p_phi_idx = np.zeros(num_records, dtype=np.int8)
        p_theta_idx = np.zeros(num_records, dtype=np.int8)

        for i in tqdm(range(num_records), desc="  Slicing Protons"):
            vdf_3d = vdf_p[i]
            if np.any(np.isfinite(vdf_3d) & (vdf_3d > 0)):
                i_theta_max, _, i_phi_max = np.unravel_index(np.argmax(np.nan_to_num(vdf_3d)), vdf_3d.shape)
            else:
                i_theta_max, i_phi_max = 0, 0

            p_phi_idx[i] = int(i_phi_max)
            p_theta_idx[i] = int(i_theta_max)

            vdf_xz_2d = vdf_3d[:, :, i_phi_max]
            with np.errstate(divide='ignore', invalid='ignore'):
                log_vdf_xz = np.log10(vdf_xz_2d)
            p_xz[i] = np.where(vdf_xz_2d > 0, log_vdf_xz, np.where(vdf_xz_2d == 0, 0.0, np.nan)).astype(np.float32)

            vdf_xy_2d = vdf_3d[i_theta_max, :, :].T
            with np.errstate(divide='ignore', invalid='ignore'):
                log_vdf_xy = np.log10(vdf_xy_2d)
            p_xy[i] = np.where(vdf_xy_2d > 0, log_vdf_xy, np.where(vdf_xy_2d == 0, 0.0, np.nan)).astype(np.float32)

        web_dict = {
            'times': np.round(times_all, 3).tolist(),
            'magf_inst': np.nan_to_num(np.round(magf_p, 2), nan=0.0).tolist(),
            'p_xz': np.nan_to_num(np.round(p_xz, 2), nan=0.0).tolist(),
            'p_xy': np.nan_to_num(np.round(p_xy, 2), nan=0.0).tolist(),
            'p_phi_idx': p_phi_idx.tolist(),
            'p_theta_idx': p_theta_idx.tolist(),
            'p_vx_3d': np.round(vx_p[0], 1).tolist(),
            'p_vy_3d': np.round(vy_p[0], 1).tolist(),
            'p_vz_3d': np.round(vz_p[0], 1).tolist(),
        }

        # Process Alphas if sf0a file exists
        file_he = resolve_vdf_file(date_str, '0a')
        if file_he:
            try:
                cdf_he = cdflib.CDF(file_he)
                times_he = cdflib.cdfepoch.unixtime(cdf_he.varget('Epoch'))
                num_rec_he = len(times_he)

                if 'EFLUX' in cdf_he.cdf_info().zVariables:
                    eflux_he_raw = cdf_he.varget('EFLUX')[:num_rec_he]
                else:
                    eflux_he_raw = cdf_he.varget('DATA')[:num_rec_he] * 0.97e8

                theta_he = prepare_4d_grid(theta_p[0], num_rec_he)
                phi_he = prepare_4d_grid(phi_p[0], num_rec_he)
                energy_he = prepare_4d_grid(energy_p[0], num_rec_he)

                mass_he, charge_he = 4.0 * 0.010438870, 2.0

                eflux_he_4d = eflux_he_raw.reshape((num_rec_he, 8, 32, 8))
                eflux_he_clean = np.where((eflux_he_4d < 0) | (~np.isfinite(eflux_he_4d)), np.nan, eflux_he_4d)
                energy_he_clean = np.where((energy_he <= 0) | (~np.isfinite(energy_he)), np.nan, energy_he)

                vdf_he = (eflux_he_clean / energy_he_clean) * (mass_he ** 2) / ((2e-5) * energy_he_clean)
                
                energy_geom_he = np.nan_to_num(energy_he, nan=100.0)
                theta_geom_he = np.nan_to_num(theta_he, nan=0.0)
                phi_geom_he = np.nan_to_num(phi_he, nan=0.0)

                vel_he_geom = np.sqrt(2 * charge_he * np.maximum(energy_geom_he, 1.0) / mass_he)
                rad_th_he, rad_ph_he = np.radians(theta_geom_he), np.radians(phi_geom_he)
                
                vx_he = vel_he_geom * np.cos(rad_ph_he) * np.cos(rad_th_he)
                vy_he = vel_he_geom * np.sin(rad_th_he)
                vz_he = vel_he_geom * np.sin(rad_ph_he) * np.cos(rad_th_he)

                he_xz = np.full((num_records, 8, 32), np.nan, dtype=np.float32)
                he_xy = np.full((num_records, 8, 32), np.nan, dtype=np.float32)
                he_phi_idx = np.zeros(num_records, dtype=np.int8)
                he_theta_idx = np.zeros(num_records, dtype=np.int8)

                for i in tqdm(range(num_rec_he), desc="  Slicing Alphas"):
                    target_idx = np.clip(np.searchsorted(times_all, times_he[i]), 0, num_records - 1)
                    vdf_3d = vdf_he[i]
                    if np.any(np.isfinite(vdf_3d) & (vdf_3d > 0)):
                        i_theta_max, _, i_phi_max = np.unravel_index(np.argmax(np.nan_to_num(vdf_3d)), vdf_3d.shape)
                    else:
                        i_theta_max, i_phi_max = 0, 0

                    he_phi_idx[target_idx] = int(i_phi_max)
                    he_theta_idx[target_idx] = int(i_theta_max)

                    vdf_he_xz = vdf_3d[:, :, i_phi_max]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        log_vdf_he_xz = np.log10(vdf_he_xz)
                    he_xz[target_idx] = np.where(vdf_he_xz > 0, log_vdf_he_xz, np.where(vdf_he_xz == 0, 0.0, np.nan)).astype(np.float32)

                    vdf_he_xy = vdf_3d[i_theta_max, :, :].T
                    with np.errstate(divide='ignore', invalid='ignore'):
                        log_vdf_he_xy = np.log10(vdf_he_xy)
                    he_xy[target_idx] = np.where(vdf_he_xy > 0, log_vdf_he_xy, np.where(vdf_he_xy == 0, 0.0, np.nan)).astype(np.float32)

                web_dict.update({
                    'he_xz': np.nan_to_num(np.round(he_xz, 2), nan=0.0).tolist(),
                    'he_xy': np.nan_to_num(np.round(he_xy, 2), nan=0.0).tolist(),
                    'he_phi_idx': he_phi_idx.tolist(),
                    'he_theta_idx': he_theta_idx.tolist(),
                    'he_vx_3d': np.round(vx_he[0], 1).tolist(),
                    'he_vy_3d': np.round(vy_he[0], 1).tolist(),
                    'he_vz_3d': np.round(vz_he[0], 1).tolist(),
                })

            except Exception as e:
                print(f"  [!] Alpha slicing warning: {e}")

        with gzip.open(out_file, 'wt', encoding='utf-8') as f:
            json.dump(web_dict, f, separators=(',', ':'))

        file_size_mb = os.path.getsize(out_file) / (1024 * 1024)
        print(f"--> Saved Optimized VDF Dataset ({file_size_mb:.2f} MB): {out_file}")
        return out_file

    finally:
        if cdf_p is not None and hasattr(cdf_p, 'close'): cdf_p.close()
        if cdf_he is not None and hasattr(cdf_he, 'close'): cdf_he.close()
        gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PSP SPAN-I VDF JSON Pre-Processor")
    parser.add_argument("--enc", "--encounter", type=int, default=None, help="Encounter number (e.g., 29)")
    parser.add_argument("--start-date", type=str, default=None, help="Start Date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="End Date YYYY-MM-DD")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite existing slice files")
    args = parser.parse_args()

    # 1. Resolve date range from encounter number
    if args.enc is not None:
        if args.enc not in ENCOUNTER_DATES:
            print(f"[!] Error: Encounter {args.enc} is not defined in ENCOUNTER_DATES.")
            sys.exit(1)
        start_str, end_str = ENCOUNTER_DATES[args.enc]
        print(f"[*] Processing Encounter {args.enc:02d} date range: {start_str} to {end_str}")
        t_start = datetime.strptime(start_str, "%Y-%m-%d")
        t_end = datetime.strptime(end_str, "%Y-%m-%d")

    # 2. Resolve date range from explicit date arguments (cron job mode)
    elif args.start_date is not None:
        t_start = datetime.strptime(args.start_date, "%Y-%m-%d")
        t_end = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else t_start

    # 3. Raise error if neither flag was supplied
    else:
        parser.error("You must specify either --enc/--encounter OR --start-date!")

    # Execute daily slicing batch
    curr = t_start
    while curr <= t_end:
        date_str = curr.strftime("%Y-%m-%d")
        try:
            process_day(date_str, overwrite=args.overwrite)
        except Exception as e:
            print(f"[!] Exception encountered while processing {date_str}: {e}")
        curr += timedelta(days=1)
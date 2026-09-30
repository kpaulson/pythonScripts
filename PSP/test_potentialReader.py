import os
import sys
import glob
import numpy as np
import pandas as pd

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import config

try:
    from PSP.psp_spacecraftPotential_tools import (
        load_auth_credentials,
        get_or_fetch_vdc_files,
        load_vdc_potential,
        get_spice_distance_au,
        load_lfr_or_csv_density
    )
except ImportError:
    from psp_spacecraftPotential_tools import (
        load_auth_credentials,
        get_or_fetch_vdc_files,
        load_vdc_potential,
        get_spice_distance_au,
        load_lfr_or_csv_density
    )


def run_toolset_diagnostic(dates_to_test):
    print("=" * 70)
    print("     PARKER SOLAR PROBE TOOLSET DIAGNOSTIC TEST")
    print("=" * 70)

    # 1. Test Auth Loading
    print("\n[1] Checking SSL Authentication Credentials...")
    creds = load_auth_credentials()
    if creds:
        print(f"    [+] Credentials found for user: '{creds[0]}'")
    else:
        print("    [!] WARNING: ~/MyDrive/.auth not found or invalid. Remote wget might fail.")

    DRIVE_ROOT = config.get_drive_path()

    for date_tuple in dates_to_test:
        y, m, d = date_tuple
        date_str = f"{y}-{m}-{d}"
        print(f"\n" + "-" * 70)
        print(f" TESTING DATE: {date_str}")
        print("-" * 70)

        # 2. Test VDC Discovery & Fetcher
        print(f"\n[2] Testing VDC Discovery & Remote Fetcher...")
        vdc_files = get_or_fetch_vdc_files(y, m, d, auth=creds)
        print(f"    [+] Resolved {len(vdc_files)} VDC chunk CDF(s):")
        for f in vdc_files:
            print(f"        -> {os.path.basename(f)}")

        if not vdc_files:
            print(f"    [!] FAILED: No VDC files resolved for {date_str}.")
            continue

        # 3. Test DC Spacecraft Potential Extraction
        print(f"\n[3] Testing DC Spacecraft Potential Extraction...")
        v_unix, v_avg = load_vdc_potential(vdc_files, verbose=False)

        if len(v_unix) > 0:
            valid_v = np.isfinite(v_avg)
            v_clean = v_avg[valid_v]
            print(f"    [+] Total Raw Timestamps Extracted : {len(v_unix):,}")
            print(f"    [+] Valid (Non-NaN) Potentials    : {len(v_clean):,} / {len(v_unix):,} ({len(v_clean)/len(v_unix)*100:.1f}%)")
            if len(v_clean) > 0:
                print(f"    [+] Potential Range (V_avg)       : Min = {np.min(v_clean):.2f} V, Max = {np.max(v_clean):.2f} V, Mean = {np.mean(v_clean):.2f} V")
                sample_times = pd.to_datetime(v_unix[:2], unit='s', utc=True).strftime('%Y-%m-%dT%H:%M:%SZ').tolist()
                print(f"    [+] Sample Timestamps (ISO UTC)   : {sample_times}")

            # 4. Test SPICE Radial Distance Tool
            print(f"\n[4] Testing SPICE Distance Calculation (R_AU)...")
            r_au = get_spice_distance_au(v_unix[:100])
            print(f"    [+] Computed SPICE Distance (R_AU) : Mean = {np.nanmean(r_au):.4f} AU")

        else:
            print("    [!] FAILED: Potential array returned empty.")

        # 5. Check Calibration CSV / Density file availability
        print(f"\n[5] Checking Resonance Calibration CSV Ingestion...")
        cal_dir = os.path.join(DRIVE_ROOT, 'Research/PSP/Hammerheads/resonanceFiles')
        csv_matches = sorted(glob.glob(os.path.join(cal_dir, f"hammerhead_cal_*.csv")))

        if csv_matches:
            latest_csv = csv_matches[-1]
            print(f"    [+] Found Calibration CSV: {os.path.basename(latest_csv)}")
            df = pd.read_csv(latest_csv)
            print(f"    [+] Calibration Rows: {len(df):,}")
            cols_present = [c for c in ['f_weighted_Hz', 'debye_length_m', 'V_sc_volts', 'L_eff_m'] if c in df.columns]
            print(f"    [+] Relevant Columns Present: {cols_present}")
        else:
            print(f"    [-] No existing resonance CSV files found in {cal_dir}")

    print("\n" + "=" * 70)
    print("     DIAGNOSTIC TEST COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    target_dates = [
        ('2022', '09', '05'),
        ('2022', '09', '06')
    ]
    run_toolset_diagnostic(target_dates)
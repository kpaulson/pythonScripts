#!/usr/bin/env python3
"""
psp_flds_build_lfr_mission_cdf.py
=================================
Dynamic LFR Mission Density Compiler.
Extracts density & temperature arrays from multiple file formats,
normalizes timestamps to TT2000, sorts chronologically, and exports
a unified spp_fld_lfr_mission_density.cdf.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.io import readsav
import cdflib

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = '/home/kpaulson/MyDrive/Research/PSP/FIELDS/LFR_Density'
OUTPUT_CDF = os.path.join(DATA_DIR, 'spp_fld_lfr_mission_density.cdf')
BASE_IDL_FILE = os.path.join(DATA_DIR, 'spp_fld_lfr_mission_density.idl')


# =============================================================================
# HELPER PARSERS
# =============================================================================
def datetime_to_tt2000(dt_series):
    """Converts pandas DatetimeIndex/Series/strings into TT2000 int64 nanoseconds using integer component matrices."""
    dts = pd.to_datetime(dt_series, utc=True)
    if not isinstance(dts, pd.DatetimeIndex):
        dts = pd.DatetimeIndex(dts)

    years = dts.year.values
    months = dts.month.values
    days = dts.day.values
    hours = dts.hour.values
    minutes = dts.minute.values
    seconds = dts.second.values
    micros = dts.microsecond.values
    millis = micros // 1000
    micros_rem = micros % 1000
    nanos = getattr(dts, 'nanosecond', np.zeros(len(dts), dtype=int)).values

    matrix = np.column_stack([years, months, days, hours, minutes, seconds, millis, micros_rem, nanos])
    return cdflib.cdfepoch.compute_tt2000(matrix.tolist())


def parse_base_idl(fpath):
    """Extracts ne, te, epoch from the nested base IDL struct."""
    if not os.path.exists(fpath):
        return None, None, None
    try:
        sav = readsav(fpath)
        lfr_key = next((k for k in sav.keys() if k.lower() == 'lfr'), None)
        if lfr_key is None:
            return None, None, None
        
        lfr = sav[lfr_key]
        struct = lfr[0] if lfr.ndim > 0 else lfr

        def get_field(st, field_name):
            k = next((f for f in st.dtype.names if f.lower() == field_name.lower()), None)
            return st[k] if k else None

        ne_s = get_field(struct, 'n_e')
        te_s = get_field(struct, 't_c')
        ep_s = get_field(struct, 'epoch')

        def extract_dat(s):
            if s is None:
                return None
            sub = s[0] if (isinstance(s, np.ndarray) and s.size > 0) else s
            if hasattr(sub, 'dtype') and sub.dtype.names:
                k = next((f for f in sub.dtype.names if f.lower() == 'dat'), None)
                val = sub[k] if k else sub
            else:
                val = sub
            return np.atleast_1d(np.squeeze(val))

        ep_data = extract_dat(ep_s).astype(np.int64)
        ne_data = extract_dat(ne_s).astype(np.float64)
        te_data = extract_dat(te_s).astype(np.float64)

        return ep_data, ne_data, te_data
    except Exception as e:
        print(f"  [!] Error reading base IDL ({os.path.basename(fpath)}): {e}")
        return None, None, None


def parse_idl_sav(fpath):
    """Extracts ne, te, epoch from individual encounter IDL .sav files."""
    try:
        sav = readsav(fpath)
        ne_data = sav.get('denstot')
        epoch_data = sav.get('epochl2')
        if ne_data is None or epoch_data is None:
            return None, None, None

        te_data = sav.get('tctot')
        if te_data is None:
            te_data = sav.get('tcore')
        if te_data is None:
            te_data = np.zeros(len(ne_data), dtype=np.float64)

        return (
            np.atleast_1d(np.squeeze(epoch_data)).astype(np.int64),
            np.atleast_1d(np.squeeze(ne_data)).astype(np.float64),
            np.atleast_1d(np.squeeze(te_data)).astype(np.float64)
        )
    except Exception as e:
        print(f"  [!] Error reading IDL SAV ({os.path.basename(fpath)}): {e}")
        return None, None, None


def parse_csv_file(fpath):
    """Extracts ne and converts ISO timestamps to TT2000 from CSV files."""
    try:
        df = pd.read_csv(fpath)
        if 'Ne' in df.columns and 'Time' in df.columns:
            ne_val = df['Ne'].values
            tt2000_val = datetime_to_tt2000(df['Time'])
            te_val = np.full(len(ne_val), -1.0, dtype=np.float64)
            return tt2000_val, np.array(ne_val, dtype=np.float64), te_val
    except Exception as e:
        print(f"  [!] Error reading CSV ({os.path.basename(fpath)}): {e}")
    return None, None, None


def parse_tplot_sav(fpath):
    """Extracts ne and converts Unix seconds or byte-string dates to TT2000 from TPlot .sav files."""
    try:
        sav = readsav(fpath)
        ne_val = sav.get('nqtn')
        time_data = sav.get('time')
        if ne_val is not None and time_data is not None:
            time_arr = np.squeeze(time_data)
            
            # Handle string/bytes time arrays vs numeric Unix timestamp arrays
            first_elem = time_arr[0] if time_arr.ndim > 0 else time_arr
            if isinstance(first_elem, (bytes, str, np.bytes_)):
                if isinstance(first_elem, (bytes, np.bytes_)):
                    str_times = [t.decode('utf-8').replace('/', ' ') for t in time_arr]
                else:
                    str_times = [t.replace('/', ' ') for t in time_arr]
                dts = pd.to_datetime(str_times, utc=True)
            else:
                dts = pd.to_datetime(np.array(time_arr, dtype=np.float64), unit='s', utc=True)

            tt2000_val = datetime_to_tt2000(dts)
            te_val = np.zeros(len(ne_val), dtype=np.float64)
            return tt2000_val, np.array(ne_val, dtype=np.float64), te_val
    except Exception as e:
        print(f"  [!] Error reading TPlot SAV ({os.path.basename(fpath)}): {e}")
    return None, None, None


def write_mission_cdf(output_fpath, epoch, ne, te):
    """Writes compiled arrays out to a clean CDF file using cdflib.cdfwrite."""
    if os.path.exists(output_fpath):
        os.remove(output_fpath)

    # Construct CDF Writer
    cdf = cdflib.cdfwrite.CDF(output_fpath)

    spec_epoch = {
        'Variable': 'epoch',
        'Data_Type': 33,    # CDF_TIME_TT2000
        'Num_Elements': 1,
        'Rec_Vary': True,   # MUST BE BOOLEAN True
        'Dim_Sizes': [],
    }
    
    spec_ne = {
        'Variable': 'electronDensity',
        'Data_Type': 45,    # CDF_DOUBLE
        'Num_Elements': 1,
        'Rec_Vary': True,   # MUST BE BOOLEAN True
        'Dim_Sizes': [],
    }
    attrs_ne = {
        'NAME': 'electronDensity',
        'LABEL': 'n_e',
        'UNITS': 'cm^-3',
        'TITLE': 'Electron number density derived from FIELDS LFR QTN data product',
        'DEPEND_0': 'epoch'
    }

    spec_te = {
        'Variable': 'electronTemperatureCore',
        'Data_Type': 45,    # CDF_DOUBLE
        'Num_Elements': 1,
        'Rec_Vary': True,   # MUST BE BOOLEAN True
        'Dim_Sizes': [],
    }
    attrs_te = {
        'NAME': 'electronTemperatureCore',
        'LABEL': 'T_e',
        'UNITS': 'K',
        'TITLE': 'Electron Temperature derived from FIELDS LFR QTN data product',
        'DEPEND_0': 'epoch'
    }

    # Write variables and close stream
    cdf.write_var(spec_epoch, var_data=epoch)
    cdf.write_var(spec_ne, var_attrs=attrs_ne, var_data=ne)
    cdf.write_var(spec_te, var_attrs=attrs_te, var_data=te)
    cdf.close()


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    epoch_list, ne_list, te_list = [], [], []

    # 1. Base IDL File
    if os.path.exists(BASE_IDL_FILE):
        print(f"[*] Base File Found: {os.path.basename(BASE_IDL_FILE)}")
        ep, ne, te = parse_base_idl(BASE_IDL_FILE)
        if ep is not None:
            epoch_list.append(ep); ne_list.append(ne); te_list.append(te)

    # 2. Encounter IDL SAV Files
    sav_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_mimo_*.sav")))
    for fpath in sav_files:
        print(f"[*] Ingesting IDL SAV: {os.path.basename(fpath)}")
        ep, ne, te = parse_idl_sav(fpath)
        if ep is not None:
            epoch_list.append(ep); ne_list.append(ne); te_list.append(te)

    # 3. CSV Files
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_QTN_*.csv")))
    for fpath in csv_files:
        print(f"[*] Ingesting CSV: {os.path.basename(fpath)}")
        ep, ne, te = parse_csv_file(fpath)
        if ep is not None:
            epoch_list.append(ep); ne_list.append(ne); te_list.append(te)

    # 4. TPlot SAV Files
    tplot_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_QTN_Density.sav")))
    for fpath in tplot_files:
        print(f"[*] Ingesting TPlot SAV: {os.path.basename(fpath)}")
        ep, ne, te = parse_tplot_sav(fpath)
        if ep is not None:
            epoch_list.append(ep); ne_list.append(ne); te_list.append(te)

    if not ne_list:
        print("[!] No valid input data found. Aborting.")
        return

    # 5. Concatenate, Filter, & Sort Monotonically
    all_epoch = np.concatenate(epoch_list)
    all_ne    = np.concatenate(ne_list)
    all_te    = np.concatenate(te_list)

    valid_mask = np.isfinite(all_ne) & (all_ne > 0) & (all_epoch > 0)
    all_epoch, all_ne, all_te = all_epoch[valid_mask], all_ne[valid_mask], all_te[valid_mask]

    # Chronological sort & duplicate timestamp removal
    sort_idx = np.argsort(all_epoch)
    all_epoch, all_ne, all_te = all_epoch[sort_idx], all_ne[sort_idx], all_te[sort_idx]

    _, unique_idx = np.unique(all_epoch, return_index=True)
    all_epoch, all_ne, all_te = all_epoch[unique_idx], all_ne[unique_idx], all_te[unique_idx]

    print(f"\n=======================================================")
    print(f" Total Consolidated Records : {len(all_epoch)}")
    print(f" Earliest Record            : {cdflib.cdfepoch.to_datetime(all_epoch[0])}")
    print(f" Latest Record              : {cdflib.cdfepoch.to_datetime(all_epoch[-1])}")
    print(f"=======================================================\n")

    # 6. Export CDF
    write_mission_cdf(OUTPUT_CDF, all_epoch, all_ne, all_te)
    print(f"[SUCCESS] Compiled mission CDF written to:\n  -> {OUTPUT_CDF}\n")


if __name__ == "__main__":
    main()
"""
psp_json_exporter.py
====================
Headless Exporter for Modular Parker Solar Probe (PSP) Instrument JSON Payloads.

Purpose
-------
1. Extracts orbital-length instrument data via `psp_swp_browser_dataProcessor`.
2. Formats and sanitizes values (converting NaNs to JSON nulls, pre-scaling log10 matrices).
3. Saves output JSON payloads directly into your Google Drive directory structure:
   `Research/PSP/JSON/{team|pub}/E{enc_num}/`

Note on File Syncing & Staging
------------------------------
This script ONLY generates local JSON files in your team/drive directory. Staging,
remote server copying, and public deployment are handled externally by conductor cron jobs.

Command-Line Usage
------------------
  # Export MAG for Encounter 28 (Team internal directory):
  python psp_json_exporter.py --enc 28 --module mag

  # Export all available instrument modules for Encounter 28:
  python psp_json_exporter.py --enc 28 --module all --overwrite
"""

import os
import sys
import json
import gzip
import argparse
from datetime import datetime, timedelta, timezone

import numpy as np

# --- IMPORT HEADLESS BACKEND DATA PROCESSOR ---
import psp_swp_browser_dataProcessor as data_processor

# =============================================================================
# INSTRUMENT MODULE TARGET CADENCES (Seconds)
# =============================================================================
# Set the exact temporal resolution per instrument for web JSON exports.
MODULE_CADENCES = {
    "mag": 10,           # 10 sec: FIELDS Magnetometer (Br, Bt, Bn, |B|)
    "spc": 60,           #  1 min: SPC L3i Plasma Fit Moments (Vr, Np, Wp)
    "spc_l2": 600,       # 10 min: SPC L2 2D Spectra & Currents (Keeps matrix light)
    "spani": 60,         #  1 min: SPAN-i L3 Ion Moments
    "spane": 300,        #  5 min: SPAN-E L3 Electron Pitch Angle Distributions
    "waves": 60,         #  1 min: Wave Analysis Spectrograms
    "hammerhead": 300,   #  5 min: Hammerhead String Occurrence Counts
    "lfr": 180,          #  3 min: LFR Mission Electron Density
    "mergedsweap": 900   # 15 min: the native cadence of this fileset
}

# Number of log-spaced frequency channels for Wave Analysis (Default: 42)
WAVE_FREQ_BINS = 42

# =============================================================================
# DATA SANITIZATION & FORMATTING HELPERS
# =============================================================================

def sanitize_1d_array(input_array, decimal_precision=2):
    """
    Sanitizes a 1D numerical array for valid JSON serialization.

    Parameters
    ----------
    input_array : array-like or None
        The raw 1D numerical time-series array.
    decimal_precision : int, optional
        Number of decimal places to round values to (default: 2).

    Returns
    -------
    list
        A clean Python list ready for `json.dump()`.
    """
    if input_array is None or len(input_array) == 0:
        return []

    clean_list = []
    for value in input_array:
        if value is None or np.isnan(value) or np.isinf(value):
            clean_list.append(None)
        else:
            clean_list.append(round(float(value), decimal_precision))

    return clean_list


def sanitize_2d_matrix_log10(input_matrix_2d, decimal_precision=2):
    """
    Pre-computes log10 scaling and sanitizes a 2D matrix for JSON export.
    Reduces output file size by ~60% and offloads math from the web browser client.

    Parameters
    ----------
    input_matrix_2d : np.ndarray or None
        2D data matrix, expected shape: (time_bins, channel_bins).
    decimal_precision : int, optional
        Number of decimal places to round values to (default: 2).

    Returns
    -------
    list of lists
        A 2D Python list of pre-logged values with `None` for non-positive or invalid points.
    """
    if input_matrix_2d is None or len(input_matrix_2d) == 0:
        return []

    sanitized_matrix = []
    for row in input_matrix_2d:
        sanitized_row = []
        for element in row:
            if element is None or np.isnan(element) or np.isinf(element) or element <= 0:
                sanitized_row.append(None)
            else:
                log_value = np.log10(float(element))
                sanitized_row.append(round(float(log_value), decimal_precision))
        sanitized_matrix.append(sanitized_row)

    return sanitized_matrix
    
def sanitize_2d_matrix_linear(input_matrix_2d, decimal_precision=2):
    """
    Sanitizes a 2D numerical matrix without log-transforming values.
    Used for linear 2D parameters like Ellipticity, Poynting Angle, and Wave Normal.
    """
    if input_matrix_2d is None or len(input_matrix_2d) == 0:
        return []

    sanitized_matrix = []
    for row in input_matrix_2d:
        sanitized_row = []
        for element in row:
            # Filter out NaNs, Infs, and CDF fill values (-1e31)
            if element is None or np.isnan(element) or np.isinf(element) or element < -1e5:
                sanitized_row.append(None)
            else:
                sanitized_row.append(round(float(element), decimal_precision))
        sanitized_matrix.append(sanitized_row)

    return sanitized_matrix


def build_common_metadata(encounter_number, module_name, orbit_start_dt, orbit_end_dt, encounter_start_dt, encounter_end_dt):
    """Constructs a standardized metadata header inserted into every JSON payload."""
    return {
        "encounter_num": encounter_number,
        "module": module_name,
        "orbit_start": orbit_start_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "orbit_end": orbit_end_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "encounter_start": encounter_start_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "encounter_end": encounter_end_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "generated_utc": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    }


# =============================================================================
# MODULAR INSTRUMENT EXPORTERS
# =============================================================================

def save_json_payload(payload, output_directory, output_filename, overwrite=False):
    """
    Handles overwrite checks, JIT directory creation, and gzipped JSON serialization.
    Returns True if written, False if skipped.
    """
    full_output_path = os.path.join(output_directory, output_filename)

    if not overwrite and os.path.exists(full_output_path):
        print(f"[*] SKIPPED: {output_filename} already exists.")
        return False

    # Force JIT directory verification on network/drive mounts right before write
    os.makedirs(os.path.dirname(full_output_path), exist_ok=True)

    with gzip.open(full_output_path, 'wt', encoding='utf-8') as gz_file:
        json.dump(payload, gz_file)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] SUCCESS: Exported {output_filename}")
    return True

def export_fields_mag_json(encounter_number, orbit_start_dt, orbit_end_dt, encounter_start_dt, encounter_end_dt, output_directory, overwrite=False, verbose=False):
    """Extracts and exports FIELDS MAG 4Sa magnetic field vectors (|B|, Br, Bt, Bn)."""
    output_filename = f"psp_mag_enc_{encounter_number}.json.gz"
    if not overwrite and os.path.exists(os.path.join(output_directory, output_filename)):
        print(f"[*] SKIPPED: {output_filename} already exists.")
        return

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Extracting FIELDS MAG vectors for E{encounter_number}...")
    target_cadence = MODULE_CADENCES["mag"]
    mag_data = data_processor.extract_fields_data(orbit_start_dt, orbit_end_dt, target_cadence_sec=target_cadence, verbose=verbose)

    payload = {
        "metadata": build_common_metadata(encounter_number, "mag", orbit_start_dt, orbit_end_dt, encounter_start_dt, encounter_end_dt),
        "times": {"mag": data_processor.to_plotly_time(mag_data["t_mag_dt"])},
        "data": {
            "b_tot": sanitize_1d_array(mag_data["b_tot"], decimal_precision=2),
            "b_r":   sanitize_1d_array(mag_data["b_r"],   decimal_precision=2),
            "b_t":   sanitize_1d_array(mag_data["b_t"],   decimal_precision=2),
            "b_n":   sanitize_1d_array(mag_data["b_n"],   decimal_precision=2)
        }
    }
    save_json_payload(payload, output_directory, output_filename, overwrite=overwrite)


def export_spc_moments_json(encounter_number, orbit_start_dt, orbit_end_dt, encounter_start_dt, encounter_end_dt, output_directory, overwrite=False, verbose=False):
    """Extracts and exports SPC L3i plasma moments (Vr, Np, Wp)."""
    output_filename = f"psp_spc_enc_{encounter_number}.json.gz"
    full_output_path = os.path.join(output_directory, output_filename)

    if not overwrite and os.path.exists(full_output_path):
        print(f"[*] SKIPPED: {output_filename} already exists.")
        return

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Extracting SPC L3i Moments for E{encounter_number}...")
    target_cadence = MODULE_CADENCES["spc"]
    spc_data = data_processor.extract_spc_l3_data(orbit_start_dt, orbit_end_dt, target_cadence_sec=target_cadence)

    # Calculate V_A using existing exported JSONs in output_directory
    t_valf, valfven = data_processor.calculate_valfven_from_jsons(encounter_number, output_directory)

    payload = {
        "metadata": build_common_metadata(encounter_number, "spc", orbit_start_dt, orbit_end_dt, encounter_start_dt, encounter_end_dt),
        "times": {
            "spc": data_processor.to_plotly_time(spc_data["t_spc_dt"]),
            "valf": t_valf
        },
        "data": {
            "spc_vr_peak": sanitize_1d_array(spc_data["spc_vr_peak"], decimal_precision=1),
            "spc_np_peak": sanitize_1d_array(spc_data["spc_np_peak"], decimal_precision=2),
            "spc_wp_peak": sanitize_1d_array(spc_data["spc_wp_peak"], decimal_precision=1),
            "spc_vr_full": sanitize_1d_array(spc_data["spc_vr_full"], decimal_precision=1),
            "spc_np_full": sanitize_1d_array(spc_data["spc_np_full"], decimal_precision=2),
            "spc_wp_full": sanitize_1d_array(spc_data["spc_wp_full"], decimal_precision=1),
            "valfven":     sanitize_1d_array(valfven,     decimal_precision=1)
        }
    }
    
    save_json_payload(payload, output_directory, output_filename, overwrite=overwrite)


def export_spc_l2_spectra_json(encounter_number, orbit_start_dt, orbit_end_dt, encounter_start_dt, encounter_end_dt, output_directory, overwrite=False, verbose=False):
    """Extracts and exports SPC L2 charge flux density, flow angles, and A/B/C/D collector currents."""
    output_filename = f"psp_spc_l2_enc_{encounter_number}.json.gz"
    full_output_path = os.path.join(output_directory, output_filename)

    if not overwrite and os.path.exists(full_output_path):
        print(f"[*] SKIPPED: {output_filename} already exists.")
        return

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Extracting SPC L2 Spectra for E{encounter_number}...")
    target_cadence = MODULE_CADENCES["spc_l2"]
    spc_l2_data = data_processor.extract_spc_l2_data(orbit_start_dt, orbit_end_dt, target_cadence_sec=target_cadence)

    if spc_l2_data is None:
        print(f"      [!] WARNING: No SPC L2 data found for Encounter {encounter_number}. Skipping export.")
        return

    flux_matrix_transposed = spc_l2_data["flux"].T if spc_l2_data["flux"] is not None else None
    a_current_transposed   = spc_l2_data["a_current"].T if spc_l2_data["a_current"] is not None else None
    b_current_transposed   = spc_l2_data["b_current"].T if spc_l2_data["b_current"] is not None else None
    c_current_transposed   = spc_l2_data["c_current"].T if spc_l2_data["c_current"] is not None else None
    d_current_transposed   = spc_l2_data["d_current"].T if spc_l2_data["d_current"] is not None else None

    payload = {
        "metadata": build_common_metadata(encounter_number, "spc_l2", orbit_start_dt, orbit_end_dt, encounter_start_dt, encounter_end_dt),
        "times": {
            "spc": data_processor.to_plotly_time(spc_l2_data["t_spc_dt"])
        },
        "data": {
            "master_vz":  sanitize_1d_array(spc_l2_data["master_vz"], decimal_precision=1),
            "azimuth":    sanitize_1d_array(spc_l2_data["azimuth"],   decimal_precision=1),
            "elevation":  sanitize_1d_array(spc_l2_data["elevation"], decimal_precision=1),
            "flux":       sanitize_2d_matrix_log10(flux_matrix_transposed, decimal_precision=2),
            "a_current":  sanitize_2d_matrix_log10(a_current_transposed,   decimal_precision=2),
            "b_current":  sanitize_2d_matrix_log10(b_current_transposed,   decimal_precision=2),
            "c_current":  sanitize_2d_matrix_log10(c_current_transposed,   decimal_precision=2),
            "d_current":  sanitize_2d_matrix_log10(d_current_transposed,   decimal_precision=2)
        }
    }
    
    save_json_payload(payload, output_directory, output_filename, overwrite=overwrite)


def export_spani_moments_json(encounter_number, orbit_start_dt, orbit_end_dt, encounter_start_dt, encounter_end_dt, output_directory, overwrite=False, verbose=False):
    """Extracts and exports SPAN-i L3 ion moments (Vr, Np, Wp, Vel_Inst, Wpara, Wperp)."""
    output_filename = f"psp_spani_enc_{encounter_number}.json.gz"
    full_output_path = os.path.join(output_directory, output_filename)

    if not overwrite and os.path.exists(full_output_path):
        print(f"[*] SKIPPED: {output_filename} already exists.")
        return

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Extracting SPAN-i L3 Moments for E{encounter_number}...")
    target_cadence = MODULE_CADENCES["spani"]
    spani_data = data_processor.extract_spani_data(orbit_start_dt, orbit_end_dt, target_cadence_sec=target_cadence)

    if spani_data["t_spi_dt"] is None:
        print(f"      [!] WARNING: No SPAN-i L3 data found for Encounter {encounter_number}. Skipping export.")
        return

    payload = {
        "metadata": build_common_metadata(encounter_number, "spani", orbit_start_dt, orbit_end_dt, encounter_start_dt, encounter_end_dt),
        "times": {
            "spi": data_processor.to_plotly_time(spani_data["t_spi_dt"])
        },
        "data": {
            "spi_vr":       sanitize_1d_array(spani_data["spi_vr"], decimal_precision=1),
            "spi_np":       sanitize_1d_array(spani_data["spi_np"], decimal_precision=2),
            "spi_wp":       sanitize_1d_array(spani_data["spi_wp"], decimal_precision=1),
            "spi_vel_inst": sanitize_2d_matrix_linear(spani_data.get("spi_vel_inst"), decimal_precision=1),
            "spi_wpara":    sanitize_1d_array(spani_data.get("spi_wpara"), decimal_precision=1),
            "spi_wperp":    sanitize_1d_array(spani_data.get("spi_wperp"), decimal_precision=1)
        }
    }
    
    save_json_payload(payload, output_directory, output_filename, overwrite=overwrite)


def export_spane_pad_json(encounter_number, orbit_start_dt, orbit_end_dt, encounter_start_dt, encounter_end_dt, output_directory, overwrite=False, verbose=False):
    """Extracts and exports SPAN-E L3 electron Pitch Angle Distribution (PAD) heatmaps."""
    output_filename = f"psp_spane_enc_{encounter_number}.json.gz"
    if not overwrite and os.path.exists(os.path.join(output_directory, output_filename)):
        print(f"[*] SKIPPED: {output_filename} already exists.")
        return

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Extracting SPAN-E L3 PAD Matrix for E{encounter_number}...")
    target_cadence = MODULE_CADENCES["spane"]

    # Updated: Pass target_energy_ev=476.0 instead of legacy span_energy_idx
    spane_data = data_processor.extract_spane_data(
        orbit_start_dt, orbit_end_dt, target_energy_ev=476.0, target_cadence_sec=target_cadence, verbose=verbose
    )

    pad_matrix = spane_data["pad_dec"]
    pitch_angle_bins = np.linspace(0, 180, pad_matrix.shape[1]) if pad_matrix is not None else []
    pad_matrix_transposed = pad_matrix.T if pad_matrix is not None else None

    metadata = build_common_metadata(encounter_number, "spane", orbit_start_dt, orbit_end_dt, encounter_start_dt, encounter_end_dt)
    metadata["span_energy_ev"] = spane_data.get("span_energy_ev", 476.0)

    payload = {
        "metadata": metadata,
        "times": {"span_e": data_processor.to_plotly_time(spane_data["t_span_e_dt"])},
        "data": {
            "pa_bins":  sanitize_1d_array(pitch_angle_bins, decimal_precision=1),
            "span_pad": sanitize_2d_matrix_log10(pad_matrix_transposed, decimal_precision=2)
        }
    }
    save_json_payload(payload, output_directory, output_filename, overwrite=overwrite)


def export_hammerhead_counts_json(encounter_number, orbit_start_dt, orbit_end_dt, encounter_start_dt, encounter_end_dt, output_directory, overwrite=False, verbose=False):
    """Extracts and exports Hammerhead String occurrence counts binned over uniform 5-minute intervals."""
    output_filename = f"psp_hammerhead_enc_{encounter_number}.json.gz"
    full_output_path = os.path.join(output_directory, output_filename)

    if not overwrite and os.path.exists(full_output_path):
        print(f"[*] SKIPPED: {output_filename} already exists.")
        return

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Extracting Hammerhead String Counts for E{encounter_number}...")
    target_cadence = MODULE_CADENCES["hammerhead"]
    # Convert cadence to minutes for ham_bin parameter
    bin_minutes = int(target_cadence / 60)
    hammerhead_data = data_processor.extract_hammerhead_data(orbit_start_dt, orbit_end_dt, ham_bin=bin_minutes, target_cadence_sec=target_cadence)

    payload = {
        "metadata": build_common_metadata(encounter_number, "hammerhead", orbit_start_dt, orbit_end_dt, encounter_start_dt, encounter_end_dt),
        "times": {
            "ham": data_processor.to_plotly_time(hammerhead_data["t_ham_dt"])
        },
        "data": {
            "ham_counts": sanitize_1d_array(hammerhead_data["ham_counts"], decimal_precision=1)
        }
    }
    
    save_json_payload(payload, output_directory, output_filename, overwrite=overwrite)
    
def export_wave_analysis_json(encounter_number, orbit_start_dt, orbit_end_dt, encounter_start_dt, encounter_end_dt, output_directory, overwrite=False, verbose=False):
    """
    Extracts and exports Wave Analysis spectrograms and polarization parameters.
    Target cadence: 120-300 seconds.
    """
    output_filename = f"psp_waves_enc_{encounter_number}.json.gz"
    full_output_path = os.path.join(output_directory, output_filename)

    if not overwrite and os.path.exists(full_output_path):
        print(f"[*] SKIPPED: {output_filename} already exists.")
        return
        
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Extracting Wave Analysis Data for E{encounter_number}...")
    target_cadence = MODULE_CADENCES.get("waves", 300)

    wave_data = data_processor.extract_wave_data(
        orbit_start_dt, 
        orbit_end_dt, 
        target_cadence_sec=target_cadence, 
        target_freq_bins=WAVE_FREQ_BINS,
        verbose=verbose
    )

    t_fft_raw = wave_data["t_fft_raw"]
    raw_data_dict = wave_data["data"]
    frequencies = wave_data["freqs"]

    if t_fft_raw is None or raw_data_dict is None:
        print(f"      [!] WARNING: No Wave Analysis CDF data found for Encounter {encounter_number}. Skipping export.")
        return

    # Direct variable lookup matching v1.4 CDF variable names
    wave_power_raw  = raw_data_dict.get('B_power_perp')
    poynting_raw    = raw_data_dict.get('S_theta')
    ellipticity_raw = raw_data_dict.get('ellipticity')
    coherency_raw   = raw_data_dict.get('coherency')
    wavenormal_raw  = raw_data_dict.get('wave_normal')

    # Downsample 2D matrices to target temporal cadence
    t_fft_dt, wave_power_dec = data_processor.downsample_time_binned(t_fft_raw, wave_power_raw, target_cadence_sec=target_cadence, is_2d=True)
    _, poynting_dec          = data_processor.downsample_time_binned(t_fft_raw, poynting_raw,   target_cadence_sec=target_cadence, is_2d=True)
    _, ellipticity_dec       = data_processor.downsample_time_binned(t_fft_raw, ellipticity_raw, target_cadence_sec=target_cadence, is_2d=True)
    _, coherency_dec         = data_processor.downsample_time_binned(t_fft_raw, coherency_raw,   target_cadence_sec=target_cadence, is_2d=True)
    _, wavenormal_dec        = data_processor.downsample_time_binned(t_fft_raw, wavenormal_raw,  target_cadence_sec=target_cadence, is_2d=True)

    power_transposed      = wave_power_dec.T if wave_power_dec is not None else None
    poynting_transposed   = poynting_dec.T   if poynting_dec is not None else None
    ellipticity_transposed= ellipticity_dec.T if ellipticity_dec is not None else None
    coherency_transposed  = coherency_dec.T  if coherency_dec is not None else None
    wavenormal_transposed = wavenormal_dec.T if wavenormal_dec is not None else None

    payload = {
        "metadata": build_common_metadata(encounter_number, "waves", orbit_start_dt, orbit_end_dt, encounter_start_dt, encounter_end_dt),
        "times": {
            "wave": data_processor.to_plotly_time(t_fft_dt)
        },
        "data": {
            "freqs":        sanitize_1d_array(frequencies, decimal_precision=2),
            "wave_power":   sanitize_2d_matrix_log10(power_transposed, decimal_precision=2),
            "poynting_deg": sanitize_2d_matrix_linear(poynting_transposed, decimal_precision=1),
            "ellipticity":  sanitize_2d_matrix_linear(ellipticity_transposed, decimal_precision=2),
            "coherency":    sanitize_2d_matrix_linear(coherency_transposed, decimal_precision=2),
            "wave_normal":  sanitize_2d_matrix_linear(wavenormal_transposed, decimal_precision=1)
        }
    }
    
    save_json_payload(payload, output_directory, output_filename, overwrite=overwrite)
    
def export_lfr_density_json(encounter_number, orbit_start_dt, orbit_end_dt, encounter_start_dt, encounter_end_dt, output_directory, overwrite=False, verbose=False):
    """Extracts and exports LFR mission electron density time series."""
    output_filename = f"psp_lfr_enc_{encounter_number}.json.gz"
    full_output_path = os.path.join(output_directory, output_filename)

    if not overwrite and os.path.exists(full_output_path):
        print(f"[*] SKIPPED: {output_filename} already exists.")
        return

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Extracting LFR Electron Density for E{encounter_number}...")
    target_cadence = MODULE_CADENCES["lfr"]
    lfr_data = data_processor.extract_lfr_data(orbit_start_dt, orbit_end_dt, target_cadence_sec=target_cadence)

    payload = {
        "metadata": build_common_metadata(encounter_number, "lfr", orbit_start_dt, orbit_end_dt, encounter_start_dt, encounter_end_dt),
        "times": {
            "lfr": data_processor.to_plotly_time(lfr_data["t_lfr_dt"])
        },
        "data": {
            "np_lfr": sanitize_1d_array(lfr_data["np_lfr"], decimal_precision=2)
        }
    }
    
    save_json_payload(payload, output_directory, output_filename, overwrite=overwrite)

def export_mergedSWEAP_moments_json(encounter_number, orbit_start_dt, orbit_end_dt, encounter_start_dt, encounter_end_dt, output_directory, overwrite=False, verbose=False):
    """Extracts and exports 15-minute merged SWEAP plasma & B-field moments."""
    output_filename = f"psp_mergedSWEAP_enc_{encounter_number}.json.gz"
    full_output_path = os.path.join(output_directory, output_filename)

    if not overwrite and os.path.exists(full_output_path):
        print(f"[*] SKIPPED: {output_filename} already exists.")
        return

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Extracting Merged SWEAP Moments for E{encounter_number}...")
    merged_data = data_processor.extract_merged_sweap_data(orbit_start_dt, orbit_end_dt, encounter_number)

    if merged_data is None:
        print(f"      [!] WARNING: No Merged SWEAP CSV data found for Encounter {encounter_number}. Skipping export.")
        return

    payload = {
        "metadata": build_common_metadata(encounter_number, "mergedSWEAP", orbit_start_dt, orbit_end_dt, encounter_start_dt, encounter_end_dt),
        "times": {
            "merged": data_processor.to_plotly_time(merged_data["t_dt"]),
            "lfr": data_processor.to_plotly_time(merged_data["t_lfr_dt"])
        },
        "data": {
            "b_tot":    sanitize_1d_array(merged_data["b_tot"],    decimal_precision=2),
            "b_r":      sanitize_1d_array(merged_data["b_r"],      decimal_precision=2),
            "b_t":      sanitize_1d_array(merged_data["b_t"],      decimal_precision=2),
            "b_n":      sanitize_1d_array(merged_data["b_n"],      decimal_precision=2),
            "vr":       sanitize_1d_array(merged_data["vr"],       decimal_precision=1),
            "v_alfven": sanitize_1d_array(merged_data["v_alfven"], decimal_precision=1),
            "np":       sanitize_1d_array(merged_data["np"],       decimal_precision=2),
            "tp":       sanitize_1d_array(merged_data["tp"],       decimal_precision=1),
            "np_lfr":   sanitize_1d_array(merged_data["np_lfr"],   decimal_precision=2),
            "x_carr":   sanitize_1d_array(merged_data["x_carr"],   decimal_precision=2),
            "y_carr":   sanitize_1d_array(merged_data["y_carr"],   decimal_precision=2)
        }
    }

    save_json_payload(payload, output_directory, output_filename, overwrite=overwrite)


# =============================================================================
# MAIN CONTROLLER & CLI ROUTER
# =============================================================================

def parse_command_line_arguments():
    """Parses command-line arguments for the JSON exporter."""
    parser = argparse.ArgumentParser(description="Export modular PSP instrument JSON files for web consumption.")
    
    parser.add_argument("--enc", type=int, required=True, help="Encounter number (e.g., 28)")
    
    MODULE_CHOICES = ["mag", "spc", "spc_l2", "spani", "spane", "waves", "hammerhead", "lfr", "mergedSWEAP", "all"]
    parser.add_argument("--module", required=True, choices=MODULE_CHOICES, help="Select instrument module to export")
    
    parser.add_argument("--access", default="team", choices=["team", "pub"], help="Access tier pathing ('team' vs 'pub')")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite existing JSON files")
    
    parser.add_argument("--test", action="store_true", help="Run fast test mode over a 6-hour window")
    parser.add_argument("--test-hours", type=int, default=6, help="Duration for test window in hours (default: 6)")
    
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose stage diagnostics during extraction")
    
    return parser.parse_args()


def main():
    arguments = parse_command_line_arguments()
    encounter_number = arguments.enc
    selected_module  = arguments.module
    overwrite_flag   = arguments.overwrite
    verbose_flag     = arguments.verbose

    # Configure environment runtime paths
    data_processor.resolve_runtime_paths(arguments.access, verbose_flag=False)

    if encounter_number not in data_processor.ENCOUNTER_DATES:
        print(f"[!] Error: Encounter {encounter_number} is not defined in config.py")
        sys.exit(1)

    # 1. Resolve Full Orbit Datetime Window (e.g., E28: April 25 to July 22, 2026)
    orbit_start_string, orbit_end_string = data_processor.config.ORBIT_DATES.get(
        encounter_number, data_processor.ENCOUNTER_DATES[encounter_number]
    )
    orbit_start_datetime = datetime.strptime(orbit_start_string, "%Y-%m-%d")
    orbit_end_datetime   = datetime.strptime(orbit_end_string, "%Y-%m-%d") + timedelta(days=1)

    # 2. Resolve Core Encounter Datetime Window (e.g., E28: June 04 to June 13, 2026)
    encounter_start_string, encounter_end_string = data_processor.ENCOUNTER_DATES[encounter_number]
    encounter_start_datetime = datetime.strptime(encounter_start_string, "%Y-%m-%d")
    encounter_end_datetime   = datetime.strptime(encounter_end_string, "%Y-%m-%d") + timedelta(days=1)

    if arguments.test:
        hours = arguments.test_hours
        print(f"  *** FAST TEST MODE ACTIVE: Restricting window to {hours} hours ***")
        orbit_start_datetime = encounter_start_datetime + timedelta(hours=-(24*29))
        orbit_end_datetime   = orbit_start_datetime + timedelta(hours=hours)
        encounter_start_datetime, encounter_end_datetime = orbit_start_datetime, orbit_end_datetime

    # 3. Establish Target Output Directory (Defaults to Research/PSP/JSON/team/E{enc_num})
    output_directory = os.path.join(
        data_processor.DRIVE_ROOT, f"Research/PSP/JSON/{arguments.access}/E{encounter_number}"
    )
    os.makedirs(output_directory, exist_ok=True)

    print(f"\n============================================================")
    print(f"  PSP JSON EXPORTER | Encounter {encounter_number} | Module: {selected_module.upper()}")
    print(f"  Orbit Bounds: {orbit_start_string} to {orbit_end_string}")
    print(f"  Target Dir:   {output_directory}")
    print(f"============================================================\n")

    # 4. Dispatch Exporters
    if selected_module == "all":
        export_fields_mag_json(encounter_number, orbit_start_datetime, orbit_end_datetime, encounter_start_datetime, encounter_end_datetime, output_directory, overwrite=overwrite_flag, verbose=verbose_flag)
        export_spc_moments_json(encounter_number, orbit_start_datetime, orbit_end_datetime, encounter_start_datetime, encounter_end_datetime, output_directory, overwrite=overwrite_flag, verbose=verbose_flag)
        export_spc_l2_spectra_json(encounter_number, orbit_start_datetime, orbit_end_datetime, encounter_start_datetime, encounter_end_datetime, output_directory, overwrite=overwrite_flag, verbose=verbose_flag)
        export_spani_moments_json(encounter_number, orbit_start_datetime, orbit_end_datetime, encounter_start_datetime, encounter_end_datetime, output_directory, overwrite=overwrite_flag, verbose=verbose_flag)
        export_spane_pad_json(encounter_number, orbit_start_datetime, orbit_end_datetime, encounter_start_datetime, encounter_end_datetime, output_directory, overwrite=overwrite_flag, verbose=verbose_flag)
        export_wave_analysis_json(encounter_number, orbit_start_datetime, orbit_end_datetime, encounter_start_datetime, encounter_end_datetime, output_directory, overwrite=overwrite_flag, verbose=verbose_flag)
        export_hammerhead_counts_json(encounter_number, orbit_start_datetime, orbit_end_datetime, encounter_start_datetime, encounter_end_datetime, output_directory, overwrite=overwrite_flag, verbose=verbose_flag)
        export_lfr_density_json(encounter_number, orbit_start_datetime, orbit_end_datetime, encounter_start_datetime, encounter_end_datetime, output_directory, overwrite=overwrite_flag, verbose=verbose_flag)
    
    elif selected_module == "mag":
        export_fields_mag_json(encounter_number, orbit_start_datetime, orbit_end_datetime, encounter_start_datetime, encounter_end_datetime, output_directory, overwrite=overwrite_flag, verbose=verbose_flag)
    elif selected_module == "spc":
        export_spc_moments_json(encounter_number, orbit_start_datetime, orbit_end_datetime, encounter_start_datetime, encounter_end_datetime, output_directory, overwrite=overwrite_flag, verbose=verbose_flag)
    elif selected_module == "spc_l2":
        export_spc_l2_spectra_json(encounter_number, orbit_start_datetime, orbit_end_datetime, encounter_start_datetime, encounter_end_datetime, output_directory, overwrite=overwrite_flag, verbose=verbose_flag)
    elif selected_module == "spani":
        export_spani_moments_json(encounter_number, orbit_start_datetime, orbit_end_datetime, encounter_start_datetime, encounter_end_datetime, output_directory, overwrite=overwrite_flag, verbose=verbose_flag)
    elif selected_module == "spane":
        export_spane_pad_json(encounter_number, orbit_start_datetime, orbit_end_datetime, encounter_start_datetime, encounter_end_datetime, output_directory, overwrite=overwrite_flag, verbose=verbose_flag)
    elif selected_module == "waves":
        export_wave_analysis_json(encounter_number, orbit_start_datetime, orbit_end_datetime, encounter_start_datetime, encounter_end_datetime, output_directory, overwrite=overwrite_flag, verbose=verbose_flag)
    elif selected_module == "hammerhead":
        export_hammerhead_counts_json(encounter_number, orbit_start_datetime, orbit_end_datetime, encounter_start_datetime, encounter_end_datetime, output_directory, overwrite=overwrite_flag, verbose=verbose_flag)
    elif selected_module == "lfr":
        export_lfr_density_json(encounter_number, orbit_start_datetime, orbit_end_datetime, encounter_start_datetime, encounter_end_datetime, output_directory, overwrite=overwrite_flag, verbose=verbose_flag)
    elif selected_module.lower() in ["mergedsweap", "merged_sweap"]:
        export_mergedSWEAP_moments_json(encounter_number, orbit_start_datetime, orbit_end_datetime, encounter_start_datetime, encounter_end_datetime, output_directory, overwrite=overwrite_flag, verbose=verbose_flag)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] ALL EXPORTS COMPLETE.\n")


if __name__ == "__main__":
    main()
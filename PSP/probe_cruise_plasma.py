import os
import sys
import glob
import cdflib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse

# --- PATH RESOLUTION ---
SCRIPT_DIR = "/home/kpaulson/MyDrive/Software/Python/githubScripts_python" if os.name != 'nt' else "G:/My Drive/Software/Python/githubScripts_python"
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

import config

SWEAP_DIR = config.get_sweapCacheData()
FIELDS_DIR = config.get_berkeleyCacheData()

SPC_L3_ROOT = os.path.join(SWEAP_DIR, 'sweap/spc/L3/')
SPI_L3_ROOT = os.path.join(SWEAP_DIR, 'sweap/spi/L3/spi_sf00/')
SPE_L3_ROOT = os.path.join(SWEAP_DIR, 'sweap/spe/L3/spe_sf0_pad/')
MAG_4SA_ROOT = os.path.join(FIELDS_DIR, 'fields/l2/mag_RTN_4_Sa_per_Cyc/')
LFR_DENSITY_FILE = os.path.join(SWEAP_DIR, 'sweap/spc/LFR/spp_fld_lfr_mission_density.cdf')

# Physical Constants for SPANi Wp
K_B_ERG = 1.3807e-16
M_P_G = 1.6726e-24
EV_TO_K = 11604.525
CM_TO_KM = 100000.0

def cdf_time_to_dt(val):
    """Converts CDF timestamps directly to naive Python datetimes."""
    if val is None or len(val) == 0:
        return []
    val = np.array(val, dtype=np.float64)
    valid = (val > 0) & np.isfinite(val)
    if not np.any(valid):
        return []
    sample = np.nanmedian(val[valid])
    if sample > 1e17:     # TT2000
        unix_ts = (val[valid] / 1e9) + 946727935.816
    elif sample > 1e14:   # us2000
        unix_ts = (val[valid] / 1e6) + 946684800.0
    else:                 # CDF_EPOCH
        unix_ts = (val[valid] - 62167219200000.0) / 1000.0
    return [datetime.fromtimestamp(ts, tz=None) for ts in unix_ts]

def run_probe(start_str, end_str, span_energy_idx=10):
    start_dt = datetime.strptime(start_str, '%Y-%m-%d')
    end_dt = datetime.strptime(end_str, '%Y-%m-%d')

    # Raw Accumulators
    t_mag, b_tot = [], []
    t_spc_peak, spc_vr_peak, spc_np_peak = [], [], []
    t_spc_full, spc_vr_full, spc_np_full = [], [], []
    t_spi, spi_vr, spi_np, spi_wp = [], [], [], []
    t_lfr, np_lfr = [], []
    t_spe, spe_eflux = [], []

    curr_dt = start_dt
    print(f"[*] Extracting raw un-decimated records: {start_str} to {end_str}...")

    while curr_dt <= end_dt:
        y, m, d = curr_dt.strftime('%Y'), curr_dt.strftime('%m'), curr_dt.strftime('%d')

        # 1. MAG 4Sa
        mag_files = glob.glob(os.path.join(MAG_4SA_ROOT, y, m, f"*mag_RTN_4_Sa_per_Cyc_{y}{m}{d}_v*.cdf"))
        if mag_files:
            try:
                with cdflib.CDF(sorted(mag_files)[-1]) as cdf:
                    t_chunk = cdf_time_to_dt(cdf.varget('epoch_mag_RTN_4_Sa_per_Cyc'))
                    data = cdf.varget('psp_fld_l2_mag_RTN_4_Sa_per_Cyc')
                    data[np.abs(data) > 1e10] = np.nan
                    if len(t_chunk) == len(data):
                        t_mag.extend(t_chunk)
                        b_tot.extend(np.linalg.norm(data, axis=1))
            except Exception as e: print(f"MAG Err {curr_dt}: {e}")

        # 2. SPC L3i (Differentiating Peak Tracks vs Full Scans via DQF[:,16])
        spc_files = glob.glob(os.path.join(SPC_L3_ROOT, y, m, f"*swp_spc_l3i_{y}{m}{d}_v*.cdf"))
        if spc_files:
            try:
                with cdflib.CDF(sorted(spc_files)[-1]) as cdf:
                    t_chunk = cdf_time_to_dt(cdf.varget('Epoch'))
                    dqf = cdf.varget('DQF')
                    vr_raw = cdf.varget('vp_fit_RTN')[:, 0]
                    np_raw = cdf.varget('np_fit')

                    if dqf.ndim > 1:
                        mask_gen = (dqf[:, 0] == 0)
                        mask_full = (dqf[:, 16] == 1) if dqf.shape[1] > 16 else np.zeros(len(t_chunk), dtype=bool)
                        mask_peak = mask_gen & (~mask_full)
                    else:
                        mask_peak = np.ones(len(t_chunk), dtype=bool)
                        mask_full = np.zeros(len(t_chunk), dtype=bool)

                    # Peak Track Points
                    m_p = mask_peak & np.isfinite(vr_raw) & (vr_raw > 50) & (vr_raw < 2500)
                    t_spc_peak.extend([t_chunk[i] for i in range(len(m_p)) if m_p[i]])
                    spc_vr_peak.extend(vr_raw[m_p])
                    spc_np_peak.extend(np_raw[m_p])

                    # Full Scan Points
                    m_f = mask_full & np.isfinite(vr_raw) & (vr_raw > 50) & (vr_raw < 2500)
                    t_spc_full.extend([t_chunk[i] for i in range(len(m_f)) if m_f[i]])
                    spc_vr_full.extend(vr_raw[m_f])
                    spc_np_full.extend(np_raw[m_f])

            except Exception as e: print(f"SPC Err {curr_dt}: {e}")

        # 3. SPANi L3 (STRICTLY sf00 PROTON MOMENTS ONLY)
        spi_files = glob.glob(os.path.join(SPI_L3_ROOT, y, m, f"*swp_spi_sf00_[lL]3_mom_{y}{m}{d}_v*.cdf"))
        if spi_files:
            try:
                with cdflib.CDF(sorted(spi_files)[-1]) as cdf:
                    t_chunk = cdf_time_to_dt(cdf.varget('Epoch'))
                    vr_raw = cdf.varget('VEL_RTN_SUN')[:, 0]
                    dens_raw = cdf.varget('DENS')
                    temp_ev = cdf.varget('TEMP')
                    temp_ev[temp_ev <= 0] = np.nan
                    wp_raw = np.sqrt((temp_ev * EV_TO_K * K_B_ERG) / M_P_G) / CM_TO_KM

                    m_spi = np.isfinite(vr_raw) & (vr_raw > 50) & (vr_raw < 2500)
                    if len(t_chunk) == len(vr_raw):
                        t_spi.extend([t_chunk[i] for i in range(len(m_spi)) if m_spi[i]])
                        spi_vr.extend(vr_raw[m_spi])
                        spi_np.extend(dens_raw[m_spi])
                        spi_wp.extend(wp_raw[m_spi])
            except Exception as e: print(f"SPANi Err {curr_dt}: {e}")

        # 4. SPANe PAD
        spe_files = glob.glob(os.path.join(SPE_L3_ROOT, y, m, f"*swp_spe_sf0_[lL]3_pad_{y}{m}{d}_v*.cdf"))
        if spe_files:
            try:
                with cdflib.CDF(sorted(spe_files)[-1]) as cdf:
                    t_chunk = cdf_time_to_dt(cdf.varget('Epoch'))
                    pad = cdf.varget('EFLUX_VS_PA_E')
                    if len(t_chunk) == pad.shape[0]:
                        t_spe.extend(t_chunk)
                        spe_eflux.extend(np.nanmean(pad[:, :, span_energy_idx], axis=1))
            except Exception as e: print(f"SPANe Err {curr_dt}: {e}")

        curr_dt += timedelta(days=1)

    # 5. LFR Density
    if os.path.exists(LFR_DENSITY_FILE):
        try:
            with cdflib.CDF(LFR_DENSITY_FILE) as cdf:
                epochs = cdf_time_to_dt(cdf.varget('epoch'))
                dens = cdf.varget('electronDensity')
                m_lfr = np.array([start_dt <= t <= end_dt for t in epochs]) & np.isfinite(dens) & (dens > 0.01)
                t_lfr = [epochs[i] for i in range(len(m_lfr)) if m_lfr[i]]
                np_lfr = dens[m_lfr]
        except Exception as e: print(f"LFR Err: {e}")

    # --- RENDER MATPLOTLIB DIAGNOSTIC ---
    print("[*] Launching matplotlib visual diagnostic window...")
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Raw Plasma Moment Audit ({start_str} to {end_str})", fontsize=14, fontweight='bold')

    # Panel 1: B-Field Total
    if t_mag:
        axes[0].plot(t_mag, b_tot, 'k.', markersize=1, label='|B| (MAG 4Sa)')
    axes[0].set_ylabel("|B| (nT)")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Radial Velocity
    if t_spc_peak: axes[1].plot(t_spc_peak, spc_vr_peak, 'k.', markersize=2, label='SPC Peak Tracks')
    if t_spc_full: axes[1].plot(t_spc_full, spc_vr_full, 'r.', markersize=2, label='SPC Full Scans')
    if t_spi:      axes[1].plot(t_spi, spi_vr, 'c.', markersize=1.5, label='SPANi sf00')
    axes[1].set_ylabel("Vr (km/s)")
    axes[1].set_ylim(200, 1000)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Density
    if t_lfr:      axes[2].plot(t_lfr, np_lfr, 'g.', markersize=2, label='LFR Ne')
    if t_spc_peak: axes[2].plot(t_spc_peak, spc_np_peak, 'k.', markersize=1.5, label='SPC Peak Np')
    if t_spc_full: axes[2].plot(t_spc_full, spc_np_full, 'r.', markersize=1.5, label='SPC Full Scan Np')
    if t_spi:      axes[2].plot(t_spi, spi_np, 'c.', markersize=1, label='SPANi Np')
    axes[2].set_yscale('log')
    axes[2].set_ylabel("Density (cm^-3)")
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    # Panel 4: SPANe PAD Energy Flux
    if t_spe:
        axes[3].plot(t_spe, spe_eflux, 'm.', markersize=1.5, label='SPANe PA-E Flux (476 eV)')
    axes[3].set_yscale('log')
    axes[3].set_ylabel("Eflux")
    axes[3].set_xlabel("UTC Time")
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick raw plasma probe.")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    run_probe(args.start, args.end)
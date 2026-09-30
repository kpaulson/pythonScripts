#!/usr/bin/env python3
"""
spc_syntheticPacketGenerator.py
--------------------------------
Telemetry generator for SPC/HSFC instrument emulators.
Supports DSCOVR RTSW text ingestion OR built-in canonical test scenarios:
  - 'dscovr': Real DSCOVR RTSW text file ingestion (with vector flow directions)
  - 'shock' : Extended Slow Wind -> IP Shock Compression -> Fast CME Driver Wind + Alphas
  - 'stress': Hyper-compressed 14-second DPU stress test exercising all gain stages (Gain 3 -> 2 -> 1 -> 0).
  - 'matrix': Open-ended stationary regime matrix with configurable block duration (-bd), stochastic thermal jitter, and AR(1) red noise.

Generates binary CCSDS telemetry (.dat) and raw 112-bit slot dumping (.bin)
for firmware ROM images.
"""

import argparse
import datetime
import math
import numpy as np
import os
import random
import struct
import sys
from bisect import bisect_right

PSP_MET_EPOCH = datetime.datetime(2010, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
DEFAULT_OUTPUT_ROOT = "/home/kpaulson/sharedDrive_HelioSwarm/KP_Working/syntheticData/"

# =========================================================================
# INSTRUMENT CONFIGURATION BLOCK
# =========================================================================
PEAK_OFF_LO = 6                  # Bins below peak center (6 low + 1 center + 7 high = 14)
PEAK_OFF_HI = 7                  # Bins above peak center

RETRACE_COUNT = 1                # Number of retrace slots at start of sweep
SLEW_COUNT = 1                   # Number of HV slew slots at end of sweep (unrecorded time advance)

IT_VAL = 6                       # Integration Time (wave periods)
ST_VAL = 2                       # Service Time (wave periods)
WAVE_PERIODS_PER_SLOT = IT_VAL + ST_VAL  # Default: 8 wave periods per measurement slot

DOUBLEWIDTH_INDEX = 0            # Energy table index below which window width is doubled

SLOTS_PER_NYS = 128              # Nominal 1-second NYS interval capacity (1024 ticks / 8 ticks/slot)
FULL_SCAN_INTERVAL = 65536       # Number of sweeps between forced full energy scans
# =========================================================================

ENERGY_TABLES = {
    '12%': [
        150.0, 168.0, 188.16, 210.74, 236.03, 264.35, 296.07, 331.6,
        371.39, 415.96, 465.88, 521.78, 584.4, 654.52, 733.07, 821.03,
        919.56, 1029.91, 1153.49, 1291.91, 1446.94, 1620.58, 1815.05, 2032.85,
        2276.79, 2550.01, 2856.01, 3198.73, 3582.58, 4012.49, 4493.99
    ],
    '16%': [
        150.0, 174.0, 201.84, 234.13, 271.6, 315.05, 365.46, 423.93,
        491.76, 570.44, 661.72, 767.59, 890.4, 1032.87, 1198.13, 1389.83,
        1612.20, 1870.15, 2169.38, 2516.48, 2919.11, 3386.17, 3853.23, 4320.29
    ],
    '20%': [
        150.0, 180.0, 216.0, 259.2, 311.04, 373.25, 447.9, 537.48,
        644.97, 773.97, 928.76, 1114.51, 1337.42, 1604.90, 1925.88, 2311.05,
        2773.26, 3235.47, 3697.69, 4159.90, 4622.11
    ]
}

AC_AMPLITUDES = {
    '12%': [
        18.0, 20.16, 22.58, 25.29, 28.32, 31.72, 35.53, 39.79,
        44.57, 49.92, 55.91, 62.61, 70.13, 78.54, 87.97, 98.52,
        110.35, 123.59, 138.42, 155.03, 173.63, 194.47, 217.81, 243.94,
        273.22, 306.0, 342.72, 383.85, 429.91, 481.5, 539.28
    ],
    '16%': [
        24.0, 27.84, 32.29, 37.46, 43.46, 50.41, 58.47, 67.83,
        78.68, 91.27, 105.87, 122.81, 142.46, 165.26, 191.7, 222.37,
        257.95, 299.22, 347.1, 402.64, 467.06, 467.06, 467.06, 467.06
    ],
    '20%': [
        30.0, 36.0, 43.2, 51.84, 62.21, 74.65, 89.58, 107.5,
        128.99, 154.79, 185.75, 222.9, 267.48, 320.98, 385.18, 462.21,
        462.21, 462.21, 462.21, 462.21, 462.21
    ]
}

class BitStreamWriter:
    """Packs arbitrary bit-width fields into big-endian byte arrays."""
    def __init__(self):
        self.buffer = bytearray()
        self.current_byte = 0
        self.bit_offset = 0

    def write_bits(self, value, num_bits):
        value = int(value) & ((1 << num_bits) - 1)
        for i in range(num_bits - 1, -1, -1):
            bit = (value >> i) & 1
            self.current_byte = (self.current_byte << 1) | bit
            self.bit_offset += 1
            if self.bit_offset == 8:
                self.buffer.append(self.current_byte)
                self.current_byte = 0
                self.bit_offset = 0

    def get_bytes(self):
        if self.bit_offset > 0:
            padded = self.current_byte << (8 - self.bit_offset)
            return bytes(self.buffer + bytearray([padded]))
        return bytes(self.buffer)

def iso_to_met(iso_str):
    clean_str = iso_str.strip().replace('Z', '+00:00')
    dt = datetime.datetime.fromisoformat(clean_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return int((dt - PSP_MET_EPOCH).total_seconds()), dt

def wrap_packet(spc_pkt, met_sec, met_subsec=0, wrap_apid=0x349, seq_cnt=0):
    hdr_w0 = 0x0800 | (wrap_apid & 0x07FF)
    hdr_w1 = 0xC000 | (seq_cnt & 0x3FFF)
    hdr_w2 = len(spc_pkt) + 6 - 1
    return struct.pack('>HHHIH', hdr_w0, hdr_w1, hdr_w2, met_sec, met_subsec) + spc_pkt

def make_spc_header(apid, payload_bytes, met_sec, seq_cnt=0):
    hdr_w0 = 0x0800 | (apid & 0x07FF)
    hdr_w1 = 0xC000 | (seq_cnt & 0x3FFF)
    hdr_w2 = len(payload_bytes) + 4 - 1
    return struct.pack('>HHHI', hdr_w0, hdr_w1, hdr_w2, met_sec) + payload_bytes

def find_input_file(filepath, fallback_dir):
    if os.path.isfile(filepath):
        return filepath
    joined = os.path.join(fallback_dir, os.path.basename(filepath))
    if os.path.isfile(joined):
        return joined
    raise FileNotFoundError(f"Unable to locate RTSW dataset file in '{filepath}' or '{joined}'")

class SyntheticShockScenario:
    """
    Extended Canonical Interplanetary Shock Profile:
      Slow Ambient Solar Wind -> IP Shock Compression Ramp -> Fast CME Driver Wind + Heavy Alphas.
    
    Phases & Characteristics:
      1. Pre-Shock Ambient Slow Wind (0% - 35% Duration):
         - Velocity: ~350 km/s with multi-scale turbulent drift (18s & 7.5s waves + noise).
         - Density: ~5 cm^-3, Temperature: ~50,000 K, Alpha Fraction: 0.8%.
         - Flow Deflection: Baseline theta ~ 2.5 deg, phi ~ 30 deg + ambient fluctuations.
      
      2. IP Shock Front Compression Ramp (35% - 40% Duration):
         - Velocity Jump: 350 -> 680 km/s over smooth S-curve transition + shock jitter.
         - Density Spike: 5 -> 180 cm^-3, Temperature Jump: 50,000 -> 350,000 K.
         - Secular Flow Angle Shift: Step jump in deflection theta (2.5 -> 7.5 deg) 
           and azimuthal direction phi (30 -> 135 deg) across the discontinuity.
      
      3. Post-Shock CME Driver Wind (40% - 100% Duration):
         - Velocity: High-speed driver wind decaying slowly from 680 -> 600 km/s with post-shock turbulence.
         - Density: High compressed post-shock sheath decaying 1200 -> 250 cm^-3.
         - Alpha Contamination: Sustained 4.0% He++ fraction producing distinct 2x E/q peak.
         - Flow Deflection: Sustained post-shock deflection (theta ~ 7.5 deg, phi ~ 135 deg) + noise.

    Returns: v_sw, density, temp, alpha_frac, theta_deg, phi_deg
    """
    def __init__(self, duration_sec):
        self.duration = duration_sec

    def get_state(self, elapsed_sec):
        frac = min(max(elapsed_sec / float(self.duration), 0.0), 1.0)

        # 1. Pre-Shock Ambient Slow Wind (0% - 35% Duration) -> Target Gain 3 (n ~ 5 cm^-3)
        if frac < 0.35:
            v_slow_drift = (15.0 * math.sin(2.0 * math.pi * elapsed_sec / 18.0) +
                            6.0 * math.sin(2.0 * math.pi * elapsed_sec / 7.5) +
                            4.0 * random.gauss(0, 1.0))
            n_slow_drift = 0.8 * math.sin(2.0 * math.pi * elapsed_sec / 12.0) + 0.3 * random.gauss(0, 1.0)
            
            v_sw = 350.0 + v_slow_drift
            n_sw = 5.0 + n_slow_drift
            temp_sw = 50000.0 + 4000.0 * random.gauss(0, 1.0)
            alpha_frac = 0.008

            theta_deg = 2.5 + 0.5 * math.sin(2.0 * math.pi * elapsed_sec / 8.0) + 0.3 * random.gauss(0, 1.0)
            phi_deg = 30.0 + 5.0 * math.sin(2.0 * math.pi * elapsed_sec / 15.0) + 2.0 * random.gauss(0, 1.0)

        # 2. IP Shock Front Compression Ramp (35% - 40% Duration) -> Gain 2 -> Gain 1 Transition
        elif frac < 0.40:
            s_frac = (frac - 0.35) / 0.05
            s_ramp = 0.5 * (1.0 - math.cos(math.pi * s_frac))

            v_jitter = 20.0 * random.gauss(0, 1.0)
            v_sw = 350.0 + s_ramp * (680.0 - 350.0) + v_jitter
            n_sw = 5.0 + s_ramp * (180.0 - 5.0) + 5.0 * random.gauss(0, 1.0)
            temp_sw = 50000.0 + s_ramp * (350000.0 - 50000.0) + 15000.0 * random.gauss(0, 1.0)
            alpha_frac = 0.008 + s_ramp * (0.040 - 0.008)

            theta_deg = 2.5 + s_ramp * (7.5 - 2.5) + 0.8 * random.gauss(0, 1.0)
            phi_deg = 30.0 + s_ramp * (135.0 - 30.0) + 8.0 * random.gauss(0, 1.0)

        # 3. Post-Shock CME Driver Wind (40% - 100% Duration) -> Target Gain 1 (n ~ 1200 cm^-3)
        else:
            f_frac = (frac - 0.40) / 0.60
            v_turb = 12.0 * math.sin(2.0 * math.pi * elapsed_sec / 5.0) + 8.0 * random.gauss(0, 1.0)
            
            v_sw = 680.0 - f_frac * (680.0 - 600.0) + v_turb
            n_sw = 1200.0 - f_frac * (1200.0 - 250.0) + random.gauss(0, 10.0)
            temp_sw = 350000.0 - f_frac * (350000.0 - 180000.0) + 10000.0 * random.gauss(0, 1.0)
            alpha_frac = 0.040

            theta_deg = 7.5 + 0.8 * math.sin(2.0 * math.pi * elapsed_sec / 4.0) + 0.5 * random.gauss(0, 1.0)
            phi_deg = 135.0 + 8.0 * math.sin(2.0 * math.pi * elapsed_sec / 9.0) + 4.0 * random.gauss(0, 1.0)

        return max(150.0, v_sw), max(0.05, n_sw), max(5000.0, temp_sw), max(0.0, alpha_frac), min(max(theta_deg, 0.0), 10.0), phi_deg % 360.0

class SyntheticDPUStressScenario:
    """
    Hyper-compressed 14-second DPU Stress-Test Scenario for Firmware ROM Image Validation:
      Full dynamic range stress-testing across all gain stages (Gain 3 -> 2 -> 1 -> 0):

      0.0s - 2.5s : Authentic Multi-Scale Turbulent Slow Wind 
                    (v = 220-290 km/s, theta = 3.5 deg, phi = 45 deg; exercises Gain 3 & Gain 2).
      2.5s - 3.8s : Off-Scale Low Rail Lockout Test 
                    (v = 95 km/s below table min, theta = 1.0 deg; tests low voltage lockout in Gain 3).
      3.8s - 6.5s : Smooth Velocity Acceleration Ramp 
                    (v = 150 -> 750 km/s, theta drifts 2.0 -> 7.5 deg; tests dynamic peak tracking).
      6.5s - 8.0s : Fast CME Driver Core + Heavy Density Burst 
                    (v = 750 km/s, n = 25,000 cm^-3, 5% He++, theta = 8.0 deg, phi = -60 deg; 
                     forces DPU into Gain 0 / 1x gain).
      8.0s - 10.0s: Organic Mid-Energy Transition Dip 
                    (v = 750 -> 460 -> 750 km/s, theta = 4.0 deg, phi = 180 deg; 
                     tests rapid tracking rebound across ~1100V transition).
      10.0s- 11.5s: Off-Scale High Rail Test 
                    (v = 980 km/s near table max, theta = 9.5 deg, phi = -135 deg; 
                     tests high rail boundary limits).
      11.5s- 14.0s: Extreme Vacuum Drop & Recovery 
                    (n = 0.01 cm^-3 vacuum drop -> recovery to n = 15 cm^-3; tests noise floor lockout).

    Returns: v_sw, density, temp, alpha_frac, theta_deg, phi_deg
    """
    def __init__(self, duration_sec=14.0):
        self.duration = duration_sec

    def get_state(self, elapsed_sec):
        clamped_time = min(max(elapsed_sec, 0.0), self.duration)

        # 1. Gain 3 Target (0.0s - 2.5s): Low density (n ~ 5 cm^-3 -> I_coll ~ 18 pA < 1752 pA)
        if clamped_time < 2.5:
            v_turb = (18.0 * math.sin(2.0 * math.pi * clamped_time * 0.43 + 0.5) +
                      9.0 * math.sin(2.0 * math.pi * clamped_time * 1.37 + 1.2) +
                      2.5 * random.gauss(0, 1.0))
            v_sw = 250.0 + v_turb
            n_sw = 5.0 + 0.5 * random.gauss(0, 1.0)
            temp_sw = 25000.0 + 3000.0 * random.gauss(0, 1.0)
            alpha_frac = 0.005
            theta_deg = 3.5 + 1.2 * math.sin(2.0 * math.pi * clamped_time * 0.3)
            phi_deg = 45.0

        # 2. Low Rail Test (2.5s - 3.8s): Gain 2 target density (n ~ 120 cm^-3 -> I_coll ~ 300 pA)
        elif clamped_time < 3.8:
            v_sw = 95.0 + 3.0 * random.gauss(0, 1.0)
            n_sw = 120.0
            temp_sw = 15000.0
            alpha_frac = 0.002
            theta_deg = 1.0
            phi_deg = 0.0

        # 3. Gain 1 Target (3.8s - 6.5s): High density ramp (n ~ 1500 cm^-3 -> I_coll ~ 12,000 pA)
        elif clamped_time < 6.5:
            ramp_frac = (clamped_time - 3.8) / 2.7
            s_ramp = 0.5 * (1.0 - math.cos(math.pi * ramp_frac))
            
            v_turb = 12.0 * math.sin(2.0 * math.pi * clamped_time * 1.1) + 4.0 * random.gauss(0, 1.0)
            v_sw = 150.0 + s_ramp * (750.0 - 150.0) + v_turb
            n_sw = 120.0 + s_ramp * (1380.0)
            temp_sw = 25000.0 + s_ramp * (280000.0 - 25000.0)
            alpha_frac = 0.005 + s_ramp * 0.045
            theta_deg = 2.0 + s_ramp * 5.5
            phi_deg = 120.0

        # 4. Gain 0 Target (6.5s - 8.0s): Extreme peak burst (n ~ 25,000 cm^-3 -> I_coll ~ 209,000 pA > 448,512 pA)
        elif clamped_time < 8.0:
            v_turb = 15.0 * math.sin(2.0 * math.pi * clamped_time * 0.9) + 5.0 * random.gauss(0, 1.0)
            v_sw = 750.0 + v_turb
            n_sw = 25000.0 + 500.0 * random.gauss(0, 1.0)
            temp_sw = 300000.0
            alpha_frac = 0.050
            theta_deg = 8.0 + 1.0 * math.sin(2.0 * math.pi * clamped_time * 1.5)
            phi_deg = -60.0

        # 5. Transition Dip back to Gain 3 (8.0s - 10.0s)
        elif clamped_time < 10.0:
            dip_phase = (clamped_time - 8.0) / 2.0
            dip_factor = math.sin(math.pi * dip_phase)
            
            v_turb = (18.0 * math.sin(2.0 * math.pi * clamped_time * 1.25) +
                      8.0 * math.sin(2.0 * math.pi * clamped_time * 2.8) +
                      3.0 * random.gauss(0, 1.0))
            
            v_sw = 750.0 - dip_factor * (750.0 - 460.0) + v_turb
            n_sw = 25000.0 - dip_factor * 24995.0  # Drops back to 5.0 cm^-3
            temp_sw = 300000.0 - dip_factor * 180000.0
            alpha_frac = 0.050 - dip_factor * 0.030
            theta_deg = 8.0 - dip_factor * 4.0
            phi_deg = 180.0

        # 6. Off-Scale High Rail Test (10.0s - 11.5s)
        elif clamped_time < 11.5:
            v_sw = 980.0 + 10.0 * random.gauss(0, 1.0)
            n_sw = 800.0
            temp_sw = 250000.0
            alpha_frac = 0.030
            theta_deg = 9.5
            phi_deg = -135.0

        # 7. Extreme Vacuum Drop & Recovery (11.5s - 14.0s)
        else:
            rec_frac = (clamped_time - 11.5) / 2.5
            v_sw = 980.0 - rec_frac * (980.0 - 420.0) + 8.0 * math.sin(2.0 * math.pi * clamped_time * 0.7)
            n_sw = 0.01 if rec_frac < 0.4 else (rec_frac - 0.4) * 15.0
            temp_sw = 250000.0 - rec_frac * (250000.0 - 70000.0)
            alpha_frac = 0.010
            theta_deg = 2.0
            phi_deg = 0.0

        return max(10.0, v_sw), max(0.001, n_sw), max(1000.0, temp_sw), max(0.0, alpha_frac), theta_deg, phi_deg

class SyntheticRegimeMatrixScenario:
    """
    Open-Ended Dynamic Parameter Matrix with Configurable Block Duration (-bd / --block-duration):
    
    Divides time into stationary blocks of user-defined duration (block_size_sec, default = 1.0s).
    First 14 preset blocks use a curated canonical parameter matrix.
    Beyond 14 preset blocks, an open-ended procedural generator creates endless regime 
    blocks sampled across the 1 AU solar wind continuum.

    Procedural Sampling (Open-Ended Blocks):
      - Speed (v_base): 250 km/s to 950 km/s
      - Density (n_base): Log-sampled 0.05 cm^-3 to 2000 cm^-3 (with 5% CME spikes up to 25,000 cm^-3)
      - Temperature (t_base): Log-sampled 1.5e4 K to 6.0e5 K
      - Flow Deflections (theta, phi): theta in [0.5, 8.5] deg, phi in [-180, 180) deg

    Intra-Segment Turbulence & Thermal Jitter (Every Block):
      - Velocity Waves & Red Noise: Drives peak-to-peak speed fluctuations (+/-50 to 90 km/s),
        forcing energy window boundaries to wander up/down by 1 to 3 steps per block.
      - Stochastic Thermal Jitter: Layered on top of velocity-temperature scaling to prevent 
        compression algorithms or linear predictors from deterministically guessing T_sw.
      - Density & Flow Angle Dynamics: Modulates density (+/-30%) and deflection angles (theta, phi).

    Returns: v_sw, density, temp, alpha_frac, theta_deg, phi_deg
    """
    def __init__(self, duration_sec=60.0, block_size_sec=1.0):
        self.duration = duration_sec
        self.block_size = float(block_size_sec)  # Configurable block duration in seconds
        
        # 14 Canonical Presets for initial blocks [v (km/s), n (cm^-3), T (K), alpha_frac, theta (deg), phi (deg)]
        self.preset_regimes = [
            (280.0, 2.00, 20000.0, 0.005, 1.0, 10.0),       # Block 0: Slow, Low Density
            (300.0, 80.00, 35000.0, 0.008, 2.0, 25.0),      # Block 1: Slow, High Density
            (360.0, 1.00, 35000.0, 0.006, 1.8, 15.0),       # Block 2: Slow, Rarefied
            (380.0, 250.00, 60000.0, 0.012, 3.5, 60.0),     # Block 3: Slow, Heavy Dense
            (450.0, 0.30, 50000.0, 0.010, 2.2, -15.0),     # Block 4: Moderate, Vacuum
            (480.0, 12.00, 80000.0, 0.015, 3.0, 45.0),      # Block 5: Moderate, Nominal
            (520.0, 120.00, 150000.0, 0.025, 5.0, 80.0),    # Block 6: Moderate, High Density
            (700.0, 1.50, 180000.0, 0.030, 4.0, -25.0),     # Block 7: Fast, Low Density
            (680.0, 1800.00, 350000.0, 0.040, 6.5, 110.0),  # Block 8: Fast, Very Dense
            (750.0, 25.00, 250000.0, 0.035, 4.5, -40.0),    # Block 9: Fast, Moderate Density
            (850.0, 0.08, 100000.0, 0.020, 3.0, 10.0),      # Block 10: Very Fast, Extreme Vacuum
            (900.0, 25000.00, 500000.0, 0.050, 8.5, -120.0),# Block 11: Very Fast, CME Core (Gain 0)
            (880.0, 150.00, 400000.0, 0.045, 7.0, -90.0),   # Block 12: Very Fast, Moderate Density
            (400.0, 8.00, 45000.0, 0.010, 2.0, 30.0)        # Block 13: Recovery
        ]
        
        # Cache for procedurally generated blocks beyond initial presets
        self.generated_blocks = {}

        # AR(1) Red Noise Memory Parameters
        self.alpha_memory = 0.78
        self.v_red = 0.0
        self.n_red_frac = 0.0
        self.t_red_frac = 0.0
        self.theta_red = 0.0
        self.phi_red = 0.0
        self.last_t = None

    def _get_regime_for_block(self, block_idx):
        if block_idx < len(self.preset_regimes):
            return self.preset_regimes[block_idx]

        if block_idx in self.generated_blocks:
            return self.generated_blocks[block_idx]

        # Seeded Random for reproducible open-ended generation per block index
        rng = random.Random(42 + block_idx * 10007)

        v_base = rng.uniform(250.0, 950.0)
        
        # 5% probability of CME super-peak density, otherwise log-uniform 0.05..2000
        if rng.random() < 0.05:
            n_base = rng.uniform(2000.0, 25000.0)
        else:
            n_base = 10.0 ** rng.uniform(-1.3, 3.3)

        t_base = 10.0 ** rng.uniform(4.18, 5.78)  # ~1.5e4 K to 6e5 K
        alpha_base = rng.uniform(0.005, 0.050)
        theta_base = rng.uniform(0.5, 8.5)
        phi_base = rng.uniform(-180.0, 180.0)

        regime = (v_base, n_base, t_base, alpha_base, theta_base, phi_base)
        self.generated_blocks[block_idx] = regime
        return regime

    def get_state(self, elapsed_sec):
        clamped_time = max(elapsed_sec, 0.0)
        block_idx = int(clamped_time // self.block_size)
        
        v_base, n_base, t_base, alpha_base, theta_base, phi_base = self._get_regime_for_block(block_idx)

        # Update AR(1) Red Noise state step
        if self.last_t is None or elapsed_sec < self.last_t:
            self.v_red = random.gauss(0, 30.0)
            self.n_red_frac = random.gauss(0, 0.15)
            self.t_red_frac = random.gauss(0, 0.12)
            self.theta_red = random.gauss(0, 0.8)
            self.phi_red = random.gauss(0, 4.0)
        else:
            scale = math.sqrt(1.0 - self.alpha_memory**2)
            # AR1 updates: x_t = alpha * x_{t-1} + scale * Gauss(0, std)
            self.v_red = self.alpha_memory * self.v_red + scale * random.gauss(0, 55.0)
            self.n_red_frac = self.alpha_memory * self.n_red_frac + scale * random.gauss(0, 0.20)
            self.t_red_frac = self.alpha_memory * self.t_red_frac + scale * random.gauss(0, 0.18)
            self.theta_red = self.alpha_memory * self.theta_red + scale * random.gauss(0, 1.0)
            self.phi_red = self.alpha_memory * self.phi_red + scale * random.gauss(0, 6.0)

        self.last_t = elapsed_sec

        # Fast wave modulation driving peak energy window shifts (+/-1 to 3 windows)
        v_wave = 45.0 * math.sin(2.0 * math.pi * elapsed_sec / 0.35) + 25.0 * math.cos(2.0 * math.pi * elapsed_sec / 0.6)
        n_wave = 0.25 * math.sin(2.0 * math.pi * elapsed_sec / 0.3)

        v_sw = max(150.0, v_base + v_wave + self.v_red)
        n_sw = max(0.01, n_base * (1.0 + n_wave + self.n_red_frac))

        # Temperature: Velocity-correlated base scaling + independent stochastic red noise jitter
        t_velocity_coupled = t_base * (1.0 + 0.12 * ((v_wave + self.v_red) / 50.0))
        temp_sw = max(5000.0, t_velocity_coupled * (1.0 + self.t_red_frac))

        theta_deg = min(max(theta_base + 1.2 * math.sin(2.0 * math.pi * elapsed_sec / 0.25) + self.theta_red, 0.0), 10.0)
        phi_deg = (phi_base + 10.0 * math.cos(2.0 * math.pi * elapsed_sec / 0.4) + self.phi_red) % 360.0

        return v_sw, n_sw, temp_sw, alpha_base, theta_deg, phi_deg

class DSCOVRDataset:
    """Parses DSCOVR RTSW text file, extracts speed, density, temperature, and velocity directions."""
    def __init__(self, filepath):
        self.times, self.mets = [], []
        self.v, self.v_sigma = [], []
        self.n, self.n_sigma = [], []
        self.temp, self.temp_sigma = [], []
        self.theta, self.phi = [], []
        self._parse_file(filepath)

    def _parse_file(self, filepath):
        print(f"Loading DSCOVR RTSW dataset: {filepath}...")
        with open(filepath, 'r') as f:
            lines = f.readlines()

        start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("Timestamp"):
                start_idx = i + 1
                break

        last_v, last_n, last_temp = 400.0, 5.0, 1e5

        for line in lines[start_idx:]:
            parts = line.strip().split()
            if len(parts) < 29:
                continue

            try:
                ts_str = f"{parts[0]}T{parts[1].replace('Z', '')}"
                dt = datetime.datetime.fromisoformat(ts_str).replace(tzinfo=datetime.timezone.utc)
                met = int((dt - PSP_MET_EPOCH).total_seconds())

                raw_n = float(parts[20])
                raw_v = float(parts[23])
                raw_temp = float(parts[26])

                n_val = raw_n if raw_n > -9000 else last_n
                v_val = raw_v if raw_v > -9000 else last_v
                t_val = raw_temp if raw_temp > -9000 else last_temp

                last_n, last_v, last_temp = n_val, v_val, t_val

                theta_val = 3.0 + 1.5 * math.sin(met / 300.0)
                phi_val = (met / 10.0) % 360.0

                self.times.append(dt)
                self.mets.append(met)
                self.v.append(v_val)
                self.v_sigma.append(2.0)
                self.n.append(n_val)
                self.n_sigma.append(0.2)
                self.temp.append(t_val)
                self.temp_sigma.append(2000.0)
                self.theta.append(theta_val)
                self.phi.append(phi_val)

            except Exception:
                continue

        if not self.mets:
            raise ValueError("No valid solar wind data parsed from file.")
        print(f"Loaded {len(self.mets)} samples. Span: {self.times[0].strftime('%Y-%m-%d %H:%M')} to {self.times[-1].strftime('%Y-%m-%d %H:%M')}")

    def get_interpolated_state(self, target_met, add_noise=True):
        if target_met <= self.mets[0]:
            idx0, idx1, frac = 0, 0, 0.0
        elif target_met >= self.mets[-1]:
            idx0, idx1, frac = len(self.mets) - 1, len(self.mets) - 1, 0.0
        else:
            idx1 = bisect_right(self.mets, target_met)
            idx0 = idx1 - 1
            t0, t1 = self.mets[idx0], self.mets[idx1]
            frac = (target_met - t0) / float(t1 - t0) if t1 > t0 else 0.0

        v_interp = self.v[idx0] + frac * (self.v[idx1] - self.v[idx0])
        n_interp = self.n[idx0] + frac * (self.n[idx1] - self.n[idx0])
        t_interp = self.temp[idx0] + frac * (self.temp[idx1] - self.temp[idx0])
        theta_interp = self.theta[idx0] + frac * (self.theta[idx1] - self.theta[idx0])
        phi_interp = self.phi[idx0] + frac * (self.phi[idx1] - self.phi[idx0])

        if add_noise:
            v_interp += random.gauss(0, self.v_sigma[idx0])
            n_interp += random.gauss(0, self.n_sigma[idx0])
            t_interp += random.gauss(0, self.temp_sigma[idx0])
            theta_interp += random.gauss(0, 0.2)

        alpha_frac = 0.035 if v_interp > 500.0 else 0.008
        return max(150.0, v_interp), max(0.05, n_interp), max(5000.0, t_interp), alpha_frac, min(max(theta_interp, 0.0), 10.0), phi_interp

def proton_speed_to_voltage(v_kms):
    m_p = 1.6726219e-27
    e = 1.6021766e-19
    v_m_s = v_kms * 1000.0
    return (0.5 * m_p * (v_m_s ** 2)) / e

def calculate_channel_measurement(v_window, ac_amp, v_peak_sw, thermal_speed_v, density, 
                                  theta_deg=0.0, phi_deg=0.0, alpha_frac=0.0, noise_floor_pA=0.5, thresh_counts=1752):
    baseline = 2048
    q_e = 1.60217663e-19  # Coulomb

    # 1. Physics First Principles Current Flux J (A/cm^2)
    j_proton = density * v_peak_sw * 1e5 * q_e
    j_alpha = (2.0 * alpha_frac * density) * v_peak_sw * 1e5 * q_e if alpha_frac > 0.0 else 0.0

    # 2. Window Energy Thermal Broadening Response W(V)
    delta_v_p = math.sqrt(v_window) - math.sqrt(v_peak_sw)
    w_proton = math.exp(-0.5 * (delta_v_p / (thermal_speed_v * 0.05)) ** 2)

    w_alpha = 0.0
    if alpha_frac > 0.0:
        v_peak_alpha = 2.0 * v_peak_sw
        delta_v_a = math.sqrt(v_window) - math.sqrt(v_peak_alpha)
        w_alpha = math.exp(-0.5 * (delta_v_a / (thermal_speed_v * 0.05)) ** 2)

    j_total_a_cm2 = (j_proton * w_proton) + (j_alpha * w_alpha)

    # 3. Geometric Aperture & Combined Grid Transparency Scaling
    a_eff_cm2 = 13.5          # Effective aperture area (cm^2)
    grid_transparency = 0.387  # 8-grid combined optical transparency (88.8%^8)

    i_plasma_total_pA = j_total_a_cm2 * a_eff_cm2 * grid_transparency * 1e12

    # 4. 4-Quadrant Geometric Collector Split (Clamped to +/- 10-degree cone)
    theta_rad = math.radians(min(max(theta_deg, 0.0), 10.0))
    phi_rad = math.radians(phi_deg)
    
    alpha_resp = 2.5
    tan_theta = math.tan(theta_rad)

    k_a = max(0.05, 1.0 + alpha_resp * tan_theta * math.cos(phi_rad))
    k_b = max(0.05, 1.0 + alpha_resp * tan_theta * math.sin(phi_rad))
    k_c = max(0.05, 1.0 - alpha_resp * tan_theta * math.cos(phi_rad))
    k_d = max(0.05, 1.0 - alpha_resp * tan_theta * math.sin(phi_rad))

    k_sum = k_a + k_b + k_c + k_d
    k_weights = [4.0 * k_a / k_sum, 4.0 * k_b / k_sum, 4.0 * k_c / k_sum, 4.0 * k_d / k_sum]

    i_cap = 0.0025 * ac_amp

    # Canonical Jython Gain Formula: Current = counts * 16^(3 - gain)
    pA_per_count = [16.0**(3 - g) for g in range(4)]  # [4096.0, 256.0, 16.0, 1.0]
    channel_meas = []

    for col_idx in range(4):
        i_collector_sig = (i_plasma_total_pA * k_weights[col_idx] / 4.0) + i_cap

        phi_sig_rad = math.radians(random.gauss(200.0, 3.0))
        i_sin_sig = i_collector_sig * math.sin(phi_sig_rad)
        i_cos_sig = i_collector_sig * math.cos(phi_sig_rad)

        i_noise_pA = max(0.0, random.gauss(noise_floor_pA, noise_floor_pA * 0.2))
        phi_noise_rad = math.radians(random.gauss(160.0, 15.0))
        i_sin_noise = i_noise_pA * math.sin(phi_noise_rad)
        i_cos_noise = i_noise_pA * math.cos(phi_noise_rad)

        i_sin_total = i_sin_sig + i_sin_noise
        i_cos_total = i_cos_sig + i_cos_noise

        selected_gain = 0
        sin_adc = 0
        cos_adc = 0

        # Select highest sensitivity gain (3 -> 2 -> 1 -> 0)
        for g_code in [3, 2, 1, 0]:
            scale = pA_per_count[g_code]
            scaled_sin = i_sin_total / scale
            scaled_cos = i_cos_total / scale

            if abs(scaled_sin) <= thresh_counts and abs(scaled_cos) <= thresh_counts:
                selected_gain = g_code
                # Two's complement 12-bit encoding (0..4095)
                sin_adc = int(round(scaled_sin)) & 0xFFF
                cos_adc = int(round(scaled_cos)) & 0xFFF
                break
        else:
            selected_gain = 0
            scaled_sin = i_sin_total / pA_per_count[0]
            scaled_cos = i_cos_total / pA_per_count[0]
            sin_adc = int(round(min(max(scaled_sin, -2048), 2047))) & 0xFFF
            cos_adc = int(round(min(max(scaled_cos, -2048), 2047))) & 0xFFF

        channel_meas.append((selected_gain, sin_adc, cos_adc))

    return channel_meas

def pack_raw_measurement_slot(win_idx, channel_measurements):
    w = BitStreamWriter()
    w.write_bits(win_idx, 8)
    for gain_code, sin_adc, cos_adc in channel_measurements:
        w.write_bits(gain_code, 2)
        w.write_bits(sin_adc, 12)
        w.write_bits(cos_adc, 12)
    return w.get_bytes()

def build_0x353_payload_bytes(met_subsec, slot_measurements, doublewidth_idx=DOUBLEWIDTH_INDEX):
    w = BitStreamWriter()
    w.write_bits(met_subsec, 16)
    w.write_bits(ST_VAL, 6)
    w.write_bits(IT_VAL, 10)
    w.write_bits(0, 8)
    w.write_bits(doublewidth_idx, 4)
    w.write_bits(0, 4)
    
    for win_idx, channel_data in slot_measurements:
        w.write_bits(win_idx, 8)
        for gain_code, sin_adc, cos_adc in channel_data:
            w.write_bits(gain_code, 2)
            w.write_bits(sin_adc, 12)
            w.write_bits(cos_adc, 12)

    return w.get_bytes()

def generate_apid351(met_sec, met_subsec=0):
    w = BitStreamWriter()
    w.write_bits(met_subsec, 16)
    w.write_bits(0, 8)
    w.write_bits(128, 8)
    for i in range(128):
        baseline = 2048 + int(50 * math.sin(2 * math.pi * i / 16.0))
        w.write_bits(min(max(baseline, 0), 4095), 12)
        w.write_bits(min(max(baseline, 0), 4095), 12)
    return make_spc_header(0x351, w.get_bytes(), met_sec)

def generate_apid35e(met_sec, peak_repeat_cnt=FULL_SCAN_INTERVAL, doublewidth_idx=DOUBLEWIDTH_INDEX):
    w = BitStreamWriter()
    w.write_bits(0, 16)
    w.write_bits(100, 16)
    w.write_bits(PEAK_OFF_LO, 8)
    w.write_bits(PEAK_OFF_HI, 8)
    w.write_bits(peak_repeat_cnt, 16)
    for _ in range(4): w.write_bits(1000, 12)
    w.write_bits(RETRACE_COUNT, 4)
    w.write_bits(45, 8)
    w.write_bits(0, 4)
    w.write_bits(255, 8)
    w.write_bits(doublewidth_idx, 4)
    w.write_bits(1, 4)
    for _ in range(4): w.write_bits(3800, 12)
    for _ in range(12): w.write_bits(3500, 12)
    w.write_bits(doublewidth_idx, 8)
    w.write_bits(0, 8)
    w.write_bits(0, 5)
    w.write_bits(1, 3)
    w.write_bits(12, 8)
    return make_spc_header(0x35E, w.get_bytes(), met_sec)

def generate_apid35f(met_sec, v_sw):
    w = BitStreamWriter()
    w.write_bits(0, 16)
    w.write_bits(1, 8)
    w.write_bits(0, 8)
    w.write_bits(0xF0, 8)
    w.write_bits(0, 16)
    w.write_bits(0b11100000, 8)
    for val in [2556, 2052, 2540, 1922, 1708, 312, 1755, 1408]: w.write_bits(val, 12)
    for val in [610, 642, 131, 130, 896, 643, 613, 643, 899, 640, 384, 640, 128, 384, 613, 640]: w.write_bits(val, 10)
    w.write_bits(int(v_sw), 16)
    w.write_bits(1, 4)
    w.write_bits(1200, 12)
    w.write_bits(100, 16)
    w.write_bits(200, 16)
    w.write_bits(10, 8)
    w.write_bits(2, 8)
    w.write_bits(1, 4)
    w.write_bits(2048, 12)
    w.write_bits(1, 8)
    w.write_bits(500, 16)
    w.write_bits(0, 8)
    w.write_bits(255, 8)
    w.write_bits(0, 8)
    w.write_bits(0xABCD, 16)
    return make_spc_header(0x35F, w.get_bytes(), met_sec)

def format_duration_str(seconds: int) -> str:
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    elif seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"

def resolve_output_filepaths(args):
    run_date_str = datetime.date.today().strftime("%Y-%m-%d")
    mode_str = "spc" if args.mod_freq == 1171.875 else "hsfc"
    table_str = args.table
    dur_str = format_duration_str(args.duration)
    wrap_str = "noWrapper" if args.no_wrapper else "swemWrapper"
    scen_str = f"_{args.scenario}Profile"

    folder_name = f"{run_date_str}_{mode_str}_{table_str}_{dur_str}_{wrap_str}{scen_str}"
    full_dir = os.path.join(args.output_dir, folder_name)
    os.makedirs(full_dir, exist_ok=True)

    if args.output:
        telemetry_filepath = args.output
        raw_filepath = args.output.replace(".dat", "_rawBits.bin") if ".dat" in args.output else f"{args.output}_rawBits.bin"
    else:
        file_base = f"{run_date_str}_{mode_str}-modulation_{table_str}-windowSize_{dur_str}_{wrap_str}{scen_str}"
        telemetry_filepath = os.path.join(full_dir, f"{file_base}.dat")
        raw_filepath = os.path.join(full_dir, f"{file_base}_rawBits.bin")

    return telemetry_filepath, raw_filepath

def main():
    parser = argparse.ArgumentParser(description="DSCOVR-Driven Synthetic Telemetry Generator")
    parser.add_argument("-i", "--input-file", type=str, default="rtsw_data_20260805_023054.txt", help="Path to DSCOVR RTSW text file")
    parser.add_argument("--start-day", type=str, default="2026-06-24", help="Start date in dataset")
    parser.add_argument("-d", "--duration", type=int, default=60, help="Simulation duration in seconds")
    parser.add_argument("-bd", "--block-duration", type=float, default=1.0, help="Duration of each regime block in seconds (matrix scenario)")
    parser.add_argument("--scenario", choices=['dscovr', 'shock', 'stress', 'matrix'], default='matrix', help="Data source profile")
    parser.add_argument("--mod-freq", choices=[1171.875, 1024.0], default=1024.0, type=float, help="Modulation Clock Freq Hz")
    parser.add_argument("--table", choices=['12%', '16%', '20%'], default='16%', help="Logarithmic Energy Table")
    parser.add_argument("-rm", "--raw-measurements", action="store_true", help="Dump parallel raw 112-bit measurement slots (.bin)")
    parser.add_argument("--apids", nargs="+", default=['351', '353', '35E', '35F'], help="Select APIDs to output")
    parser.add_argument("-nw", "--no-wrapper", action="store_true", help="Omit 12-byte SWEM wrappers")
    parser.add_argument("-o", "--output", type=str, default=None, help="Explicit output file path")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_ROOT, help="Root directory for structured outputs")
    parser.add_argument("--no-noise", action="store_true", help="Disable read noise floor")
    args = parser.parse_args()

    telemetry_filepath, raw_filepath = resolve_output_filepaths(args)

    if args.scenario == 'matrix':
        dataset = SyntheticRegimeMatrixScenario(args.duration, block_size_sec=args.block_duration)
    elif args.scenario == 'stress':
        dataset = SyntheticDPUStressScenario(args.duration)
    elif args.scenario == 'shock':
        dataset = SyntheticShockScenario(args.duration)
    else:
        input_file_path = find_input_file(args.input_file, args.output_dir)
        dataset = DSCOVRDataset(input_file_path)

    energy_table = np.array(ENERGY_TABLES[args.table], dtype=np.float32)
    ac_amplitudes = np.array(AC_AMPLITUDES[args.table], dtype=np.float32)
    n_table_steps = len(energy_table)

    if 'T' not in args.start_day:
        start_iso = f"{args.start_day}T00:00:00Z"
    else:
        start_iso = args.start_day
        
    start_met, start_dt = iso_to_met(start_iso)
    mod_freq = args.mod_freq

    slot_time_sec = WAVE_PERIODS_PER_SLOT / mod_freq
    subsec_per_slot = slot_time_sec * 65536.0

    print(f"\nStarting Synthetic Generator:")
    print(f"  Scenario Mode        : {args.scenario.upper()}")
    print(f"  Duration             : {args.duration} s ({args.duration * 8} total sweeps)")
    if args.scenario == 'matrix':
        print(f"  Regime Block Duration: {args.block_duration} s ({int(args.duration / args.block_duration)} total blocks)")
    print(f"  Modulation Frequency : {mod_freq} Hz ({'SPC' if mod_freq==1171.875 else 'HSFC'})")
    print(f"  Energy Table         : {args.table} ({n_table_steps} total windows)")
    print(f"  Encoding             : 12-Bit Two's Complement Signed Integers")
    print(f"  Gain Scaling         : Current = counts * 16^(3 - gain)")
    print(f"  Primary Telemetry    : {telemetry_filepath}")

    seq_cnt = 0
    current_met = start_met
    current_subsec_exact = 0.0
    elapsed_time = 0.0
    last_sec_met = -1
    packet_353_count = 0
    total_slots_written = 0

    sweep_counter = 0
    prev_peak_win = n_table_steps // 2
    
    slot_buffer = []
    nys_slot_count = 0
    packet_start_met = current_met
    packet_start_subsec = 0

    selected_apids = {a.upper().replace('0X', '') for a in args.apids}

    f_dat = open(telemetry_filepath, "wb")
    f_raw = open(raw_filepath, "wb") if args.raw_measurements else None

    def advance_time_one_slot():
        nonlocal elapsed_time, current_subsec_exact, current_met
        elapsed_time += slot_time_sec
        current_subsec_exact += subsec_per_slot
        while current_subsec_exact >= 65536.0:
            current_met += 1
            current_subsec_exact -= 65536.0

    def flush_0x353_packet():
        nonlocal seq_cnt, packet_353_count, slot_buffer
        if len(slot_buffer) > 0 and '353' in selected_apids:
            payload_bytes = build_0x353_payload_bytes(packet_start_subsec, slot_buffer)
            pkt_353 = make_spc_header(0x353, payload_bytes, packet_start_met)

            if not args.no_wrapper:
                pkt_353 = wrap_packet(pkt_353, packet_start_met, met_subsec=packet_start_subsec, seq_cnt=seq_cnt)
            f_dat.write(pkt_353)
            seq_cnt += 1
            packet_353_count += 1
        slot_buffer = []

    try:
        while elapsed_time < args.duration:
            target_met = start_met + elapsed_time
            met_sec = current_met

            if args.scenario in ['matrix', 'stress', 'shock']:
                v_sw, density, temp, alpha_frac, theta_deg, phi_deg = dataset.get_state(elapsed_time)
            else:
                v_sw, density, temp, alpha_frac, theta_deg, phi_deg = dataset.get_interpolated_state(target_met, add_noise=(not args.no_noise))

            if met_sec > last_sec_met:
                if '351' in selected_apids:
                    pkt_351 = generate_apid351(met_sec, int(current_subsec_exact) & 0xFFFF)
                    if not args.no_wrapper:
                        pkt_351 = wrap_packet(pkt_351, met_sec, met_subsec=int(current_subsec_exact) & 0xFFFF, seq_cnt=seq_cnt)
                    f_dat.write(pkt_351)
                    seq_cnt += 1

                if '35F' in selected_apids:
                    pkt_35f = generate_apid35f(met_sec, v_sw)
                    if not args.no_wrapper:
                        pkt_35f = wrap_packet(pkt_35f, met_sec, seq_cnt=seq_cnt)
                    f_dat.write(pkt_35f)
                    seq_cnt += 1

                if '35E' in selected_apids:
                    pkt_35e = generate_apid35e(met_sec)
                    if not args.no_wrapper:
                        pkt_35e = wrap_packet(pkt_35e, met_sec, seq_cnt=seq_cnt)
                    f_dat.write(pkt_35e)
                    seq_cnt += 1

                last_sec_met = met_sec

            v_peak_target = proton_speed_to_voltage(v_sw)
            thermal_v = math.sqrt(2.0 * 1.380649e-23 * temp / 1.6726219e-27) / 1000.0

            actual_peak_win = min(range(n_table_steps), key=lambda i: abs(energy_table[i] - v_peak_target))

            is_full_scan = (sweep_counter == 0)

            if is_full_scan:
                scan_windows = list(range(n_table_steps))
            else:
                low_target = prev_peak_win - PEAK_OFF_LO
                high_target = prev_peak_win + PEAK_OFF_HI
                window_count = PEAK_OFF_LO + 1 + PEAK_OFF_HI

                if low_target < 0:
                    low_win = 0
                    high_win = window_count - 1
                elif high_target >= n_table_steps:
                    high_win = n_table_steps - 1
                    low_win = n_table_steps - window_count
                else:
                    low_win = low_target
                    high_win = high_target

                scan_windows = list(range(low_win, high_win + 1))

            prev_peak_win = actual_peak_win
            sweep_counter = (sweep_counter + 1) % FULL_SCAN_INTERVAL

            retrace_win = scan_windows[0]
            telemetered_windows = [retrace_win] + scan_windows

            noise_floor_pA = 0.0 if args.no_noise else 0.5

            for win_idx in telemetered_windows:
                v_win = energy_table[win_idx]
                ac_amp = ac_amplitudes[win_idx]
                
                channel_meas = calculate_channel_measurement(
                    v_win, ac_amp, v_peak_target, thermal_v, density, 
                    theta_deg=theta_deg, phi_deg=phi_deg, alpha_frac=alpha_frac, 
                    noise_floor_pA=noise_floor_pA
                )

                if nys_slot_count == 0:
                    packet_start_met = current_met
                    packet_start_subsec = int(current_subsec_exact) & 0xFFFF

                if f_raw is not None:
                    raw_bytes = pack_raw_measurement_slot(win_idx, channel_meas)
                    f_raw.write(raw_bytes)
                    total_slots_written += 1

                slot_buffer.append((win_idx, channel_meas))
                nys_slot_count += 1

                if nys_slot_count == SLOTS_PER_NYS:
                    flush_0x353_packet()
                    nys_slot_count = 0

                advance_time_one_slot()

            for _ in range(SLEW_COUNT):
                nys_slot_count += 1
                if nys_slot_count == SLOTS_PER_NYS:
                    flush_0x353_packet()
                    nys_slot_count = 0
                advance_time_one_slot()

    finally:
        f_dat.close()
        if f_raw is not None:
            f_raw.close()

    print(f"\nSuccessfully generated DPU synthetic telemetry stream:")
    print(f"  ✔ Primary CCSDS Telemetry (.dat): {packet_353_count} APID 0x353 science packets.")

if __name__ == "__main__":
    main()
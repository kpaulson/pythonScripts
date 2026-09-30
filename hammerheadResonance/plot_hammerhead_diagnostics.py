import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors
import matplotlib.patheffects as path_effects
from scipy.stats import binned_statistic_2d
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import glob
import re

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import config

class DualLogger:
    """Duplicates stdout prints to a text file while maintaining live terminal output."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log_file = open(filepath, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

# =========================================================================
# COLD-PLASMA DISPERSION WAVENUMBER SOLVER
# =========================================================================

def calculate_cold_plasma_kdisp(df_input):
    """
    Calculates Doppler-corrected R-mode cold plasma wavenumber k_disp 
    independently from bulk solar wind plasma parameters (B0, ne, Vsw).
    """
    e = 1.602176634e-19
    mp = 1.67262192369e-27
    me = 9.1093837015e-31
    eps0 = 8.8541878128e-12
    c = 299792458.0

    k_disp_list = []

    for _, row in df_input.iterrows():
        f_sc = row.get('f_weighted_Hz', np.nan)
        B0_nT = row.get('B0_mag_nT', np.nan)
        ne_cm3 = row.get('ne_lfr', np.nan)

        if not (np.isfinite(f_sc) and np.isfinite(B0_nT) and np.isfinite(ne_cm3) and f_sc > 0 and B0_nT > 0 and ne_cm3 > 0):
            k_disp_list.append(np.nan)
            continue

        w_sc = f_sc * 2 * np.pi
        B0_T = B0_nT * 1e-9
        ne_m3 = ne_cm3 * 1e6

        v_sw_kms = row.get('v_sw_kms', np.nan)
        v_sw_ms = (v_sw_kms * 1000.0) if np.isfinite(v_sw_kms) else 350000.0

        theta_deg = row.get('theta_kB_deg', np.nan)
        theta_kB_deg = theta_deg if np.isfinite(theta_deg) else 15.0
        cos_theta = max(np.cos(np.radians(theta_kB_deg)), 0.1)

        w_ci = (e * B0_T) / mp
        w_ce = (e * B0_T) / me
        w_pi = np.sqrt((ne_m3 * e**2) / (eps0 * mp))
        w_pe = np.sqrt((ne_m3 * e**2) / (eps0 * me))

        def dispersion_objective(w_pl):
            if w_pl <= 0 or w_pl >= w_ci:
                return 1e9
            n2_R = 1.0 - (w_pe**2 / (w_pl * (w_pl - w_ce * cos_theta))) - (w_pi**2 / (w_pl * (w_pl + w_ci)))
            if n2_R <= 0:
                return 1e9
            k = (w_pl / c) * np.sqrt(n2_R)
            return w_pl + k * v_sw_ms * cos_theta - w_sc

        try:
            sol = root_scalar(dispersion_objective, bracket=[1e-4 * w_sc, min(w_sc, 0.99 * w_ci)], method='brentq')
            if sol.converged:
                w_pl_sol = sol.root
                n2_R = 1.0 - (w_pe**2 / (w_pl_sol * (w_pl_sol - w_ce * cos_theta))) - (w_pi**2 / (w_pl_sol * (w_pl_sol + w_ci)))
                k_val = (w_pl_sol / c) * np.sqrt(n2_R)
                k_disp_list.append(k_val)
            else:
                k_disp_list.append(np.nan)
        except Exception:
            k_disp_list.append(np.nan)

    return np.array(k_disp_list)
    
def compute_in_situ_f_RC(v_sc_array, r_au_array=None):
    """
    Computes dynamic in-situ preamplifier RC corner frequency with 1/r^2 EUV photoflux scaling.
    
    Mathematical Formulation:
        I_ph0(r)     = I_ph0_1AU / max(r_AU, 0.04)^2
        R_sh(V_sc,r) = (T_ph_eV / I_ph0(r)) * exp(clip(V_sc, 0, 15) / T_ph_eV)
        f_RC(V_sc,r) = 1.0 / (2 * pi * R_sh * C_stray)
        
    Physics:
    - T_ph_eV   = 1.5 eV (characteristic photoelectron thermal energy scale)
    - I_ph0_1AU = 10 uA (photoelectron saturation current at 1 AU)
    - C_stray   = 36 pF (preamplifier parasitic input stray capacitance)
    - r_AU      = Heliocentric distance in Astronomical Units (scales photoflux as 1/r^2)
    - V_sc      = Spacecraft potential [V], clamped at V_sc <= 0V (no photoelectron trapping)
    
    Parameters:
        v_sc_array : Array of spacecraft potentials [V]
        r_au_array : Optional array of heliocentric distances [AU]. Defaults to 1.0 AU if None.
        
    Returns:
        f_RC       : In-situ low-pass corner frequency [Hz], lower-bounded at 1e-3 Hz.
    """
    T_ph_eV = 1.5       # Photoelectron thermal scale [eV]
    I_ph0_1AU = 10e-6   # Saturation photoelectron current at 1 AU [A]
    C_stray = 36e-12    # Preamplifier parasitic stray capacitance [F]

    v_sc_clean = np.clip(np.nan_to_num(v_sc_array, nan=0.0), 0.0, 15.0)
    
    if r_au_array is not None:
        r_au_clean = np.clip(np.nan_to_num(r_au_array, nan=1.0), 0.04, 1.2)
    else:
        r_au_clean = np.ones_like(v_sc_clean)

    # Heliocentric EUV irradiance scaling: I_ph0(r) = I_ph0_1AU / r_AU^2
    I_ph0_r = I_ph0_1AU / (r_au_clean ** 2)
    
    # Sheath resistance R_sh [Ohms]
    R_sh = (T_ph_eV / I_ph0_r) * np.exp(v_sc_clean / T_ph_eV)
    f_RC = 1.0 / (2.0 * np.pi * R_sh * C_stray)
    
    return np.maximum(f_RC, 1e-3)

# =========================================================================
# ANALYTICAL SURFACE FITTING MODELS
# =========================================================================

def analytical_leff_model_noVsc(X, A, alpha, f0):
    """
    Model 1: 2-Variable Analytical Debye Surface Model
    
    Mathematical Formulation:
        L_eff(f, lambda_D) = L_phys + A * (lambda_D ** alpha) * exp(-f / f0)
        
    Parameters:
        X        : Tuple of (f_weighted_Hz, debye_length_m)
        A        : Debye amplitude boost factor [m]
        alpha    : Debye power-law scaling exponent
        f0       : Characteristic e-folding frequency cutoff [Hz]
    """
    f, debye = X
    L_phys = 3.5
    return L_phys + A * (debye ** alpha) * np.exp(-f / f0)


def analytical_leff_model_3d(X, A, alpha, b, beta, f0):
    """
    Model 3: 3-Variable Analytical Debye & Voltage Surface Model
    
    Mathematical Formulation:
        Sheath = max(1.0 + b * V_sc, 1e-3)
        L_eff(f, lambda_D, V_sc) = L_phys + A * (lambda_D ** alpha) * (Sheath ** beta) * exp(-f / f0)
        
    Parameters:
        X        : Tuple of (f_weighted_Hz, debye_length_m, V_sc_volts)
        A        : Base amplitude boost [m]
        alpha    : Debye length scaling exponent
        b        : Spacecraft potential linear coupling factor [V^-1]
        beta     : Sheath potential exponent
        f0       : Frequency roll-off scale [Hz]
    """
    f, debye, v_sc = X
    L_phys = 3.5
    sheath_factor = np.maximum(1.0 + b * v_sc, 1e-3)
    return L_phys + A * (debye ** alpha) * (sheath_factor ** beta) * np.exp(-f / f0)


def analytical_leff_model_f_vsc(X, A, b, beta, f0):
    """
    Model 2: Streamlined Spacecraft Potential Model
    
    Mathematical Formulation:
        Sheath = max(1.0 + b * V_sc, 1e-3)
        L_eff(f, V_sc) = L_phys + A * (Sheath ** beta) * exp(-f / f0)
        
    Parameters:
        X        : Tuple of (f_weighted_Hz, V_sc_volts)
        A        : Voltage expansion amplitude [m]
        b        : Potential coupling coefficient [V^-1]
        beta     : Sheath exponent
        f0       : Exponential roll-off frequency [Hz]
    """
    f, v_sc = X
    L_phys = 3.5
    sheath_factor = np.maximum(1.0 + b * v_sc, 1e-3)
    return L_phys + A * (sheath_factor ** beta) * np.exp(-f / f0)


def analytical_leff_model_physics_bandpass(X, A, V0, f0, alpha, beta):
    """
    Model 4: Physics-Based Sheath Bandpass Model
    
    Mathematical Formulation:
        V_norm = max(V_sc / V0, 1e-5)
        f_norm = max(f / f0, 1e-5)
        V_factor = (V_norm * exp(1.0 - V_norm)) ** alpha
        f_factor = (f_norm * exp(1.0 - f_norm)) ** beta
        
        L_eff(f, V_sc) = L_phys + A * V_factor * f_factor
        
    Parameters:
        X        : Tuple of (f_weighted_Hz, V_sc_volts)
        A        : Resonant peak effective length boost [m]
        V0       : Characteristic sheath turnaround potential [V]
        f0       : Peak bandpass resonance frequency [Hz]
        alpha    : Spacecraft potential asymmetric skewness
        beta     : Wave frequency asymmetric skewness
    """
    f, v_sc = X
    L_phys = 3.5

    v_norm = np.maximum(v_sc / np.maximum(V0, 1e-3), 1e-5)
    f_norm = np.maximum(f / np.maximum(f0, 1e-3), 1e-5)

    v_factor = (v_norm * np.exp(1.0 - v_norm)) ** alpha
    f_factor = (f_norm * np.exp(1.0 - f_norm)) ** beta

    return L_phys + A * v_factor * f_factor


def analytical_leff_model_physics_wavepower(X, L_base, A, V0, f0, B0, alpha, beta, gamma):
    """
    Model 5: Physics Bandpass + Magnetic Wave Power Damping
    
    Mathematical Formulation:
        V_norm = max(V_sc / V0, 1e-5)
        f_norm = max(f / f0, 1e-5)
        B_norm = max(delta_B / B0, 0.0)
        
        Damping  = 1.0 / (1.0 + (B_norm ** gamma))
        L_eff(f, V_sc, delta_B) = L_base + A * V_factor * f_factor * Damping
        
    Parameters:
        X        : Tuple of (f_weighted_Hz, V_sc_volts, B_wave_nT)
        L_base   : Un-expanded baseline antenna length [m]
        A        : Maximum photoelectron expansion boost [m]
        V0       : Peak expansion spacecraft potential [V]
        f0       : Peak expansion wave frequency [Hz]
        B0       : Magnetic rectification damping threshold [nT]
        alpha    : Voltage bandpass skewness exponent
        beta     : Frequency bandpass skewness exponent
        gamma    : Non-linear magnetic damping exponent
    """
    f, v_sc, delta_B = X

    v_norm = np.maximum(v_sc / np.maximum(V0, 1e-3), 1e-5)
    f_norm = np.maximum(f / np.maximum(f0, 1e-3), 1e-5)
    b_norm = np.maximum(delta_B / np.maximum(B0, 1e-3), 0.0)

    v_factor = (v_norm * np.exp(1.0 - v_norm)) ** alpha
    f_factor = (f_norm * np.exp(1.0 - f_norm)) ** beta
    damping_factor = 1.0 / (1.0 + (b_norm ** gamma))

    return L_base + A * v_factor * f_factor * damping_factor


def analytical_leff_model6_physics(X, L_base, A, V0, B0, alpha, beta, gamma):
    """
    Model 6: Power-Law Roll-Off Anchored to Baseline f_RC + AC Sheath Damping
    
    Mathematical Formulation:
        f_RC0    = 14.74 Hz  (Baseline preamplifier cutoff at V_sc = 0V)
        f_norm   = max(f / f_RC0, 1e-5)
        f_factor = 1.0 / (1.0 + (f_norm ** beta))
        
        V_norm   = max(V_sc / V0, 1e-5)
        V_factor = (V_norm * exp(1.0 - V_norm)) ** alpha
        
        B_norm   = max(delta_B_perp / B0, 0.0)
        Damping  = 1.0 / (1.0 + (B_norm ** gamma))
        
        L_eff(f, V_sc, delta_B_perp, f_RC) = L_base + A * f_factor * V_factor * Damping
        
    Parameters:
        X            : Tuple of (f_weighted_Hz, V_sc_volts, B_perp_nT, f_RC_corner_Hz)
        L_base       : Asymptotic baseline physical dipole length [m]
        A            : Sheath expansion magnitude [m]
        V0           : Peak potential turnaround threshold [V]
        B0           : Perpendicular wave magnetic damping threshold [nT]
        alpha        : Spacecraft potential skewness exponent
        beta         : High-frequency RC attenuation power-law exponent
        gamma        : Magnetic sheath suppression exponent (or fixed canonical = 2.0)
    """
    f, v_sc, delta_B_perp, f_RC = X

    # Normalize to baseline f_RC0 (minimum R_sh at V_sc = 0V => f_RC0 ~ 14.74 Hz)
    f_RC0 = 14.74
    f_norm = np.maximum(f / f_RC0, 1e-5)
    f_factor = 1.0 / (1.0 + f_norm**beta)

    v_norm = np.maximum(v_sc / np.maximum(V0, 1e-3), 1e-5)
    v_factor = (v_norm * np.exp(1.0 - v_norm)) ** alpha

    b_norm = np.maximum(delta_B_perp / np.maximum(B0, 1e-3), 0.0)
    damping_factor = 1.0 / (1.0 + (b_norm ** gamma))

    return L_base + A * f_factor * v_factor * damping_factor


def analytical_leff_model7_physics(X, L_base, A, V0, f0, V_wave0, alpha, beta, gamma):
    """
    Model 7: Physics Bandpass + Direct Raw Terminal AC Voltage Damping
    
    Mathematical Formulation:
        V_norm      = max(V_sc / V0, 1e-5)
        f_norm      = max(f / f0, 1e-5)
        V_wave_norm = max(delta_V_raw / V_wave0, 0.0)
        
        V_factor    = (V_norm * exp(1.0 - V_norm)) ** alpha
        f_factor    = (f_norm * exp(1.0 - f_norm)) ** beta
        Damping     = 1.0 / (1.0 + (V_wave_norm ** gamma))
        
        L_eff(f, V_sc, delta_V_raw) = L_base + A * V_factor * f_factor * Damping
        
    Parameters:
        X            : Tuple of (f_weighted_Hz, V_sc_volts, V_wave_volts)
        L_base       : Baseline physical dipole length [m]
        A            : Sheath expansion boost [m]
        V0           : Peak potential turnaround threshold [V]
        f0           : Peak resonance frequency [Hz]
        V_wave0      : AC preamplifier terminal voltage rectification threshold [V]
        alpha        : Spacecraft potential skewness exponent
        beta         : Frequency skewness exponent
        gamma        : Super-linear AC sheath collapse exponent
    """
    f, v_sc, delta_V = X

    v_norm = np.maximum(v_sc / np.maximum(V0, 1e-3), 1e-5)
    f_norm = np.maximum(f / np.maximum(f0, 1e-3), 1e-5)
    v_wave_norm = np.maximum(delta_V / np.maximum(V_wave0, 1e-6), 0.0)

    v_factor = (v_norm * np.exp(1.0 - v_norm)) ** alpha
    f_factor = (f_norm * np.exp(1.0 - f_norm)) ** beta
    
    damping_factor = 1.0 / (1.0 + (v_wave_norm ** gamma))

    return L_base + A * v_factor * f_factor * damping_factor
    
def analytical_leff_model8_wake(X, L_base, A_iso, A_wake, V0, B0, alpha, beta, gamma):
    """
    Model 8: EUV Photoflux (1/r^2) + Additive Plasma Wake-Aligned AC Sheath Damping
    
    Mathematical Formulation:
        f_norm    = max(f / f_RC(V_sc, r_AU), 1e-5)
        f_factor  = 1.0 / (1.0 + (f_norm ** beta))
        
        V_norm    = max(V_sc / V0, 1e-5)
        V_factor  = (V_norm * exp(1.0 - V_norm)) ** alpha
        
        B_norm    = max(delta_B_perp / B0, 0.0)
        Damping   = 1.0 / (1.0 + (B_norm ** gamma))
        
        A_total   = A_iso + A_wake * sin^2(theta_BX)
        
        L_eff(f, V_sc, delta_B_perp, f_RC, theta_BX) = L_base + A_total * f_factor * V_factor * Damping
        
    Parameters:
        X              : Tuple of (f_weighted_Hz, V_sc_volts, B_perp_nT, f_RC_corner_Hz, sin2_theta_bx)
        L_base         : Asymptotic baseline physical dipole length [m]
        A_iso          : Isotropic photoelectron sheath expansion boost [m]
        A_wake         : Plasma wake cross-field anisotropy expansion boost [m]
        V0             : Sheath potential turnaround threshold [V]
        B0             : Perpendicular wave magnetic damping threshold [nT]
        alpha          : Spacecraft potential asymmetric skewness exponent
        beta           : High-frequency RC attenuation power-law exponent
        gamma          : AC sheath magnetic suppression exponent
    """
    f, v_sc, delta_B_perp, f_RC, sin2_theta_bx = X

    f_norm = np.maximum(f / np.maximum(f_RC, 1e-3), 1e-5)
    f_factor = 1.0 / (1.0 + f_norm**beta)

    v_norm = np.maximum(v_sc / np.maximum(V0, 1e-3), 1e-5)
    v_factor = (v_norm * np.exp(1.0 - v_norm)) ** alpha

    b_norm = np.maximum(delta_B_perp / np.maximum(B0, 1e-3), 0.0)
    damping_factor = 1.0 / (1.0 + (b_norm ** gamma))
    
    a_total = A_iso + A_wake * sin2_theta_bx

    return L_base + a_total * f_factor * v_factor * damping_factor
    
def analytical_leff_model9_tanh(X, L_base, A_iso, A_wake, V0, f0, Delta, B0, alpha, gamma):
    """
    Model 9: EUV Flux + Wake Anisotropy + Sigmoidal Tanh Frequency Step (Mondal+ 2026)
    
    Mathematical Formulation:
        f_norm   = log10(f / f0) / Delta
        f_factor = 0.5 * (1.0 + tanh(f_norm))
        V_norm   = max(V_sc / V0, 1e-5)
        V_factor = (V_norm * exp(1.0 - V_norm)) ** alpha
        B_norm   = max(delta_B_perp / B0, 0.0)
        Damping  = 1.0 / (1.0 + (B_norm ** gamma))
        
        L_eff = L_base + (A_iso + A_wake * sin2_theta_bx) * f_factor * V_factor * Damping
    """
    f, v_sc, delta_B_perp, sin2_theta_bx = X

    # Sigmoidal log-frequency transition centered at knee frequency f0
    f_log_ratio = np.log10(np.maximum(f, 1e-3) / np.maximum(f0, 1e-3))
    f_factor = 0.5 * (1.0 + np.tanh(f_log_ratio / np.maximum(Delta, 0.05)))

    v_norm = np.maximum(v_sc / np.maximum(V0, 1e-3), 1e-5)
    v_factor = (v_norm * np.exp(1.0 - v_norm)) ** alpha

    b_norm = np.maximum(delta_B_perp / np.maximum(B0, 1e-3), 0.0)
    damping_factor = 1.0 / (1.0 + (b_norm ** gamma))

    a_total = A_iso + A_wake * sin2_theta_bx

    return L_base + a_total * f_factor * v_factor * damping_factor
    
def prepare_fit_dataset(df_clean, required_cols):
    if not all(col in df_clean.columns for col in required_cols):
        missing = [c for c in required_cols if c not in df_clean.columns]
        print(f"    [!] Fit aborted: Missing required column(s): {missing}")
        return None

    valid_mask = np.ones(len(df_clean), dtype=bool)
    for col in required_cols:
        valid_mask &= np.isfinite(df_clean[col].values)

    valid_mask &= (df_clean['f_weighted_Hz'].values > 0)
    valid_mask &= (df_clean['L_eff_m'].values > 0)

    df_fit = df_clean[valid_mask].copy()

    if len(df_fit) < 10:
        print(f"    [!] Fit aborted: Insufficient valid data points ({len(df_fit)} points).")
        return None

    return df_fit

def fit_leff_surface_noVsc(df_clean):
    """Fits 2-variable Model 1: L_eff(f, lambda_D)"""
    df_fit = prepare_fit_dataset(df_clean, ['f_weighted_Hz', 'debye_length_m', 'L_eff_m'])
    if df_fit is None: return None, None

    f_data = df_fit['f_weighted_Hz'].values
    d_data = df_fit['debye_length_m'].values
    l_data = df_fit['L_eff_m'].values

    param_names  = ['A',     'alpha', 'f_0']
    p0           = [ 5.0,     1.0,     15.0]
    lower_bounds = [ 0.0,     0.0,      1.0]
    upper_bounds = [50.0,     3.0,    100.0]

    try:
        popt, pcov = curve_fit(analytical_leff_model_noVsc, (f_data, d_data), l_data, p0=p0, bounds=(lower_bounds, upper_bounds))
        r_squared = 1 - (np.sum((l_data - analytical_leff_model_noVsc((f_data, d_data), *popt))**2) / np.sum((l_data - np.mean(l_data))**2))
        
        print("\n" + "="*65)
        print("     2-VARIABLE ANALYTICAL L_eff(f, λ_D) FIT RESULTS")
        print("="*65)
        print(f" Evaluated Points           : {len(df_fit):,} / {len(df_clean):,}")
        print(f"   -> R-squared (R²)        : {r_squared:.4f}")
        print("="*65 + "\n")
        return popt, r_squared
    except Exception as e:
        print(f"    [!] 2D Surface fitting failed: {e}")
        return None, None


def fit_leff_surface_f_vsc(df_clean):
    """Fits Streamlined Model 2: L_eff(f, V_sc)"""
    df_fit = prepare_fit_dataset(df_clean, ['f_weighted_Hz', 'V_sc_volts', 'L_eff_m'])
    if df_fit is None: return None, None

    f_data = df_fit['f_weighted_Hz'].values
    v_data = df_fit['V_sc_volts'].values
    l_data = df_fit['L_eff_m'].values

    param_names  = ['A',     'b',     'beta',  'f_0']
    p0           = [ 1.5,    30.0,     0.45,   20.0]
    lower_bounds = [ 0.0,     0.0,     0.00,    1.0]
    upper_bounds = [50.0,   100.0,     2.00,  100.0]

    try:
        popt, pcov = curve_fit(analytical_leff_model_f_vsc, (f_data, v_data), l_data, p0=p0, bounds=(lower_bounds, upper_bounds))
        r_squared = 1 - (np.sum((l_data - analytical_leff_model_f_vsc((f_data, v_data), *popt))**2) / np.sum((l_data - np.mean(l_data))**2))
        
        print("\n" + "="*65)
        print("     STREAMLINED L_eff(f, V_sc) FIT RESULTS")
        print("="*65)
        print(f" Evaluated Points           : {len(df_fit):,} / {len(df_clean):,}")
        print(f"   -> R-squared (R²)        : {r_squared:.4f}")
        print("="*65 + "\n")
        return popt, r_squared
    except Exception as e:
        print(f"    [!] L_eff(f, V_sc) fitting failed: {e}")
        return None, None


def fit_leff_surface_3d(df_clean):
    """Fits 3-Variable Model 3: L_eff(f, lambda_D, V_sc)"""
    df_fit = prepare_fit_dataset(df_clean, ['f_weighted_Hz', 'debye_length_m', 'V_sc_volts', 'L_eff_m'])
    if df_fit is None: return None, None

    f_data = df_fit['f_weighted_Hz'].values
    d_data = df_fit['debye_length_m'].values
    v_data = df_fit['V_sc_volts'].values
    l_data = df_fit['L_eff_m'].values

    param_names  = ['A',     'alpha', 'b',     'beta',  'f_0']
    p0           = [ 5.0,     1.0,     0.1,     0.5,    15.0]
    lower_bounds = [ 0.0,     0.0,     0.0,     0.0,     1.0]
    upper_bounds = [50.0,     3.0,    50.0,     2.0,   100.0]

    try:
        popt, pcov = curve_fit(analytical_leff_model_3d, (f_data, d_data, v_data), l_data, p0=p0, bounds=(lower_bounds, upper_bounds))
        r_squared = 1 - (np.sum((l_data - analytical_leff_model_3d((f_data, d_data, v_data), *popt))**2) / np.sum((l_data - np.mean(l_data))**2))
        
        print("\n" + "="*65)
        print("     3-VARIABLE ANALYTICAL L_eff(f, λ_D, V_sc) FIT RESULTS")
        print("="*65)
        print(f" Evaluated Points           : {len(df_fit):,} / {len(df_clean):,}")
        print(f"   -> R-squared (R²)        : {r_squared:.4f}")
        print("="*65 + "\n")
        return popt, r_squared
    except Exception as e:
        print(f"    [!] 3D Surface fitting failed: {e}")
        return None, None


def fit_leff_surface_physics_bandpass(df_clean):
    """Fits Model 4: Physics-Based Bandpass L_eff(f, V_sc)"""
    df_fit = prepare_fit_dataset(df_clean, ['f_weighted_Hz', 'V_sc_volts', 'L_eff_m'])
    if df_fit is None: return None, None

    f_data = df_fit['f_weighted_Hz'].values
    v_data = df_fit['V_sc_volts'].values
    l_data = df_fit['L_eff_m'].values

    param_names  = ['A',     'V_0',   'f_0',   'alpha', 'beta']
    p0           = [12.0,     6.0,     7.0,     1.0,     1.0]
    lower_bounds = [ 0.0,     1.0,     1.0,     0.1,     0.1]
    upper_bounds = [50.0,    20.0,    30.0,     5.0,     5.0]

    try:
        popt, pcov = curve_fit(analytical_leff_model_physics_bandpass, (f_data, v_data), l_data, p0=p0, bounds=(lower_bounds, upper_bounds))
        r_squared = 1 - (np.sum((l_data - analytical_leff_model_physics_bandpass((f_data, v_data), *popt))**2) / np.sum((l_data - np.mean(l_data))**2))
        
        print("\n" + "="*65)
        print("   PHYSICS-BASED BANDPASS L_eff(f, V_sc) FIT RESULTS")
        print("="*65)
        print(rf" Evaluated Points           : {len(df_fit):,} / {len(df_clean):,}")
        print(rf"   -> R-squared (R²)        : {r_squared:.4f}")
        print("="*65 + "\n")
        return popt, r_squared
    except Exception as e:
        print(rf"    [!] Physics bandpass fitting failed: {e}")
        return None, None


def fit_leff_surface_physics_wavepower(df_clean):
    """Fits Model 5: Bandpass + Non-Linear Magnetic Wave Damping"""
    df_fit = prepare_fit_dataset(df_clean, ['f_weighted_Hz', 'V_sc_volts', 'B_wave_nT', 'L_eff_m'])
    if df_fit is None: return None, None

    f_data = df_fit['f_weighted_Hz'].values
    v_data = df_fit['V_sc_volts'].values
    b_data = df_fit['B_wave_nT'].values
    l_data = df_fit['L_eff_m'].values

    param_names  = ['L_base', 'A',    'V_0',  'f_0',  'B_0',  'alpha', 'beta',  'gamma']
    p0           = [ 3.5,     12.0,    4.8,    2.0,    1.5,    1.0,     0.1,     1.0]
    lower_bounds = [ 0.5,      0.0,    1.0,    0.1,    0.1,    0.1,     0.01,    0.1]
    upper_bounds = [ 6.0,     50.0,   20.0,   30.0,   20.0,    5.0,     2.0,     5.0]

    try:
        popt, pcov = curve_fit(analytical_leff_model_physics_wavepower, (f_data, v_data, b_data), l_data, p0=p0, bounds=(lower_bounds, upper_bounds))
        perr = np.sqrt(np.diag(pcov))
        r_squared = 1 - (np.sum((l_data - analytical_leff_model_physics_wavepower((f_data, v_data, b_data), *popt))**2) / np.sum((l_data - np.mean(l_data))**2))
        
        print("\n" + "="*65)
        print("   MODEL 5: BANDPASS + WAVE DAMPING (FREE L_base) FIT RESULTS")
        print("="*65)
        print(f" Evaluated Points           : {len(df_fit):,} / {len(df_clean):,}")
        print(f"   -> Baseline Length (L_base): {popt[0]:.3f} ± {perr[0]:.3f} m")
        print(f"   -> Amplitude Boost (A)     : {popt[1]:.3f} ± {perr[1]:.3f} m")
        print(f"   -> Peak Potential (V_0)    : {popt[2]:.3f} ± {perr[2]:.3f} V")
        print(f"   -> Peak Frequency (f_0)    : {popt[3]:.3f} ± {perr[3]:.3f} Hz")
        print(f"   -> Rectification B_0       : {popt[4]:.3f} ± {perr[4]:.3f} nT")
        print(f"   -> Potential Skew (α)      : {popt[5]:.3f} ± {perr[5]:.3f}")
        print(f"   -> Frequency Skew (β)      : {popt[6]:.3f} ± {perr[6]:.3f}")
        print(f"   -> Damping Exponent (γ)    : {popt[7]:.3f} ± {perr[7]:.3f}")
        print(f"   -> R-squared (R²)          : {r_squared:.4f}")
        print("="*65 + "\n")
        return popt, r_squared
    except Exception as e:
        print(f"    [!] Model 5 fitting failed: {e}")
        return None, None


def fit_leff_surface_model6(df_clean):
    """
    Fits 7-parameter Model 6 incorporating deterministic preamplifier RC corner frequency
    f_RC(V_sc, r_AU) roll-off and non-linear broadband magnetic AC sheath damping.

    Mathematical Formulation:
        f_norm   = max(f / f_RC0, 1e-5)
        f_factor = 1.0 / (1.0 + (f_norm ** beta))
        V_norm   = max(V_sc / V0, 1e-5)
        V_factor = (V_norm * exp(1.0 - V_norm)) ** alpha
        B_norm   = max(delta_B_high_freq / B0, 0.0)
        Damping  = 1.0 / (1.0 + (B_norm ** gamma))

        L_eff(f, V_sc, B_high_freq) = L_base + A * f_factor * V_factor * Damping

    Physics & Coupling:
        - f_RC0    = 14.74 Hz (Baseline preamplifier cutoff at V_sc = 0V)
        - f_RC     = Anchored directly to preamplifier input stray capacitance (C_stray = 36 pF)
        - B_high   = Strictly requires broadband rectifying magnetic power (f > f_RC)

    Strict Physics Requirements:
        Requires 'B_high_freq_nT'. Aborts cleanly if missing rather than using narrowband fallbacks.

    Parameters:
        df_clean : Pre-filtered calibration dataframe.

    Returns:
        popt      : Fitted 7-parameter array [L_base, A, V_0, B_0, alpha, beta, gamma].
        r_squared : Calculated coefficient of determination (R^2).
    """
    df_work = df_clean.copy()

    # Strict physical column check - NO fallback to B_wave_nT
    if 'B_high_freq_nT' not in df_work.columns:
        print("    [!] Model 6 fit aborted: Missing required column 'B_high_freq_nT'.")
        return None, None

    if 'f_RC_corner_Hz' not in df_work.columns or df_work['f_RC_corner_Hz'].isna().all():
        r_col = next((c for c in ['r_au', 'r_AU', 'radial_distance_AU'] if c in df_work.columns), None)
        r_vals = df_work[r_col].values if r_col else None
        df_work['f_RC_corner_Hz'] = compute_in_situ_f_RC(df_work['V_sc_volts'].values, r_vals)

    req_cols = ['f_weighted_Hz', 'V_sc_volts', 'B_high_freq_nT', 'L_eff_m', 'f_RC_corner_Hz']
    df_fit = prepare_fit_dataset(df_work, req_cols)
    if df_fit is None: return None, None

    f_data   = df_fit['f_weighted_Hz'].values
    v_data   = df_fit['V_sc_volts'].values
    l_data   = df_fit['L_eff_m'].values
    frc_data = df_fit['f_RC_corner_Hz'].values
    
    raw_b    = df_fit['B_high_freq_nT'].values
    b_data   = raw_b * np.sin(np.radians(np.nan_to_num(df_fit['theta_kB_deg'].values, nan=90.0))) if 'theta_kB_deg' in df_fit.columns else raw_b

    param_names  = ['L_base', 'A',    'V_0',  'B_0',  'alpha', 'beta',   'gamma']
    p0           = [ 1.70,     20.0,   5.20,   3.0,    0.85,    1.50,       1.85]
    lower_bounds = [ 0.50,      1.0,   1.00,   0.10,   0.01,    0.050,      0.01]
    upper_bounds = [ 5.00,    135.0,  15.00,  15.0,    3.00,   10.0,        4.00]

    try:
        popt, pcov = curve_fit(analytical_leff_model6_physics, (f_data, v_data, b_data, frc_data), l_data, p0=p0, bounds=(lower_bounds, upper_bounds))
        perr = np.sqrt(np.diag(pcov))
        r_squared = 1 - (np.sum((l_data - analytical_leff_model6_physics((f_data, v_data, b_data, frc_data), *popt))**2) / np.sum((l_data - np.mean(l_data))**2))
        warnings_list = check_pegged_parameters(popt, lower_bounds, upper_bounds, param_names)
        
        print("\n" + "="*65)
        print("   MODEL 6: POWER-LAW ROLL-OFF (f_RC Anchored) FIT RESULTS")
        print("="*65)
        print(f" Evaluated Points           : {len(df_fit):,} / {len(df_clean):,}")
        print(f"   -> Baseline Length (L_base): {popt[0]:.3f} ± {perr[0]:.3f} m")
        print(f"   -> Expansion Boost (A)     : {popt[1]:.3f} ± {perr[1]:.3f} m")
        print(f"   -> Peak Potential (V_0)    : {popt[2]:.3f} ± {perr[2]:.3f} V")
        print(f"   -> Rectification B_0       : {popt[3]:.3f} ± {perr[3]:.3f} nT")
        print(f"   -> Potential Skew (α)      : {popt[4]:.3f} ± {perr[4]:.3f}")
        print(f"   -> Frequency Exponent (β)  : {popt[5]:.3f} ± {perr[5]:.3f}")
        print(f"   -> Damping Exponent (γ)    : {popt[6]:.3f} ± {perr[6]:.3f}")
        print(f"   -> R-squared (R²)          : {r_squared:.4f}")
        if warnings_list:
            print("-" * 65)
            for w in warnings_list: print(w)
        print("="*65 + "\n")
        return popt, r_squared
    except Exception as e:
        print(f"    [!] Model 6 fitting failed: {e}")
        return None, None


def fit_leff_surface_model7(df_clean):
    """Fits 8-parameter Model 7 using raw terminal AC voltage (V_wave_volts)"""
    req_cols = ['f_weighted_Hz', 'V_sc_volts', 'V_wave_volts', 'L_eff_m']
    df_fit = prepare_fit_dataset(df_clean, req_cols)
    if df_fit is None: return None, None

    f_data  = df_fit['f_weighted_Hz'].values
    v_data  = df_fit['V_sc_volts'].values
    vw_data = df_fit['V_wave_volts'].values
    l_data  = df_fit['L_eff_m'].values

    param_names  = ['L_base', 'A',    'V_0',  'f_0',  'V_wave0', 'alpha', 'beta',  'gamma']
    p0           = [ 1.60,     15.0,   4.75,   0.5,    0.057,     0.65,   0.008,   2.30]
    lower_bounds = [ 0.50,      1.0,   1.00,   0.01,   1e-5,      0.01,   0.0001,  0.01]
    upper_bounds = [ 5.00,    100.0,  15.00,  50.0,    0.20,      3.00,   2.00,    5.00]

    try:
        popt, pcov = curve_fit(analytical_leff_model7_physics, (f_data, v_data, vw_data), l_data, p0=p0, bounds=(lower_bounds, upper_bounds))
        perr = np.sqrt(np.diag(pcov))

        residuals = l_data - analytical_leff_model7_physics((f_data, v_data, vw_data), *popt)
        r_squared = 1 - (np.sum(residuals**2) / np.sum((l_data - np.mean(l_data))**2))
        warnings_list = check_pegged_parameters(popt, lower_bounds, upper_bounds, param_names)

        print("\n" + "="*65)
        print("   MODEL 7: BANDPASS + RAW TERMINAL VOLTAGE DAMPING FIT RESULTS")
        print("="*65)
        print(f" Evaluated Points           : {len(df_fit):,} / {len(df_clean):,}")
        print(f"   -> Baseline Length (L_base): {popt[0]:.3f} ± {perr[0]:.3f} m")
        print(f"   -> Expansion Boost (A)     : {popt[1]:.3f} ± {perr[1]:.3f} m")
        print(f"   -> Peak Potential (V_0)    : {popt[2]:.3f} ± {perr[2]:.3f} V")
        print(f"   -> Peak Frequency (f_0)    : {popt[3]:.3f} ± {perr[3]:.3f} Hz")
        print(f"   -> Rectification V_wave0   : {popt[4]:.5f} ± {perr[4]:.5f} V ({popt[4]*1000:.2f} mV)")
        print(f"   -> Potential Skew (α)      : {popt[5]:.3f} ± {perr[5]:.3f}")
        print(f"   -> Frequency Skew (β)      : {popt[6]:.3f} ± {perr[6]:.3f}")
        print(f"   -> Damping Exponent (γ)    : {popt[7]:.3f} ± {perr[7]:.3f}")
        print(f"   -> R-squared (R²)          : {r_squared:.4f}")
        if warnings_list:
            print("-" * 65)
            for w in warnings_list: print(w)
        print("="*65 + "\n")

        return popt, r_squared
    except Exception as e:
        print(f"    [!] Model 7 fitting failed: {e}")
        return None, None
    
def fit_leff_surface_model8(df_clean):
    """
    Fits 8-parameter Model 8 incorporating 1/r^2 EUV photoflux scaling, additive
    plasma wake alignment anisotropy (sin^2 theta_BX), and broadband magnetic AC sheath damping.

    Mathematical Formulation:
        f_norm    = max(f / f_RC(V_sc, r_AU), 1e-5)
        f_factor  = 1.0 / (1.0 + (f_norm ** beta))
        V_norm    = max(V_sc / V0, 1e-5)
        V_factor  = (V_norm * exp(1.0 - V_norm)) ** alpha
        B_norm    = max(delta_B_high_freq / B0, 0.0)
        Damping   = 1.0 / (1.0 + (B_norm ** gamma))
        A_total   = A_iso + A_wake * sin^2(theta_BX)

        L_eff(f, V_sc, B_high_freq, f_RC, theta_BX) = L_base + A_total * f_factor * V_factor * Damping

    Physics & Anisotropy:
        - r_AU     = Heliocentric distance scaling EUV photoflux (I_ph0 ~ 1/r_AU^2)
        - theta_BX = Angle between background B-field and spacecraft +X axis
        - A_wake   = Anisotropic sheath expansion/compression inside plasma wake

    Strict Physics Requirements:
        Requires 'r_au', 'theta_BX_deg', and 'B_high_freq_nT'. Aborts cleanly without fallbacks.

    Parameters:
        df_clean : Pre-filtered calibration dataframe.

    Returns:
        popt      : Fitted 8-parameter array [L_base, A_iso, A_wake, V_0, B_0, alpha, beta, gamma].
        r_squared : Calculated coefficient of determination (R^2).
    """
    df_work = df_clean.copy()

    # Strict physical column verification - NO fallbacks
    r_col = next((c for c in ['r_au', 'r_AU', 'radial_distance_AU'] if c in df_work.columns), None)
    bx_col = next((c for c in ['theta_BX_deg', 'theta_bx_deg'] if c in df_work.columns), None)
    
    missing_cols = []
    if not r_col: missing_cols.append('r_au')
    if not bx_col: missing_cols.append('theta_BX_deg')
    if 'B_high_freq_nT' not in df_work.columns: missing_cols.append('B_high_freq_nT')

    if missing_cols:
        print(f"    [!] Model 8 fit aborted: Missing required physical input column(s): {missing_cols}")
        return None, None

    df_work['f_RC_corner_Hz'] = compute_in_situ_f_RC(df_work['V_sc_volts'].values, df_work[r_col].values)
    df_work['sin2_theta_bx'] = np.sin(np.radians(df_work[bx_col].values))**2

    req_cols = ['f_weighted_Hz', 'V_sc_volts', 'B_high_freq_nT', 'L_eff_m', 'f_RC_corner_Hz', 'sin2_theta_bx']
    df_fit = prepare_fit_dataset(df_work, req_cols)
    if df_fit is None: return None, None

    f_data   = df_fit['f_weighted_Hz'].values
    v_data   = df_fit['V_sc_volts'].values
    l_data   = df_fit['L_eff_m'].values
    frc_data = df_fit['f_RC_corner_Hz'].values
    sbx_data = df_fit['sin2_theta_bx'].values
    
    raw_b    = df_fit['B_high_freq_nT'].values
    b_data   = raw_b * np.sin(np.radians(np.nan_to_num(df_fit['theta_kB_deg'].values, nan=90.0))) if 'theta_kB_deg' in df_fit.columns else raw_b
    
    param_names  = ['L_base', 'A_iso', 'A_wake', 'V_0',  'B_0',  'alpha', 'beta',  'gamma']
    p0           = [ 1.80,     20.0,    5.0,      3.8,    0.10,   0.55,    0.15,    0.40]
    lower_bounds = [ 0.50,      0.0,   -20.0,     1.00,   0.01,   0.01,    0.01,    0.01]
    upper_bounds = [ 5.00,    100.0,    50.0,    15.00,  15.0,    3.00,   10.0,     4.00]

    try:
        popt, pcov = curve_fit(analytical_leff_model8_wake, (f_data, v_data, b_data, frc_data, sbx_data), l_data, p0=p0, bounds=(lower_bounds, upper_bounds))
        perr = np.sqrt(np.diag(pcov))
        r_squared = 1 - (np.sum((l_data - analytical_leff_model8_wake((f_data, v_data, b_data, frc_data, sbx_data), *popt))**2) / np.sum((l_data - np.mean(l_data))**2))
        warnings_list = check_pegged_parameters(popt, lower_bounds, upper_bounds, param_names)
        
        print("\n" + "="*65)
        print("   MODEL 8: EUV FLUX (1/r²) + ADDITIVE WAKE ALIGNMENT FIT")
        print("="*65)
        print(f" Evaluated Points           : {len(df_fit):,} / {len(df_clean):,}")
        print(f"   -> Baseline Length (L_base): {popt[0]:.3f} ± {perr[0]:.3f} m")
        print(f"   -> Isotropic Boost (A_iso) : {popt[1]:.3f} ± {perr[1]:.3f} m")
        print(f"   -> Wake Boost (A_wake)     : {popt[2]:.3f} ± {perr[2]:.3f} m")
        print(f"   -> Peak Potential (V_0)    : {popt[3]:.3f} ± {perr[3]:.3f} V")
        print(f"   -> Rectification B_0       : {popt[4]:.3f} ± {perr[4]:.3f} nT")
        print(f"   -> Potential Skew (α)      : {popt[5]:.3f} ± {perr[5]:.3f}")
        print(f"   -> Frequency Exponent (β)  : {popt[6]:.3f} ± {perr[6]:.3f}")
        print(f"   -> Damping Exponent (γ)    : {popt[7]:.3f} ± {perr[7]:.3f}")
        print(f"   -> R-squared (R²)          : {r_squared:.4f}")
        if warnings_list:
            print("-" * 65)
            for w in warnings_list: print(w)
        print("="*65 + "\n")
        return popt, r_squared
    except Exception as e:
        print(f"    [!] Model 8 fitting failed: {e}")
        return None, None
        
def fit_leff_surface_model9(df_clean):
    """
    Fits 9-parameter Model 9 incorporating Mondal et al. (2026) sigmoidal hyperbolic
    tangent transfer function in log-frequency space, EUV photoflux scaling,
    plasma wake alignment anisotropy, and broadband AC sheath damping.

    Mathematical Formulation:
        f_log_ratio = log10(max(f, 1e-3) / max(f0, 1e-3))
        f_factor    = 0.5 * (1.0 + tanh(f_log_ratio / max(Delta, 0.05)))
        V_norm      = max(V_sc / V0, 1e-5)
        V_factor    = (V_norm * exp(1.0 - V_norm)) ** alpha
        B_norm      = max(delta_B_high_freq / B0, 0.0)
        Damping     = 1.0 / (1.0 + (B_norm ** gamma))
        A_total     = A_iso + A_wake * sin^2(theta_BX)

        L_eff(f, V_sc, B_high_freq, theta_BX) = L_base + A_total * f_factor * V_factor * Damping

    Physics & Sigmoidal Transfer Function:
        - f_0   = Knee frequency (~10 Hz) marking transition from DC-resistive to AC-capacitive regime
        - Delta = Width of sigmoidal transition in log10 frequency space (decades)

    Strict Physics Requirements:
        Requires 'theta_BX_deg' and 'B_high_freq_nT'. Aborts cleanly without fallbacks.

    Parameters:
        df_clean : Pre-filtered calibration dataframe.

    Returns:
        popt      : Fitted 9-parameter array [L_base, A_iso, A_wake, V_0, f_0, Delta, B_0, alpha, gamma].
        r_squared : Calculated coefficient of determination (R^2).
    """
    df_work = df_clean.copy()

    # Strict physical column verification - NO fallbacks
    bx_col = next((c for c in ['theta_BX_deg', 'theta_bx_deg'] if c in df_work.columns), None)
    
    missing_cols = []
    if not bx_col: missing_cols.append('theta_BX_deg')
    if 'B_high_freq_nT' not in df_work.columns: missing_cols.append('B_high_freq_nT')

    if missing_cols:
        print(f"    [!] Model 9 fit aborted: Missing required physical input column(s): {missing_cols}")
        return None, None

    df_work['sin2_theta_bx'] = np.sin(np.radians(df_work[bx_col].values))**2

    req_cols = ['f_weighted_Hz', 'V_sc_volts', 'B_high_freq_nT', 'L_eff_m', 'sin2_theta_bx']
    df_fit = prepare_fit_dataset(df_work, req_cols)
    if df_fit is None: return None, None

    f_data   = df_fit['f_weighted_Hz'].values
    v_data   = df_fit['V_sc_volts'].values
    l_data   = df_fit['L_eff_m'].values
    sbx_data = df_fit['sin2_theta_bx'].values
    
    raw_b    = df_fit['B_high_freq_nT'].values
    b_data   = raw_b * np.sin(np.radians(np.nan_to_num(df_fit['theta_kB_deg'].values, nan=90.0))) if 'theta_kB_deg' in df_fit.columns else raw_b

    param_names  = ['L_base', 'A_iso', 'A_wake', 'V_0',  'f_0',   'Delta', 'B_0',  'alpha', 'gamma']
    p0           = [ 1.80,     20.0,    5.0,      3.8,   10.0,    0.50,    0.10,   0.55,    0.40]
    lower_bounds = [ 0.50,      0.0,   -20.0,     1.00,   1.0,    0.05,    0.01,   0.01,    0.01]
    upper_bounds = [ 5.00,    100.0,    50.0,    15.00,  50.0,    2.00,   15.0,    3.00,    4.00]

    try:
        popt, pcov = curve_fit(analytical_leff_model9_tanh, (f_data, v_data, b_data, sbx_data), l_data, p0=p0, bounds=(lower_bounds, upper_bounds))
        perr = np.sqrt(np.diag(pcov))
        r_squared = 1 - (np.sum((l_data - analytical_leff_model9_tanh((f_data, v_data, b_data, sbx_data), *popt))**2) / np.sum((l_data - np.mean(l_data))**2))
        warnings_list = check_pegged_parameters(popt, lower_bounds, upper_bounds, param_names)

        print("\n" + "="*65)
        print("   MODEL 9: TANH FREQUENCY STEP (Mondal+ 2026) FIT RESULTS")
        print("="*65)
        print(f" Evaluated Points           : {len(df_fit):,} / {len(df_clean):,}")
        print(f"   -> Baseline Length (L_base): {popt[0]:.3f} ± {perr[0]:.3f} m")
        print(f"   -> Isotropic Boost (A_iso) : {popt[1]:.3f} ± {perr[1]:.3f} m")
        print(f"   -> Wake Boost (A_wake)     : {popt[2]:.3f} ± {perr[2]:.3f} m")
        print(f"   -> Peak Potential (V_0)    : {popt[3]:.3f} ± {perr[3]:.3f} V")
        print(f"   -> Knee Frequency (f_0)    : {popt[4]:.3f} ± {perr[4]:.3f} Hz")
        print(f"   -> Log Width (Delta)       : {popt[5]:.3f} ± {perr[5]:.3f} decades")
        print(f"   -> Rectification B_0       : {popt[6]:.3f} ± {perr[6]:.3f} nT")
        print(f"   -> Potential Skew (α)      : {popt[7]:.3f} ± {perr[7]:.3f}")
        print(f"   -> Damping Exponent (γ)    : {popt[8]:.3f} ± {perr[8]:.3f}")
        print(f"   -> R-squared (R²)          : {r_squared:.4f}")
        if warnings_list:
            print("-" * 65)
            for w in warnings_list: print(w)
        print("="*65 + "\n")
        return popt, r_squared
    except Exception as e:
        print(f"    [!] Model 9 fitting failed: {e}")
        return None, None

def check_pegged_parameters(popt, lower_bounds, upper_bounds, param_names, tol=1e-3):
    pegged_warnings = []
    for name, val, lb, ub in zip(param_names, popt, lower_bounds, upper_bounds):
        if np.isclose(val, lb, atol=tol, rtol=tol):
            pegged_warnings.append(f"    [!] WARNING: Parameter '{name}' pegged at LOWER bound ({lb})")
        elif np.isclose(val, ub, atol=tol, rtol=tol):
            pegged_warnings.append(f"    [!] WARNING: Parameter '{name}' pegged at UPPER bound ({ub})")
    return pegged_warnings

def apply_adaptive_date_ticks(ax, df_times):
    if len(df_times) == 0:
        return

    t_dt = pd.to_datetime(df_times, unit='s' if pd.api.types.is_numeric_dtype(df_times) else None)
    span_days = (t_dt.max() - t_dt.min()).days

    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)

    if span_days > 730:
        formatter = mdates.DateFormatter('%Y-%m')
    elif span_days > 30:
        formatter = mdates.DateFormatter('%Y-%m-%d')
    elif span_days > 2:
        formatter = mdates.DateFormatter('%m-%d\n%H:%M')
    else:
        formatter = mdates.DateFormatter('%H:%M:%S')

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment('center')

def load_literature_event_context(csv_file):
    lit_csv = os.path.join(os.path.dirname(csv_file), "literature_events_context.csv")
    if os.path.exists(lit_csv):
        return pd.read_csv(lit_csv)
    return None

# =========================================================================
# ACTIVE MODULAR PANEL DRAWING FUNCTIONS (ALL 35 RESTORED)
# =========================================================================

def draw_frequencies(ax, df):
    ax.fill_between(df['datetime'], df['f_weighted_Hz'] - df['f_res_thermal_Hz'], df['f_weighted_Hz'] + df['f_res_thermal_Hz'], color='red', alpha=0.12, hatch='//', edgecolor='red', label=r'Particle Resonant Range ($\Delta f_{res}$)')
    ax.plot(df['datetime'], df['f_weighted_Hz'], color='blue', linewidth=1.5, label='Weighted Mean $f$')
    ax.fill_between(df['datetime'], df['f_weighted_Hz'] - df['f_bandwidth_Hz'], df['f_weighted_Hz'] + df['f_bandwidth_Hz'], color='blue', alpha=0.25, label=r'Wave Bandwidth ($\Delta f$)')
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Hammerhead Resonance Calibration Diagnostics", fontweight='bold', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

def draw_kinematics(ax, df):
    v_para_kms = df['v_para_sc_ms'] / 1000.0
    v_th_kms = df['v_th_para_ms'] / 1000.0
    ax.plot(df['datetime'], v_para_kms, color='purple', label=r'Hammerhead Beam $v_{\parallel, sc}$')
    ax.fill_between(df['datetime'], v_para_kms - v_th_kms, v_para_kms + v_th_kms, color='purple', alpha=0.2, label=r'Beam Thermal Spread ($v_{th, \parallel}$)')
    ax.set_ylabel("Hammerhead Velocity (km/s)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

def draw_k_parallel(ax, df):
    ax.plot(df['datetime'], df['k_para_weighted'], color='green', linewidth=1.5, label=r'$k_{\parallel}$ (Weighted Mean)')
    if 'v_th_para_ms' in df.columns and 'f_bandwidth_Hz' in df.columns:
        k_spread = (2 * np.pi * df['f_bandwidth_Hz']) / np.maximum(df['v_th_para_ms'], 1.0)
        ax.fill_between(df['datetime'], df['k_para_weighted'] - k_spread, df['k_para_weighted'] + k_spread, color='green', alpha=0.20, label=r'Thermal Wavenumber Spread ($\Delta k_{\parallel}$)')
    ax.set_ylabel(r"$k_{\parallel}$ (rad/m)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

def draw_correlation(ax, x_clean, y_clean, fidelity_limit=0.0):
    if len(x_clean) > 0:
        hb = ax.hexbin(x_clean, y_clean, xscale='log', yscale='log', gridsize=50, cmap='Blues', mincnt=1, bins='log')
        plt.colorbar(hb, ax=ax).set_label('Point Density (log10)')
        min_val, max_val = min(x_clean.min(), y_clean.min()), max(x_clean.max(), y_clean.max())
        x_ref = np.logspace(np.log10(max(min_val, 1e-2)), np.log10(max(max_val, 500)), 100)
        ax.plot(x_ref, x_ref, 'r--', linewidth=1.8, label='1:1 Reference')
        if fidelity_limit > 0.0:
            upper_bound, lower_bound = x_ref * fidelity_limit, x_ref / fidelity_limit
            ax.fill_between(x_ref, lower_bound, upper_bound, color='red', alpha=0.12, label=f'Fidelity Window ({1.0/fidelity_limit:.2f}x – {fidelity_limit:.1f}x)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(left=0.1, right=500.) 
    ax.set_ylim(bottom=0.1, top=500.)
    ax.set_xlabel(r"$k_{\mathrm{disp}} \cdot v_{\mathrm{th}, \parallel}$ (rad/s)")
    ax.set_ylabel(r"Wave Bandwidth $\Delta \omega$ (rad/s)")
    ax.set_title("Resonance Width Log-Log Correlation", fontweight='bold')
    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    ax.legend(loc='upper left', fontsize=9)

def draw_dispersion_validation(ax, df_clean, debug_col=None):
    """
    Physics Check: Beam-derived k_parallel,res vs. Cold Plasma k_disp.
    If debug_col is specified and present in df_clean, renders a color-coded
    scatter plot; otherwise defaults to a density hexbin map.
    """
    valid = df_clean.dropna(subset=['k_para_weighted', 'f_weighted_Hz', 'ne_lfr', 'B0_mag_nT']).copy()
    valid = valid[(valid['ne_lfr'] > 0) & (valid['f_weighted_Hz'] > 0) & (valid['B0_mag_nT'] > 0)]
    k_disp_arr = valid['k_disp'].values if 'k_disp' in valid.columns else calculate_cold_plasma_kdisp(valid)
    k_res_arr = np.abs(valid['k_para_weighted'].values)
    valid_mask = np.isfinite(k_disp_arr) & np.isfinite(k_res_arr) & (k_disp_arr > 0) & (k_res_arr > 0)

    # --- RENDER DEBUG SCATTER OR DEFAULT DENSITY HEXBIN ---
    if debug_col:
        # Scale velocity to km/s if v_para_sc is selected
        if debug_col in ['v_para_sc_ms', 'v_para_sc_kms'] and 'v_para_sc_ms' in valid.columns:
            c_vals = valid['v_para_sc_ms'].values[valid_mask] / 1000.0
            cbar_label = r'Hammerhead Beam $v_{\parallel, \mathrm{sc}}$ (km/s)'
        elif debug_col in valid.columns:
            c_vals = valid[debug_col].values[valid_mask]
            cbar_label = f'Debug Variable: {debug_col}'
        else:
            c_vals = None

        if c_vals is not None:
            sc = ax.scatter(
                k_disp_arr[valid_mask], k_res_arr[valid_mask], 
                c=c_vals, cmap='turbo', s=14, alpha=0.75, edgecolors='none'
            )
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label(cbar_label, fontweight='bold')
        else:
            hb = ax.hexbin(
                k_disp_arr[valid_mask], k_res_arr[valid_mask], 
                xscale='log', yscale='log', gridsize=45, cmap='plasma', mincnt=1, bins='log'
            )
            plt.colorbar(hb, ax=ax).set_label('Point Density (log10)')
    else:
        hb = ax.hexbin(
            k_disp_arr[valid_mask], k_res_arr[valid_mask], 
            xscale='log', yscale='log', gridsize=45, cmap='plasma', mincnt=1, bins='log'
        )
        plt.colorbar(hb, ax=ax).set_label('Point Density (log10)')

    ref = np.logspace(-6, -3, 200)
    ax.plot(ref, ref, 'r--', linewidth=2.0, label=r'1:1 Parity ($k_{\mathrm{res}} = k_{\mathrm{disp}}$)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(5e-6, 1e-3)
    ax.set_ylim(5e-6, 1e-3)
    ax.set_xlabel(r"Theoretical Plasma $k_{\mathrm{disp}}$ (rad/m)", fontweight='bold')
    ax.set_ylabel(r"Beam-Derived $k_{\parallel, \mathrm{res}}$ (rad/m)", fontweight='bold')
    
    title_suffix = f" [{debug_col}]" if (debug_col and debug_col in valid.columns) else ""
    ax.set_title(r"Physics Check: $k_{\parallel, \mathrm{res}}$ vs. Cold Plasma $k_{\mathrm{disp}}$" + title_suffix, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    ax.legend(loc='upper left', fontsize=9)

def draw_surface_map(ax, df_clean, min_events=15):
    """PANEL 05: 2D Surface Map of Median L_eff(f, debye_length_m)."""
    if df_clean.empty or 'debye_length_m' not in df_clean.columns: return
    valid = df_clean.dropna(subset=['f_weighted_Hz', 'debye_length_m', 'L_eff_m']).copy()
    if valid.empty: return

    f_vals, d_vals, l_vals = valid['f_weighted_Hz'].values, valid['debye_length_m'].values, valid['L_eff_m'].values
    ret_med = binned_statistic_2d(f_vals, d_vals, l_vals, statistic='median', bins=[35, 35])
    ret_cnt = binned_statistic_2d(f_vals, d_vals, l_vals, statistic='count', bins=[35, 35])
    
    f_edges, d_edges = ret_med.x_edge, ret_med.y_edge
    mesh_f, mesh_d = np.meshgrid(f_edges, d_edges)
    surface_matrix = ret_med.statistic.T
    surface_matrix[ret_cnt.statistic.T < min_events] = np.nan
    
    pcm = ax.pcolormesh(mesh_f, mesh_d, surface_matrix, cmap='viridis', vmin=0, vmax=20, shading='flat')
    plt.colorbar(pcm, ax=ax).set_label(rf'Median $L_{{eff}}$ (m) [$N \geq {min_events}$]')
    
    bin_f, bin_d = 0.5 * (f_edges[:-1] + f_edges[1:]), 0.5 * (d_edges[:-1] + d_edges[1:])
    X_c, Y_c = np.meshgrid(bin_f, bin_d)
    contours = ax.contour(X_c, Y_c, np.nan_to_num(surface_matrix, nan=0.0), levels=[3.5], colors='red', linewidths=2, linestyles='--')
    ax.clabel(contours, fmt=r'L_phys = 3.5m', inline=True, fontsize=10)
    
    ax.set_xlabel("Wave Frequency (Hz)", fontweight='bold')
    ax.set_ylabel(r"Electron Debye Length $\lambda_D$ (m)", fontweight='bold')
    ax.set_title(r"Surface Map: Median $L_{eff}(f, \lambda_D)$", fontweight='bold')
    ax.grid(True, alpha=0.3)


def draw_surface_map_3d(ax, df_clean, min_events=15):
    """PANEL 07: 3D Surface Map of Median L_eff(f, debye_length_m)."""
    if df_clean.empty or 'debye_length_m' not in df_clean.columns: return
    valid = df_clean.dropna(subset=['f_weighted_Hz', 'debye_length_m', 'L_eff_m']).copy()
    if valid.empty: return

    f_vals, d_vals, l_vals = valid['f_weighted_Hz'].values, valid['debye_length_m'].values, valid['L_eff_m'].values
    ret_med = binned_statistic_2d(f_vals, d_vals, l_vals, statistic='median', bins=[35, 35])
    ret_cnt = binned_statistic_2d(f_vals, d_vals, l_vals, statistic='count', bins=[35, 35])
    
    f_edges, d_edges = ret_med.x_edge, ret_med.y_edge
    X_c, Y_c = np.meshgrid(0.5 * (f_edges[:-1] + f_edges[1:]), 0.5 * (d_edges[:-1] + d_edges[1:]))
    surface_matrix = ret_med.statistic.T
    surface_matrix[ret_cnt.statistic.T < min_events] = np.nan
    
    valid_mask = ~np.isnan(surface_matrix)
    if len(surface_matrix[valid_mask]) > 3:
        ax.plot_trisurf(X_c[valid_mask], Y_c[valid_mask], surface_matrix[valid_mask], cmap='viridis', linewidth=0.2, edgecolor='k', alpha=0.85, vmin=0, vmax=20)
        ax.plot_surface(X_c, Y_c, np.full_like(surface_matrix, 3.5), color='red', alpha=0.20)
        
    ax.set_xlabel("Frequency $f$ (Hz)", labelpad=10)
    ax.set_ylabel(r"Debye Length $\lambda_D$ (m)", labelpad=10)
    ax.set_zlabel(r"Median $L_{eff}$ (m)", labelpad=10)
    ax.set_title(r"3D Calibration Surface: $L_{eff}(f, \lambda_D)$", fontweight='bold')
    ax.view_init(elev=28, azim=-125)

def draw_debye_slices(ax, df_clean, popt_bandpass=None, min_events=5):
    if df_clean.empty or 'debye_length_m' not in df_clean.columns: return
    valid_df = df_clean.dropna(subset=['f_weighted_Hz', 'debye_length_m', 'L_eff_m']).copy()
    valid_df['debye_regime'] = pd.cut(valid_df['debye_length_m'], bins=[-np.inf, 1.0, 1.8, 2.5, np.inf], labels=[r'$\lambda_D \leq 1.0\mathrm{m}$', r'$1.0 < \lambda_D \leq 1.8\mathrm{m}$', r'$1.8 < \lambda_D \leq 2.5\mathrm{m}$', r'$\lambda_D > 2.5\mathrm{m}$'])
    colors_list = ['crimson', 'darkorange', 'teal', 'purple']
    freq_bins = np.linspace(valid_df['f_weighted_Hz'].min(), valid_df['f_weighted_Hz'].max(), 15)
    f_centers = 0.5 * (freq_bins[:-1] + freq_bins[1:])
    f_eval = np.linspace(valid_df['f_weighted_Hz'].min(), valid_df['f_weighted_Hz'].max(), 100)
    for (regime_name, group), col in zip(valid_df.groupby('debye_regime', observed=False), colors_list):
        if group.empty: continue
        group_copy = group.copy()
        group_copy['f_bin'] = pd.cut(group_copy['f_weighted_Hz'], freq_bins)
        binned_med = group_copy.groupby('f_bin', observed=False)['L_eff_m'].median()
        binned_q25 = group_copy.groupby('f_bin', observed=False)['L_eff_m'].quantile(0.25)
        binned_q75 = group_copy.groupby('f_bin', observed=False)['L_eff_m'].quantile(0.75)
        mask_valid = group_copy.groupby('f_bin', observed=False)['L_eff_m'].count().values >= min_events
        med_vals, q25_vals, q75_vals = binned_med.values.copy(), binned_q25.values.copy(), binned_q75.values.copy()
        med_vals[~mask_valid], q25_vals[~mask_valid], q75_vals[~mask_valid] = np.nan, np.nan, np.nan
        ax.plot(f_centers, med_vals, color=col, linewidth=2.0, marker='o', markersize=3, label=f'{regime_name}')
        ax.fill_between(f_centers, q25_vals, q75_vals, color=col, alpha=0.15)
        if popt_bandpass is not None and 'V_sc_volts' in group.columns:
            v_med = group['V_sc_volts'].median()
            if np.isfinite(v_med):
                ax.plot(f_eval, analytical_leff_model_physics_bandpass((f_eval, v_med), *popt_bandpass), color=col, linestyle='--', linewidth=1.5, alpha=0.85)
    ax.axhline(3.5, color='gray', linestyle=':', linewidth=1.2, label=r'$L_{phys} = 3.5$m')
    ax.set_xlabel("Wave Frequency (Hz)", fontweight='bold')
    ax.set_ylabel(r"Effective Length $L_{eff}$ (m)", fontweight='bold')
    ax.set_title(r"$L_{eff}$ Slices Across $\lambda_D$ Regimes", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, ncol=2)

def draw_vsc_surface_map(ax, df_clean, min_events=15):
    if not df_clean.empty and 'V_sc_volts' in df_clean.columns:
        valid_df = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'L_eff_m'])
        f_vals, v_vals, l_vals = valid_df['f_weighted_Hz'].values, valid_df['V_sc_volts'].values, valid_df['L_eff_m'].values
        ret_med = binned_statistic_2d(f_vals, v_vals, l_vals, statistic='median', bins=[35, 35])
        ret_cnt = binned_statistic_2d(f_vals, v_vals, l_vals, statistic='count', bins=[35, 35])
        f_edges, v_edges = ret_med.x_edge, ret_med.y_edge
        mesh_f, mesh_v = np.meshgrid(f_edges, v_edges)
        surface_matrix = ret_med.statistic.T
        surface_matrix[ret_cnt.statistic.T < min_events] = np.nan
        pcm = ax.pcolormesh(mesh_f, mesh_v, surface_matrix, cmap='plasma', vmin=0, vmax=20, shading='flat')
        plt.colorbar(pcm, ax=ax).set_label(rf'Median $L_{{eff}}$ (m) [$N \geq {min_events}$]')
        X_c, Y_c = np.meshgrid(0.5 * (f_edges[:-1] + f_edges[1:]), 0.5 * (v_edges[:-1] + v_edges[1:]))
        contours = ax.contour(X_c, Y_c, np.nan_to_num(surface_matrix, nan=0.0), levels=[3.5], colors='cyan', linewidths=2, linestyles='--')
        ax.clabel(contours, fmt=r'L_phys = 3.5m', inline=True, fontsize=10)
    ax.set_xlabel("Wave Frequency (Hz)")
    ax.set_ylabel(r"Spacecraft Potential $V_{sc}$ (Volts)")
    ax.set_title(r"Surface Map: Median $L_{eff}(f, V_{sc})$", fontweight='bold')
    ax.grid(True, alpha=0.3)

def draw_vsc_slices(ax, df_clean, popt_bandpass=None, min_events=5):
    if df_clean.empty or 'V_sc_volts' not in df_clean.columns: return
    valid_df = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'L_eff_m']).copy()
    valid_df['vsc_regime'] = pd.cut(valid_df['V_sc_volts'], bins=[-np.inf, 1.0, 3.5, 6.5, np.inf], labels=[r'Low $V_{sc} \leq 1.0\mathrm{V}$', r'Mod $1.0 < V_{sc} \leq 3.5\mathrm{V}$', r'Peak $3.5 < V_{sc} \leq 6.5\mathrm{V}$', r'High $V_{sc} > 6.5\mathrm{V}$'])
    colors_list = ['purple', 'teal', 'darkorange', 'crimson']
    freq_bins = np.linspace(valid_df['f_weighted_Hz'].min(), valid_df['f_weighted_Hz'].max(), 15)
    f_centers = 0.5 * (freq_bins[:-1] + freq_bins[1:])
    f_eval = np.linspace(valid_df['f_weighted_Hz'].min(), valid_df['f_weighted_Hz'].max(), 100)
    for (regime_name, group), col in zip(valid_df.groupby('vsc_regime', observed=False), colors_list):
        if group.empty: continue
        group_copy = group.copy()
        group_copy['f_bin'] = pd.cut(group_copy['f_weighted_Hz'], freq_bins)
        binned_med = group_copy.groupby('f_bin', observed=False)['L_eff_m'].median()
        binned_q25 = group_copy.groupby('f_bin', observed=False)['L_eff_m'].quantile(0.25)
        binned_q75 = group_copy.groupby('f_bin', observed=False)['L_eff_m'].quantile(0.75)
        mask_valid = group_copy.groupby('f_bin', observed=False)['L_eff_m'].count().values >= min_events
        med_vals, q25_vals, q75_vals = binned_med.values.copy(), binned_q25.values.copy(), binned_q75.values.copy()
        med_vals[~mask_valid], q25_vals[~mask_valid], q75_vals[~mask_valid] = np.nan, np.nan, np.nan
        ax.plot(f_centers, med_vals, color=col, linewidth=2.0, marker='o', markersize=3, label=f'{regime_name}')
        ax.fill_between(f_centers, q25_vals, q75_vals, color=col, alpha=0.15)
        if popt_bandpass is not None:
            v_med = group['V_sc_volts'].median()
            if np.isfinite(v_med):
                ax.plot(f_eval, analytical_leff_model_physics_bandpass((f_eval, v_med), *popt_bandpass), color=col, linestyle='--', linewidth=1.5, alpha=0.85)
    ax.axhline(3.5, color='gray', linestyle=':', linewidth=1.2, label=r'$L_{phys} = 3.5$m')
    ax.set_xlabel("Wave Frequency (Hz)", fontweight='bold')
    ax.set_ylabel(r"Effective Length $L_{eff}$ (m)", fontweight='bold')
    ax.set_title(r"$L_{eff}$ Slices Across $V_{sc}$ Regimes", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, ncol=2)

def draw_vsc_surface_map_3d(ax, df_clean, min_events=15):
    if not df_clean.empty and 'V_sc_volts' in df_clean.columns:
        valid_df = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'L_eff_m'])
        f_vals, v_vals, l_vals = valid_df['f_weighted_Hz'].values, valid_df['V_sc_volts'].values, valid_df['L_eff_m'].values
        ret_med = binned_statistic_2d(f_vals, v_vals, l_vals, statistic='median', bins=[35, 35])
        ret_cnt = binned_statistic_2d(f_vals, v_vals, l_vals, statistic='count', bins=[35, 35])
        f_edges, v_edges = ret_med.x_edge, ret_med.y_edge
        X_c, Y_c = np.meshgrid(0.5 * (f_edges[:-1] + f_edges[1:]), 0.5 * (v_edges[:-1] + v_edges[1:]))
        surface_matrix = ret_med.statistic.T
        surface_matrix[ret_cnt.statistic.T < min_events] = np.nan
        valid_mask = ~np.isnan(surface_matrix)
        if len(surface_matrix[valid_mask]) > 3:
            ax.plot_trisurf(X_c[valid_mask], Y_c[valid_mask], surface_matrix[valid_mask], cmap='plasma', linewidth=0.2, edgecolor='k', alpha=0.85, vmin=0, vmax=20)
            ax.plot_surface(X_c, Y_c, np.full_like(surface_matrix, 3.5), color='cyan', alpha=0.20)
    ax.set_xlabel("Frequency $f$ (Hz)", labelpad=10)
    ax.set_ylabel(r"Potential $V_{sc}$ (V)", labelpad=10)
    ax.set_zlabel(r"Median $L_{eff}$ (m)", labelpad=10)
    ax.set_title(r"3D Surface: $L_{eff}(f, V_{sc})$", fontweight='bold')
    ax.view_init(elev=28, azim=-125)

def draw_vsc_analytical_surface_2d(ax, df_clean, popt_vsc):
    if popt_vsc is None: return
    F, V = np.meshgrid(np.linspace(df_clean['f_weighted_Hz'].min(), df_clean['f_weighted_Hz'].max(), 100), np.linspace(df_clean['V_sc_volts'].min(), df_clean['V_sc_volts'].max(), 100))
    Z_model = analytical_leff_model_f_vsc((F, V), *popt_vsc)
    pcm = ax.pcolormesh(F, V, Z_model, cmap='plasma', vmin=0, vmax=20, shading='auto')
    plt.colorbar(pcm, ax=ax).set_label(r'Analytical $L_{eff}$ Model (m)')
    contours = ax.contour(F, V, Z_model, levels=[3.5, 5.0, 7.5, 10.0, 15.0], colors='white', linewidths=1.2, linestyles='--')
    ax.clabel(contours, fmt='%.1fm', inline=True, fontsize=8)
    ax.set_xlabel("Wave Frequency (Hz)")
    ax.set_ylabel(r"Spacecraft Potential $V_{sc}$ (V)")
    ax.set_title(r"Analytical Surface Model: $L_{eff}(f, V_{sc})$", fontweight='bold')
    ax.grid(True, alpha=0.3)

def draw_vsc_analytical_surface_3d(ax, df_clean, popt_vsc):
    if popt_vsc is None: return
    F, V = np.meshgrid(np.linspace(df_clean['f_weighted_Hz'].min(), df_clean['f_weighted_Hz'].max(), 40), np.linspace(df_clean['V_sc_volts'].min(), df_clean['V_sc_volts'].max(), 40))
    Z_model = analytical_leff_model_f_vsc((F, V), *popt_vsc)
    ax.plot_surface(F, V, Z_model, cmap='plasma', alpha=0.6, linewidth=0.2, edgecolor='k', vmin=0, vmax=20)
    ax.plot_surface(F, V, np.full_like(Z_model, 3.5), color='cyan', alpha=0.2)
    ax.set_xlabel("Frequency $f$ (Hz)", labelpad=10)
    ax.set_ylabel(r"Potential $V_{sc}$ (V)", labelpad=10)
    ax.set_zlabel(r"Model $L_{eff}$ (m)", labelpad=10)
    ax.set_title(r"3D Model Surface Fit: $L_{eff}(f, V_{sc})$", fontweight='bold')
    ax.view_init(elev=28, azim=-125)

def draw_bandpass_analytical_surface_2d(ax, df_clean, popt_bandpass):
    if popt_bandpass is None: return
    F, V = np.meshgrid(np.linspace(df_clean['f_weighted_Hz'].min(), df_clean['f_weighted_Hz'].max(), 100), np.linspace(df_clean['V_sc_volts'].min(), df_clean['V_sc_volts'].max(), 100))
    Z_model = analytical_leff_model_physics_bandpass((F, V), *popt_bandpass)
    pcm = ax.pcolormesh(F, V, Z_model, cmap='plasma', vmin=0, vmax=20, shading='auto')
    plt.colorbar(pcm, ax=ax).set_label(r'Bandpass Model $L_{eff}$ (m)')
    contours = ax.contour(F, V, Z_model, levels=[3.5, 5.0, 7.5, 10.0, 15.0], colors='white', linewidths=1.2, linestyles='--')
    ax.clabel(contours, fmt='%.1fm', inline=True, fontsize=8)
    ax.set_xlabel("Wave Frequency (Hz)")
    ax.set_ylabel(r"Spacecraft Potential $V_{sc}$ (V)")
    ax.set_title(r"Physics Bandpass Surface Model: $L_{eff}(f, V_{sc})$", fontweight='bold')
    ax.grid(True, alpha=0.3)

def draw_bandpass_analytical_surface_3d(ax, df_clean, popt_bandpass):
    if popt_bandpass is None: return
    F, V = np.meshgrid(np.linspace(df_clean['f_weighted_Hz'].min(), df_clean['f_weighted_Hz'].max(), 40), np.linspace(df_clean['V_sc_volts'].min(), df_clean['V_sc_volts'].max(), 40))
    Z_model = analytical_leff_model_physics_bandpass((F, V), *popt_bandpass)
    ax.plot_surface(F, V, Z_model, cmap='plasma', alpha=0.6, linewidth=0.2, edgecolor='k', vmin=0, vmax=20)
    ax.plot_surface(F, V, np.full_like(Z_model, 3.5), color='cyan', alpha=0.2)
    ax.set_xlabel("Frequency $f$ (Hz)", labelpad=10)
    ax.set_ylabel(r"Potential $V_{sc}$ (V)", labelpad=10)
    ax.set_zlabel(r"Model $L_{eff}$ (m)", labelpad=10)
    ax.set_title(r"3D Bandpass Fit Surface: $L_{eff}(f, V_{sc})$", fontweight='bold')
    ax.view_init(elev=28, azim=-125)

def draw_leff_vs_wave_amplitude(ax, df_clean):
    if 'B_wave_nT' not in df_clean.columns: return
    valid = df_clean[(df_clean['B_wave_nT'] > 0) & np.isfinite(df_clean['B_wave_nT']) & (df_clean['L_eff_m'] > 0)]
    hb = ax.hexbin(valid['B_wave_nT'], valid['L_eff_m'], xscale='log', gridsize=40, cmap='magma', mincnt=1, bins='log')
    plt.colorbar(hb, ax=ax).set_label('Point Density (log10)')
    ax.axhline(3.5, color='cyan', linestyle='--', linewidth=1.5, label=r'Baseline $L_{phys} = 3.5$m')
    ax.set_xscale('log')
    ax.set_xlabel(r"Wave Fluctuation $\delta B$ (nT)")
    ax.set_ylabel(r"Effective Length $L_{eff}$ (m)")
    ax.set_title(r"Linearity Check: $L_{eff}$ vs. Wave Amplitude $\delta B$", fontweight='bold')
    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    ax.legend(loc='upper right')

def draw_vsc_vs_density(ax, df_clean, popt_bandpass=None):
    if 'ne_lfr' not in df_clean.columns or 'V_sc_volts' not in df_clean.columns: return
    valid = df_clean[(df_clean['V_sc_volts'] > -5) & (df_clean['ne_lfr'] > 0) & np.isfinite(df_clean['ne_lfr'])]
    hb = ax.hexbin(valid['V_sc_volts'], valid['ne_lfr'], yscale='log', gridsize=40, cmap='viridis', mincnt=1)
    plt.colorbar(hb, ax=ax).set_label('Point Density (log10)')
    V0_val = popt_bandpass[1] if popt_bandpass is not None else 4.84
    ax.axvline(V0_val, color='red', linestyle='--', linewidth=1.5, label=rf'Debye Threshold $V_0 = {V0_val:.2f}$V')
    ax.set_yscale('log')
    ax.set_xlabel(r"Spacecraft Potential $V_{sc}$ (Volts)")
    ax.set_ylabel(r"Electron Density $n_e$ (cm$^{-3}$)")
    ax.set_title(r"Proof 1: Density Surge at High $V_{sc}$", fontweight='bold')
    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    ax.legend(loc='upper left')

def draw_vsc_vs_debye(ax, df_clean, popt_bandpass=None):
    if 'debye_length_m' not in df_clean.columns or 'V_sc_volts' not in df_clean.columns: return
    valid = df_clean[(df_clean['V_sc_volts'] > -5) & (df_clean['debye_length_m'] > 0) & np.isfinite(df_clean['debye_length_m'])]
    hb = ax.hexbin(valid['V_sc_volts'], valid['debye_length_m'], gridsize=40, cmap='plasma', mincnt=1)
    plt.colorbar(hb, ax=ax).set_label('Point Density (log10)')
    V0_val = popt_bandpass[1] if popt_bandpass is not None else 4.84
    ax.axvline(V0_val, color='cyan', linestyle='--', linewidth=1.5, label=rf'Debye Threshold $V_0 = {V0_val:.2f}$V')
    ax.axhline(3.5, color='white', linestyle=':', linewidth=1.5, label=r'Baseline $L_{phys} = 3.5$m')
    ax.set_xlabel(r"Spacecraft Potential $V_{sc}$ (Volts)")
    ax.set_ylabel(r"Electron Debye Length $\lambda_D$ (m)")
    ax.set_title(r"Proof 2: Sheath Compression at High $V_{sc}$", fontweight='bold')
    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    ax.legend(loc='upper right')

def draw_instrument_rc_bandwidth(ax, df_clean):
    if 'f_RC_corner_Hz' not in df_clean.columns or 'f_weighted_Hz' not in df_clean.columns: return
    valid = df_clean.dropna(subset=['f_RC_corner_Hz', 'f_weighted_Hz', 'L_eff_m'])
    valid = valid[(valid['f_RC_corner_Hz'] > 0) & (valid['f_weighted_Hz'] > 0)]
    hb = ax.hexbin(valid['f_RC_corner_Hz'], valid['f_weighted_Hz'], xscale='log', yscale='log', gridsize=40, cmap='magma', mincnt=1, bins='log')
    plt.colorbar(hb, ax=ax).set_label('Point Density (log10)')
    ref_line = np.logspace(np.log10(max(0.1, valid['f_RC_corner_Hz'].min())), np.log10(min(200.0, valid['f_weighted_Hz'].max())), 100)
    ax.plot(ref_line, ref_line, 'r--', linewidth=2.0, label=r'1:1 $RC$ Cutoff ($f_{\mathrm{wave}} = f_{\mathrm{RC}}$)')
    ax.fill_between(ref_line, ref_line, 200.0, color='red', alpha=0.12, label=r'Attenuated Zone ($f > f_{\mathrm{RC}}$)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(left=0.5, right=100.0)
    ax.set_ylim(bottom=0.1, top=50.0)
    ax.set_xlabel(r"In-Situ Sheath $RC$ Corner Frequency $f_{\mathrm{RC}}$ (Hz)", fontweight='bold')
    ax.set_ylabel(r"Observed Wave Frequency $f_{\mathrm{weighted}}$ (Hz)", fontweight='bold')
    ax.set_title(r"Instrument Behavior: Frequency vs. Preamplifier $RC$ Bandwidth", fontweight='bold')
    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    ax.legend(loc='upper left', fontsize=9)

def draw_overlay_data_and_model_contours(ax, df_clean, popt_bandpass, min_events=5):
    if df_clean.empty or popt_bandpass is None: return
    valid_df = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'L_eff_m'])
    ret_med = binned_statistic_2d(valid_df['f_weighted_Hz'].values, valid_df['V_sc_volts'].values, valid_df['L_eff_m'].values, statistic='median', bins=[35, 35])
    ret_cnt = binned_statistic_2d(valid_df['f_weighted_Hz'].values, valid_df['V_sc_volts'].values, valid_df['L_eff_m'].values, statistic='count', bins=[35, 35])
    f_edges, v_edges = ret_med.x_edge, ret_med.y_edge
    mesh_f, mesh_v = np.meshgrid(f_edges, v_edges)
    surface_matrix = ret_med.statistic.T
    surface_matrix[ret_cnt.statistic.T < min_events] = np.nan
    pcm = ax.pcolormesh(mesh_f, mesh_v, surface_matrix, cmap='plasma', vmin=0, vmax=20, shading='flat')
    plt.colorbar(pcm, ax=ax).set_label(rf'Empirical Median $L_{{eff}}$ (m) [$N \geq {min_events}$]')
    F, V = np.meshgrid(np.linspace(f_edges.min(), f_edges.max(), 100), np.linspace(v_edges.min(), v_edges.max(), 100))
    contours = ax.contour(F, V, analytical_leff_model_physics_bandpass((F, V), *popt_bandpass), levels=[3.5, 5.0, 7.5, 10.0, 15.0], colors='cyan', linewidths=1.8, linestyles='--')
    ax.clabel(contours, fmt='%.1fm', inline=True, fontsize=9)
    ax.set_xlabel("Wave Frequency (Hz)")
    ax.set_ylabel(r"Spacecraft Potential $V_{sc}$ (V)")
    ax.set_title(r"Empirical Data Map with Overlaid Model Contours", fontweight='bold')
    ax.grid(True, alpha=0.3)

def draw_2d_model_residuals(ax, df_clean, popt_bandpass, min_events=5):
    if df_clean.empty or popt_bandpass is None: return
    valid_df = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'L_eff_m'])
    ret_med = binned_statistic_2d(valid_df['f_weighted_Hz'].values, valid_df['V_sc_volts'].values, valid_df['L_eff_m'].values, statistic='median', bins=[35, 35])
    ret_cnt = binned_statistic_2d(valid_df['f_weighted_Hz'].values, valid_df['V_sc_volts'].values, valid_df['L_eff_m'].values, statistic='count', bins=[35, 35])
    f_edges, v_edges = ret_med.x_edge, ret_med.y_edge
    mesh_f, mesh_v = np.meshgrid(f_edges, v_edges)
    X_c, Y_c = np.meshgrid(0.5 * (f_edges[:-1] + f_edges[1:]), 0.5 * (v_edges[:-1] + v_edges[1:]))
    residual_matrix = ret_med.statistic.T - analytical_leff_model_physics_bandpass((X_c, Y_c), *popt_bandpass)
    residual_matrix[ret_cnt.statistic.T < min_events] = np.nan
    pcm = ax.pcolormesh(mesh_f, mesh_v, residual_matrix, cmap='coolwarm', vmin=-5.0, vmax=5.0, shading='flat')
    plt.colorbar(pcm, ax=ax).set_label(r'Residual $\Delta L_{eff}$ (Data - Model) [m]')
    V0_val = popt_bandpass[1] if popt_bandpass is not None else 4.84
    ax.axhline(V0_val, color='black', linestyle=':', label=rf'$V_0 = {V0_val:.2f}\mathrm{{V}}$')
    ax.set_xlabel("Wave Frequency (Hz)")
    ax.set_ylabel(r"Spacecraft Potential $V_{sc}$ (V)")
    ax.set_title(r"Model Residual Map: $L_{data} - L_{model}$", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

def draw_leff_vs_deltaB_controlled(ax, df_clean):
    if 'B_wave_nT' not in df_clean.columns or 'V_sc_volts' not in df_clean.columns: return
    valid = df_clean.dropna(subset=['B_wave_nT', 'V_sc_volts', 'L_eff_m']).copy()
    valid = valid[(valid['B_wave_nT'] > 0) & (valid['L_eff_m'] > 0)]
    valid['vsc_slice'] = pd.cut(valid['V_sc_volts'], bins=[-np.inf, 1.0, 3.5, 6.5, np.inf], labels=[r'Low $V_{sc} \leq 1.0\mathrm{V}$', r'Mod $1.0 < V_{sc} \leq 3.5\mathrm{V}$', r'Peak $3.5 < V_{sc} \leq 6.5\mathrm{V}$', r'High $V_{sc} > 6.5\mathrm{V}$'])
    colors_list = ['purple', 'teal', 'darkorange', 'crimson']
    for (slice_name, group), col in zip(valid.groupby('vsc_slice', observed=False), colors_list):
        if len(group) < 15: continue
        b_bins = np.logspace(np.log10(group['B_wave_nT'].min()), np.log10(group['B_wave_nT'].max()), 10)
        group_copy = group.copy()
        group_copy['b_bin'] = pd.cut(group_copy['B_wave_nT'], b_bins)
        b_centers = 0.5 * (b_bins[:-1] + b_bins[1:])
        binned_med = group_copy.groupby('b_bin', observed=False)['L_eff_m'].median()
        binned_q25 = group_copy.groupby('b_bin', observed=False)['L_eff_m'].quantile(0.25)
        binned_q75 = group_copy.groupby('b_bin', observed=False)['L_eff_m'].quantile(0.75)
        mask_valid = group_copy.groupby('b_bin', observed=False)['L_eff_m'].count().values >= 3
        med_vals, q25_vals, q75_vals = binned_med.values.copy(), binned_q25.values.copy(), binned_q75.values.copy()
        med_vals[~mask_valid], q25_vals[~mask_valid], q75_vals[~mask_valid] = np.nan, np.nan, np.nan
        ax.plot(b_centers, med_vals, color=col, linewidth=2.0, marker='o', markersize=4, label=f'{slice_name}')
        ax.fill_between(b_centers, q25_vals, q75_vals, color=col, alpha=0.18)
    ax.axhline(3.5, color='gray', linestyle=':', label=r'$L_{phys} = 3.5\mathrm{m}$')
    ax.set_xscale('log')
    ax.set_xlabel(r"Wave Fluctuation Amplitude $\delta B$ (nT)", fontweight='bold')
    ax.set_ylabel(r"Median Effective Length $L_{eff}$ (m)", fontweight='bold')
    ax.set_title(r"Independence Check: $L_{eff}$ vs. $\delta B$", fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

def draw_observed_vs_predicted_parity(ax, df_clean, popt_wavepower, r2_wavepower=None):
    if popt_wavepower is None or df_clean.empty: return
    valid = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'B_wave_nT', 'L_eff_m']).copy()
    l_pred = analytical_leff_model_physics_wavepower((valid['f_weighted_Hz'].values, valid['V_sc_volts'].values, valid['B_wave_nT'].values), *popt_wavepower)
    hb = ax.hexbin(l_pred, valid['L_eff_m'].values, gridsize=45, cmap='plasma', mincnt=1, bins='log', extent=[0, 30, 0, 30])
    plt.colorbar(hb, ax=ax).set_label('Point Density (log10)')
    ref = np.linspace(0, 30, 100)
    ax.plot(ref, ref, 'r--', linewidth=2.0, label=r'1:1 Parity ($y = x$)')
    ax.fill_between(ref, ref - 2.0, ref + 2.0, color='red', alpha=0.10, label=r'$\pm 2\mathrm{m}$ Tolerance Band')
    r2_str = f"{r2_wavepower:.4f}" if r2_wavepower is not None else "N/A"
    ax.text(0.05, 0.82, rf"Model 5 (Full Physics)" + "\n" + rf"$R^2 = {r2_str}$" + "\n" + rf"$N = {len(valid):,}$ pts", 
            transform=ax.transAxes, fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_xlabel(r"Model 5 Predicted $L_{eff}$ (m)", fontweight='bold')
    ax.set_ylabel(r"Observed $L_{eff}$ from Resonance (m)", fontweight='bold')
    ax.set_title(r"Model Accuracy: Observed vs. Predicted $L_{eff}$", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)

def draw_leff_vs_deltaB_with_model5(ax, df_clean, popt_wavepower):
    if 'B_wave_nT' not in df_clean.columns or 'V_sc_volts' not in df_clean.columns or popt_wavepower is None: return
    valid = df_clean.dropna(subset=['B_wave_nT', 'V_sc_volts', 'f_weighted_Hz', 'L_eff_m']).copy()
    valid = valid[(valid['B_wave_nT'] > 0) & (valid['L_eff_m'] > 0)]
    valid['vsc_slice'] = pd.cut(valid['V_sc_volts'], bins=[-np.inf, 1.0, 3.5, 6.5, np.inf], labels=[r'Low $V_{sc} \leq 1.0\mathrm{V}$', r'Mod $1.0 < V_{sc} \leq 3.5\mathrm{V}$', r'Peak $3.5 < V_{sc} \leq 6.5\mathrm{V}$', r'High $V_{sc} > 6.5\mathrm{V}$'])
    colors_list = ['purple', 'teal', 'darkorange', 'crimson']
    b_eval = np.logspace(-1, 1.3, 100)
    for (slice_name, group), col in zip(valid.groupby('vsc_slice', observed=False), colors_list):
        if len(group) < 15: continue
        b_bins = np.logspace(np.log10(group['B_wave_nT'].min()), np.log10(group['B_wave_nT'].max()), 10)
        group_copy = group.copy()
        group_copy['b_bin'] = pd.cut(group_copy['B_wave_nT'], b_bins)
        b_centers = 0.5 * (b_bins[:-1] + b_bins[1:])
        binned_med = group_copy.groupby('b_bin', observed=False)['L_eff_m'].median()
        binned_q25 = group_copy.groupby('b_bin', observed=False)['L_eff_m'].quantile(0.25)
        binned_q75 = group_copy.groupby('b_bin', observed=False)['L_eff_m'].quantile(0.75)
        mask_valid = group_copy.groupby('b_bin', observed=False)['L_eff_m'].count().values >= 3
        med_vals, q25_vals, q75_vals = binned_med.values.copy(), binned_q25.values.copy(), binned_q75.values.copy()
        med_vals[~mask_valid], q25_vals[~mask_valid], q75_vals[~mask_valid] = np.nan, np.nan, np.nan
        ax.plot(b_centers, med_vals, color=col, linewidth=1.8, marker='o', markersize=4, label=f'{slice_name}')
        ax.fill_between(b_centers, q25_vals, q75_vals, color=col, alpha=0.15)
        v_med, f_med = group['V_sc_volts'].median(), group['f_weighted_Hz'].median()
        if np.isfinite(v_med) and np.isfinite(f_med):
            ax.plot(b_eval, analytical_leff_model_physics_wavepower((f_med, v_med, b_eval), *popt_wavepower), color=col, linestyle='--', linewidth=1.8, alpha=0.90)
    ax.axhline(popt_wavepower[0], color='gray', linestyle=':', label=rf'$L_{{base}} = {popt_wavepower[0]:.2f}\mathrm{{m}}$')
    ax.set_xscale('log')
    ax.set_xlabel(r"Wave Fluctuation Amplitude $\delta B$ (nT)", fontweight='bold')
    ax.set_ylabel(r"Effective Length $L_{eff}$ (m)", fontweight='bold')
    ax.set_title(r"AC Sheath Damping: Data vs. Model 5 Fit", fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, ncol=2)

def draw_model5_surface_and_overlay(ax, df_clean, popt_wavepower, min_events=5):
    if df_clean.empty or popt_wavepower is None: return
    valid = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'B_wave_nT', 'L_eff_m']).copy()
    ret_med_L = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['L_eff_m'], statistic='median', bins=[35, 35])
    ret_med_B = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['B_wave_nT'], statistic='median', bins=[35, 35])
    ret_cnt   = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['L_eff_m'], statistic='count', bins=[35, 35])
    f_edges, v_edges = ret_med_L.x_edge, ret_med_L.y_edge
    mesh_f, mesh_v = np.meshgrid(f_edges, v_edges)
    X_c, Y_c = np.meshgrid(0.5 * (f_edges[:-1] + f_edges[1:]), 0.5 * (v_edges[:-1] + v_edges[1:]))
    surf_L = ret_med_L.statistic.T
    surf_L[ret_cnt.statistic.T < min_events] = np.nan
    pcm = ax.pcolormesh(mesh_f, mesh_v, surf_L, cmap='plasma', vmin=0, vmax=20, shading='flat')
    plt.colorbar(pcm, ax=ax).set_label(rf'Empirical Median $L_{{eff}}$ (m) [$N \geq {min_events}$]')
    Z_model_2d = analytical_leff_model_physics_wavepower((X_c, Y_c, np.nan_to_num(ret_med_B.statistic.T, nan=1.0)), *popt_wavepower)
    contours = ax.contour(X_c, Y_c, Z_model_2d, levels=[2.0, 3.5, 5.0, 7.5, 10.0, 15.0], colors='cyan', linewidths=1.8, linestyles='--')
    ax.clabel(contours, fmt='%.1fm', inline=True, fontsize=9.5)
    ax.set_xlabel("Wave Frequency (Hz)", fontweight='bold')
    ax.set_ylabel(r"Spacecraft Potential $V_{sc}$ (V)", fontweight='bold')
    ax.set_title(r"Data Map with Overlaid Model 5 Contours ($\delta B$ Damped)", fontweight='bold')
    ax.grid(True, alpha=0.3)

def draw_model5_2d_residuals(ax, df_clean, popt_wavepower, min_events=5):
    if df_clean.empty or popt_wavepower is None: return
    valid = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'B_wave_nT', 'L_eff_m']).copy()
    ret_med_L = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['L_eff_m'], statistic='median', bins=[35, 35])
    ret_med_B = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['B_wave_nT'], statistic='median', bins=[35, 35])
    ret_cnt   = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['L_eff_m'], statistic='count', bins=[35, 35])
    f_edges, v_edges = ret_med_L.x_edge, ret_med_L.y_edge
    mesh_f, mesh_v = np.meshgrid(f_edges, v_edges)
    X_c, Y_c = np.meshgrid(0.5 * (f_edges[:-1] + f_edges[1:]), 0.5 * (v_edges[:-1] + v_edges[1:]))
    residual_matrix = ret_med_L.statistic.T - analytical_leff_model_physics_wavepower((X_c, Y_c, np.nan_to_num(ret_med_B.statistic.T, nan=1.0)), *popt_wavepower)
    residual_matrix[ret_cnt.statistic.T < min_events] = np.nan
    pcm = ax.pcolormesh(mesh_f, mesh_v, residual_matrix, cmap='coolwarm', vmin=-5.0, vmax=5.0, shading='flat')
    plt.colorbar(pcm, ax=ax).set_label(r'Residual $\Delta L_{eff}$ (Data - Model 5) [m]')
    ax.axhline(popt_wavepower[2], color='black', linestyle=':', label=rf'$V_0 = {popt_wavepower[2]:.2f}\mathrm{{V}}$')
    ax.set_xlabel("Wave Frequency (Hz)", fontweight='bold')
    ax.set_ylabel(r"Spacecraft Potential $V_{sc}$ (V)", fontweight='bold')
    ax.set_title(r"Model 5 Residual Map: $L_{\mathrm{data}} - L_{\mathrm{model5}}$", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

def draw_bias_current_residual_check(ax, df_clean, popt_wavepower):
    if 'I_bias_nA' not in df_clean.columns or popt_wavepower is None:
        ax.text(0.5, 0.5, "Bias current ('I_bias_nA') not in CSV.", ha='center', va='center', transform=ax.transAxes, fontsize=12)
        return
    valid = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'B_wave_nT', 'L_eff_m', 'I_bias_nA']).copy()
    residuals = valid['L_eff_m'].values - analytical_leff_model_physics_wavepower((valid['f_weighted_Hz'].values, valid['V_sc_volts'].values, valid['B_wave_nT'].values), *popt_wavepower)
    hb = ax.hexbin(valid['I_bias_nA'].values, residuals, gridsize=40, cmap='coolwarm', vmin=-5, vmax=5, mincnt=1)
    plt.colorbar(hb, ax=ax).set_label('Point Density')
    ax.axhline(0.0, color='black', linestyle='--', linewidth=1.8, label=r'Zero Residual ($\Delta L_{\mathrm{eff}} = 0\mathrm{m}$)')
    ax.set_ylim(-6, 6)
    ax.set_xlabel(r"Antenna Bias Current $I_{\mathrm{bias}}$ (nA)", fontweight='bold')
    ax.set_ylabel(r"Model 5 Residual $\Delta L_{\mathrm{eff}}$ (m)", fontweight='bold')
    ax.set_title(r"Residual Error vs. Antenna Bias Current", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

def draw_model6_parity(ax, df_clean, popt_model6, r2_model6=None):
    if popt_model6 is None or df_clean.empty: return
    valid = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'B_wave_nT', 'L_eff_m']).copy()
    if 'f_RC_corner_Hz' not in valid.columns or valid['f_RC_corner_Hz'].isna().all():
        valid['f_RC_corner_Hz'] = compute_in_situ_f_RC(valid['V_sc_volts'].values)

    b_perp = valid['B_wave_nT'].values * np.sin(np.radians(np.nan_to_num(valid['theta_kB_deg'].values, nan=90.0))) if 'theta_kB_deg' in valid.columns else valid['B_wave_nT'].values
    frc = valid['f_RC_corner_Hz'].values

    l_pred = analytical_leff_model6_physics((valid['f_weighted_Hz'].values, valid['V_sc_volts'].values, b_perp, frc), *popt_model6)

    hb = ax.hexbin(l_pred, valid['L_eff_m'].values, gridsize=45, cmap='plasma', mincnt=1, bins='log', extent=[0, 30, 0, 30])
    plt.colorbar(hb, ax=ax).set_label('Point Density (log10)')

    ref = np.linspace(0, 30, 100)
    ax.plot(ref, ref, 'r--', linewidth=2.0, label=r'1:1 Parity ($y = x$)')
    ax.fill_between(ref, ref - 2.0, ref + 2.0, color='red', alpha=0.10, label=r'$\pm 2\mathrm{m}$ Tolerance Band')

    r2_str = f"{r2_model6:.4f}" if r2_model6 is not None else "N/A"
    ax.text(0.05, 0.82, rf"Model 6 (RC Anchored)" + "\n" + rf"$R^2 = {r2_str}$" + "\n" + rf"$N = {len(valid):,}$ pts", 
            transform=ax.transAxes, fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_box_aspect(1)
    ax.set_xlabel(r"Model 6 Predicted $L_{eff}$ (m)", fontweight='bold')
    ax.set_ylabel(r"Observed $L_{eff}$ from Resonance (m)", fontweight='bold')
    ax.set_title(r"Model 6 Accuracy: Observed vs. Predicted $L_{eff}$", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)


def draw_model6_surface_and_overlay(ax, df_clean, popt_model6, min_events=5):
    if df_clean.empty or popt_model6 is None: return
    valid = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'B_wave_nT', 'L_eff_m']).copy()
    if 'f_RC_corner_Hz' not in valid.columns or valid['f_RC_corner_Hz'].isna().all():
        valid['f_RC_corner_Hz'] = compute_in_situ_f_RC(valid['V_sc_volts'].values)

    b_perp = valid['B_wave_nT'].values * np.sin(np.radians(np.nan_to_num(valid['theta_kB_deg'].values, nan=90.0))) if 'theta_kB_deg' in valid.columns else valid['B_wave_nT'].values
    valid['B_perp_nT'] = b_perp

    ret_med_L = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['L_eff_m'], statistic='median', bins=[35, 35])
    ret_med_B = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['B_perp_nT'], statistic='median', bins=[35, 35])
    ret_med_RC = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['f_RC_corner_Hz'], statistic='median', bins=[35, 35])
    ret_cnt   = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['L_eff_m'], statistic='count', bins=[35, 35])

    f_edges, v_edges = ret_med_L.x_edge, ret_med_L.y_edge
    mesh_f, mesh_v = np.meshgrid(f_edges, v_edges)
    X_c, Y_c = np.meshgrid(0.5 * (f_edges[:-1] + f_edges[1:]), 0.5 * (v_edges[:-1] + v_edges[1:]))
    surf_L = ret_med_L.statistic.T
    surf_L[ret_cnt.statistic.T < min_events] = np.nan
    pcm = ax.pcolormesh(mesh_f, mesh_v, surf_L, cmap='plasma', vmin=0, vmax=20, shading='flat')
    plt.colorbar(pcm, ax=ax).set_label(rf'Empirical Median $L_{{eff}}$ (m) [$N \geq {min_events}$]')

    RC_grid = np.nan_to_num(ret_med_RC.statistic.T, nan=10.0)
    B_grid = np.nan_to_num(ret_med_B.statistic.T, nan=1.0)

    contours = ax.contour(X_c, Y_c, analytical_leff_model6_physics((X_c, Y_c, B_grid, RC_grid), *popt_model6), levels=[2.0, 3.5, 5.0, 7.5, 10.0, 15.0], colors='cyan', linewidths=1.8, linestyles='--')
    ax.clabel(contours, fmt='%.1fm', inline=True, fontsize=9.5)
    ax.set_xlabel("Wave Frequency (Hz)", fontweight='bold')
    ax.set_ylabel(r"Spacecraft Potential $V_{sc}$ (V)", fontweight='bold')
    ax.set_title(r"Data Map with Overlaid Model 6 Contours ($RC$ Anchored)", fontweight='bold')
    ax.grid(True, alpha=0.3)


def draw_model6_2d_residuals(ax, df_clean, popt_model6, min_events=5):
    if df_clean.empty or popt_model6 is None: return
    valid = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'B_wave_nT', 'L_eff_m']).copy()
    if 'f_RC_corner_Hz' not in valid.columns or valid['f_RC_corner_Hz'].isna().all():
        valid['f_RC_corner_Hz'] = compute_in_situ_f_RC(valid['V_sc_volts'].values)

    b_perp = valid['B_wave_nT'].values * np.sin(np.radians(np.nan_to_num(valid['theta_kB_deg'].values, nan=90.0))) if 'theta_kB_deg' in valid.columns else valid['B_wave_nT'].values
    valid['B_perp_nT'] = b_perp

    ret_med_L = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['L_eff_m'], statistic='median', bins=[35, 35])
    ret_med_B = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['B_perp_nT'], statistic='median', bins=[35, 35])
    ret_med_RC = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['f_RC_corner_Hz'], statistic='median', bins=[35, 35])
    ret_cnt   = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['L_eff_m'], statistic='count', bins=[35, 35])

    f_edges, v_edges = ret_med_L.x_edge, ret_med_L.y_edge
    mesh_f, mesh_v = np.meshgrid(f_edges, v_edges)
    X_c, Y_c = np.meshgrid(0.5 * (f_edges[:-1] + f_edges[1:]), 0.5 * (v_edges[:-1] + v_edges[1:]))

    RC_grid = np.nan_to_num(ret_med_RC.statistic.T, nan=10.0)
    B_grid = np.nan_to_num(ret_med_B.statistic.T, nan=1.0)

    residual_matrix = ret_med_L.statistic.T - analytical_leff_model6_physics((X_c, Y_c, B_grid, RC_grid), *popt_model6)
    residual_matrix[ret_cnt.statistic.T < min_events] = np.nan

    pcm = ax.pcolormesh(mesh_f, mesh_v, residual_matrix, cmap='coolwarm', vmin=-5.0, vmax=5.0, shading='flat')
    plt.colorbar(pcm, ax=ax).set_label(r'Residual $\Delta L_{eff}$ (Data - Model 6) [m]')
    ax.axhline(popt_model6[2], color='black', linestyle=':', label=rf'$V_0 = {popt_model6[2]:.2f}\mathrm{{V}}$')
    ax.set_xlabel("Wave Frequency (Hz)", fontweight='bold')
    ax.set_ylabel(r"Spacecraft Potential $V_{sc}$ (V)", fontweight='bold')
    ax.set_title(r"Model 6 Residual Map: $L_{\mathrm{data}} - L_{\mathrm{model6}}$", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
def draw_leff_vs_deltaB_with_model6(ax, df_clean, popt_model6):
    """PANEL: AC Sheath Damping vs. deltaB_perp using V_sc Slices for Model 6."""
    if 'B_wave_nT' not in df_clean.columns or 'V_sc_volts' not in df_clean.columns or popt_model6 is None:
        ax.text(0.5, 0.5, "Missing required columns for Model 6 damping plot.", ha='center', va='center', transform=ax.transAxes)
        return

    valid = df_clean.dropna(subset=['B_wave_nT', 'V_sc_volts', 'f_weighted_Hz', 'L_eff_m']).copy()
    valid = valid[(valid['B_wave_nT'] > 0) & (valid['L_eff_m'] > 0)]
    
    b_perp = valid['B_wave_nT'].values * np.sin(np.radians(np.nan_to_num(valid['theta_kB_deg'].values, nan=90.0))) if 'theta_kB_deg' in valid.columns else valid['B_wave_nT'].values
    valid['B_perp_nT'] = b_perp

    valid['vsc_slice'] = pd.cut(
        valid['V_sc_volts'], 
        bins=[-np.inf, 1.0, 3.5, 6.5, np.inf], 
        labels=[r'Low $V_{sc} \leq 1.0\mathrm{V}$', r'Mod $1.0 < V_{sc} \leq 3.5\mathrm{V}$', r'Peak $3.5 < V_{sc} \leq 6.5\mathrm{V}$', r'High $V_{sc} > 6.5\mathrm{V}$']
    )
    colors_list = ['purple', 'teal', 'darkorange', 'crimson']
    b_eval = np.logspace(-3, 1.3, 100)

    for (slice_name, group), col in zip(valid.groupby('vsc_slice', observed=False), colors_list):
        if len(group) < 15: continue
        b_bins = np.logspace(np.log10(group['B_perp_nT'].min()), np.log10(group['B_perp_nT'].max()), 10)
        group_copy = group.copy()
        group_copy['b_bin'] = pd.cut(group_copy['B_perp_nT'], b_bins)
        b_centers = 0.5 * (b_bins[:-1] + b_bins[1:])
        
        binned_med = group_copy.groupby('b_bin', observed=False)['L_eff_m'].median()
        binned_q25 = group_copy.groupby('b_bin', observed=False)['L_eff_m'].quantile(0.25)
        binned_q75 = group_copy.groupby('b_bin', observed=False)['L_eff_m'].quantile(0.75)
        cnt_leff   = group_copy.groupby('b_bin', observed=False)['L_eff_m'].count()
        
        med_vals, q25_vals, q75_vals = binned_med.values.copy(), binned_q25.values.copy(), binned_q75.values.copy()
        mask_valid = cnt_leff.values >= 3
        med_vals[~mask_valid], q25_vals[~mask_valid], q75_vals[~mask_valid] = np.nan, np.nan, np.nan
        
        ax.plot(b_centers, med_vals, color=col, linewidth=1.8, marker='o', markersize=4, label=f'{slice_name}')
        ax.fill_between(b_centers, q25_vals, q75_vals, color=col, alpha=0.15)
        
        v_med = group['V_sc_volts'].median()
        f_med = group['f_weighted_Hz'].median()
        frc_med = compute_in_situ_f_RC(v_med)
        if np.isfinite(v_med) and np.isfinite(f_med):
            l_model = analytical_leff_model6_physics((f_med, v_med, b_eval, frc_med), *popt_model6)
            ax.plot(b_eval, l_model, color=col, linestyle='--', linewidth=1.8, alpha=0.90)

    L_base_val = popt_model6[0] if popt_model6 is not None else 1.57
    ax.axhline(L_base_val, color='gray', linestyle=':', label=rf'$L_{{base}} = {L_base_val:.2f}\mathrm{{m}}$')
    ax.set_xscale('log')
    ax.set_xlabel(r"Perpendicular Wave Fluctuation $\delta B_\perp$ (nT)", fontweight='bold')
    ax.set_ylabel(r"Effective Antenna Length $L_{\mathrm{eff}}$ (m)", fontweight='bold')
    ax.set_title(r"Model 6: AC Sheath Damping vs. Perpendicular Wave Fluctuation $\delta B_\perp$", fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, ncol=2)


def draw_literature_comparison(ax, df_clean, popt_model6=None, popt_wavepower=None, popt_model7=None, popt_model8=None, df_lit=None):
    valid = df_clean.dropna(subset=['f_weighted_Hz', 'L_eff_m']).copy()
    ax.hexbin(valid['f_weighted_Hz'], valid['L_eff_m'], xscale='log', yscale='linear', gridsize=45, cmap='Greys', mincnt=1, bins='log', alpha=0.20)

    if df_lit is not None and not df_lit.empty:
        mozer_df = df_lit[df_lit['paper'] == 'Mozer+2020'].copy()
    else:
        mozer_df = pd.DataFrame([
            {"event": "1Hz_Turbulence", "f_sc_Hz": 1.0,  "L_eff_m": 1.20, "L_eff_err": 0.50, "delta_B_nT": 10.0, "V_sc_volts": 0.583, "type": "MHD"},
            {"event": "3Hz_MHD",        "f_sc_Hz": 3.2,  "L_eff_m": 0.98, "L_eff_err": 0.31, "delta_B_nT": 1.4,  "V_sc_volts": 0.515, "type": "MHD"},
            {"event": "4Hz_AIC",        "f_sc_Hz": 4.3,  "L_eff_m": 2.78, "L_eff_err": 0.51, "delta_B_nT": 3.0,  "V_sc_volts": 0.320, "type": "MHD"},
            {"event": "4Hz_Whistler",   "f_sc_Hz": 4.5,  "L_eff_m": 1.68, "L_eff_err": 0.51, "delta_B_nT": 3.0,  "V_sc_volts": 0.320, "type": "Whistler"},
            {"event": "10Hz_Whistler",  "f_sc_Hz": 10.5, "L_eff_m": 1.51, "L_eff_err": 0.50, "delta_B_nT": 1.5,  "V_sc_volts": 0.650, "type": "Whistler"},
            {"event": "20Hz_Whistler",  "f_sc_Hz": 20.0, "L_eff_m": 3.72, "L_eff_err": 0.71, "delta_B_nT": 0.3,  "V_sc_volts": 0.650, "type": "Whistler"},
            {"event": "35Hz_Whistler",  "f_sc_Hz": 35.0, "L_eff_m": 3.75, "L_eff_err": 0.70, "delta_B_nT": 0.25, "V_sc_volts": 0.650, "type": "Whistler"},
            {"event": "59Hz_Whistler",  "f_sc_Hz": 59.0, "L_eff_m": 3.56, "L_eff_err": 0.51, "delta_B_nT": 0.22, "V_sc_volts": 0.306, "type": "Whistler"},
        ])

    if 'type' not in mozer_df.columns:
        mozer_df['type'] = mozer_df['event'].apply(
            lambda x: 'MHD' if any(k in str(x).lower() for k in ['mhd', '1hz', '3hz', 'aic', 'turbulence']) else 'Whistler'
        )

    mhd_mask, wh_mask = mozer_df['type'] == 'MHD', mozer_df['type'] == 'Whistler'
    mozer_df['V_wave_volts'] = [ (r['delta_B_nT'] * 1e-9 * 150e3) * 3.5 if 'V_wave_volts' not in r or not np.isfinite(r['V_wave_volts']) else r['V_wave_volts'] for _, r in mozer_df.iterrows() ]

    if popt_wavepower is not None:
        mozer_df['m5_pred'] = [analytical_leff_model_physics_wavepower((r['f_sc_Hz'], r['V_sc_volts'], r['delta_B_nT']), *popt_wavepower) if np.isfinite(r['V_sc_volts']) else np.nan for _, r in mozer_df.iterrows()]
    
    if popt_model6 is not None:
        f_rc_lit = compute_in_situ_f_RC(mozer_df['V_sc_volts'].values)
        mozer_df['m6_pred'] = [
            analytical_leff_model6_physics((r['f_sc_Hz'], r['V_sc_volts'], r['delta_B_nT'], frc), *popt_model6) 
            if np.isfinite(r['V_sc_volts']) else np.nan 
            for (_, r), frc in zip(mozer_df.iterrows(), f_rc_lit)
        ]
        
    if popt_model7 is not None:
        mozer_df['m7_pred'] = [analytical_leff_model7_physics((r['f_sc_Hz'], r['V_sc_volts'], r['V_wave_volts']), *popt_model7) if np.isfinite(r['V_sc_volts']) else np.nan for _, r in mozer_df.iterrows()]

    if popt_model8 is not None:
        r_lit = mozer_df['r_au'].values if 'r_au' in mozer_df.columns else np.ones(len(mozer_df))
        f_rc_lit = compute_in_situ_f_RC(mozer_df['V_sc_volts'].values, r_lit)
        bx_lit = mozer_df['theta_BX_deg'].values if 'theta_BX_deg' in mozer_df.columns else np.full(len(mozer_df), 90.0)
        sin2_bx_lit = np.sin(np.radians(np.nan_to_num(bx_lit, nan=90.0)))**2
        
        mozer_df['m8_pred'] = [
            analytical_leff_model8_wake((r['f_sc_Hz'], r['V_sc_volts'], r['delta_B_nT'], frc, sbx), *popt_model8) 
            if np.isfinite(r['V_sc_volts']) else np.nan 
            for (_, r), frc, sbx in zip(mozer_df.iterrows(), f_rc_lit, sin2_bx_lit)
        ]

    ax.errorbar(mozer_df.loc[wh_mask, 'f_sc_Hz'], mozer_df.loc[wh_mask, 'L_eff_m'], yerr=mozer_df.loc[wh_mask, 'L_eff_err'], fmt='sk', markersize=6, capsize=3, label='Mozer+ (2020) Whistler', zorder=5)
    ax.errorbar(mozer_df.loc[mhd_mask, 'f_sc_Hz'], mozer_df.loc[mhd_mask, 'L_eff_m'], yerr=mozer_df.loc[mhd_mask, 'L_eff_err'], fmt='^r', markersize=6, capsize=3, label='Mozer+ (2020) MHD-Ion', zorder=5)

    if popt_wavepower is not None:
        ax.plot(mozer_df.loc[mhd_mask, 'f_sc_Hz'], mozer_df.loc[mhd_mask, 'm5_pred'], 'b-', linewidth=1.5, marker='o', label='Model 5 (dB Damped)', zorder=4)
        ax.plot(mozer_df.loc[wh_mask, 'f_sc_Hz'], mozer_df.loc[wh_mask, 'm5_pred'], 'b--', linewidth=1.5, marker='o', zorder=4)

    if popt_model6 is not None:
        ax.plot(mozer_df.loc[mhd_mask, 'f_sc_Hz'], mozer_df.loc[mhd_mask, 'm6_pred'], 'r:', linewidth=1.5, marker='x', label=r'Model 6 ($\delta B_{\perp}$ Damped)', zorder=5)
        ax.plot(mozer_df.loc[wh_mask, 'f_sc_Hz'], mozer_df.loc[wh_mask, 'm6_pred'], 'r:', linewidth=1.5, marker='x', zorder=5)

    if popt_model7 is not None:
        ax.plot(mozer_df.loc[mhd_mask, 'f_sc_Hz'], mozer_df.loc[mhd_mask, 'm7_pred'], 'g-', linewidth=2.0, marker='s', label=r'Model 7 ($\delta V_{\mathrm{raw}}$ Damped)', zorder=6)
        ax.plot(mozer_df.loc[wh_mask, 'f_sc_Hz'], mozer_df.loc[wh_mask, 'm7_pred'], 'g--', linewidth=2.0, marker='s', zorder=6)

    if popt_model8 is not None:
        ax.plot(mozer_df.loc[mhd_mask, 'f_sc_Hz'], mozer_df.loc[mhd_mask, 'm8_pred'], color='purple', linestyle='-.', linewidth=2.0, marker='d', label='Model 8 (EUV + Wake)', zorder=7)
        ax.plot(mozer_df.loc[wh_mask, 'f_sc_Hz'], mozer_df.loc[wh_mask, 'm8_pred'], color='purple', linestyle='--', linewidth=2.0, marker='d', zorder=7)

    # Commneting out the Karbashewski data since that is not in our frequency range
    karb_f = np.array([50.0, 75.0, 100.0, 120.0, 135.0, 150.0, 170.0, 190.0, 210.0])
    karb_L = np.array([5.2, 10.5, 17.8, 23.5, 27.8, 22.0, 16.0, 14.8, 14.2])
    #ax.plot(karb_f, karb_L, 'o-', color='purple', linewidth=1.8, markersize=5, label='Karbashewski+ (2023)')

    ax.axhline(3.5, color='gray', linestyle=':', label=r'$L_{\mathrm{phys}} = 3.5\mathrm{m}$')
    ax.set_xscale('log')
    ax.set_xlim(0.3, 300.0)
    ax.set_ylim(0.0, 35.0)
    ax.set_xlabel("Spacecraft Frame Frequency $f$ (Hz)", fontweight='bold')
    ax.set_ylabel(r"Effective Antenna Length $L_{\mathrm{eff}}$ (m)", fontweight='bold')
    ax.set_title("Literature Comparison: Context-Matched Models vs. Literature", fontweight='bold')
    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    ax.legend(loc='upper right', fontsize=8, ncol=2)

def draw_leff_vs_Vwave_with_model7(ax, df_clean, popt_model7):
    """PANEL: AC Sheath Damping plotted against raw terminal voltage V_wave_volts using Quantile V_sc Slices."""
    if 'V_wave_volts' not in df_clean.columns or 'V_sc_volts' not in df_clean.columns or popt_model7 is None:
        ax.text(0.5, 0.5, "Missing 'V_wave_volts' in CSV for Model 7.", ha='center', va='center', transform=ax.transAxes)
        return

    valid = df_clean.dropna(subset=['V_wave_volts', 'V_sc_volts', 'f_weighted_Hz', 'L_eff_m']).copy()
    valid = valid[(valid['V_wave_volts'] > 0) & (valid['L_eff_m'] > 0)]

    # EXPLICIT ARRAY COPY FIX FOR READ-ONLY WINDOWS NUMPY BUFFERS
    vsc_q = np.array(valid['V_sc_volts'].quantile([0.0, 0.25, 0.50, 0.75, 1.0]).values, copy=True)
    vsc_q[0], vsc_q[-1] = -np.inf, np.inf
    
    vsc_labels = [
        rf'$Q_1$ $V_{{sc}} \leq {vsc_q[1]:.2f}\mathrm{{V}}$',
        rf'$Q_2$ ${vsc_q[1]:.2f} < V_{{sc}} \leq {vsc_q[2]:.2f}\mathrm{{V}}$',
        rf'$Q_3$ ${vsc_q[2]:.2f} < V_{{sc}} \leq {vsc_q[3]:.2f}\mathrm{{V}}$',
        rf'$Q_4$ $V_{{sc}} > {vsc_q[3]:.2f}\mathrm{{V}}$'
    ]
    colors_list = ['purple', 'teal', 'darkorange', 'crimson']
    valid['vsc_slice'] = pd.cut(valid['V_sc_volts'], bins=vsc_q, labels=vsc_labels)

    vw_eval = np.logspace(np.log10(valid['V_wave_volts'].min()), np.log10(valid['V_wave_volts'].max()), 100)

    for (slice_name, group), col in zip(valid.groupby('vsc_slice', observed=False), colors_list):
        if len(group) < 15: continue
        vw_bins = np.logspace(np.log10(group['V_wave_volts'].min()), np.log10(group['V_wave_volts'].max()), 10)
        group_copy = group.copy()
        group_copy['vw_bin'] = pd.cut(group_copy['V_wave_volts'], vw_bins)
        vw_centers = 0.5 * (vw_bins[:-1] + vw_bins[1:])
        
        binned_med = group_copy.groupby('vw_bin', observed=False)['L_eff_m'].median()
        binned_q25 = group_copy.groupby('vw_bin', observed=False)['L_eff_m'].quantile(0.25)
        binned_q75 = group_copy.groupby('vw_bin', observed=False)['L_eff_m'].quantile(0.75)
        cnt_leff   = group_copy.groupby('vw_bin', observed=False)['L_eff_m'].count()
        
        med_vals, q25_vals, q75_vals = binned_med.values.copy(), binned_q25.values.copy(), binned_q75.values.copy()
        mask_valid = cnt_leff.values >= 3
        med_vals[~mask_valid], q25_vals[~mask_valid], q75_vals[~mask_valid] = np.nan, np.nan, np.nan
        
        ax.plot(vw_centers, med_vals, color=col, linewidth=1.8, marker='o', markersize=4, label=f'{slice_name}')
        ax.fill_between(vw_centers, q25_vals, q75_vals, color=col, alpha=0.15)
        
        v_med = group['V_sc_volts'].median()
        f_med = group['f_weighted_Hz'].median()
        if np.isfinite(v_med) and np.isfinite(f_med):
            l_model = analytical_leff_model7_physics((f_med, v_med, vw_eval), *popt_model7)
            ax.plot(vw_eval, l_model, color=col, linestyle='--', linewidth=1.8, alpha=0.90)

    L_base_val = popt_model7[0] if popt_model7 is not None else 1.57
    ax.axhline(L_base_val, color='gray', linestyle=':', label=rf'$L_{{base}} = {L_base_val:.2f}\mathrm{{m}}$')
    ax.set_xscale('log')
    ax.set_xlabel(r"Raw Terminal AC Voltage Swing $\delta V_{\mathrm{raw}}$ (Volts)", fontweight='bold')
    ax.set_ylabel(r"Effective Antenna Length $L_{\mathrm{eff}}$ (m)", fontweight='bold')
    ax.set_title(r"Model 7: AC Sheath Damping vs. Preamplifier Terminal Voltage $\delta V_{\mathrm{raw}}$", fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, ncol=2)

def draw_model7_parity(ax, df_clean, popt_model7, r2_model7=None):
    """PANEL: Observed vs. Model 7 Predicted L_eff Parity Plot."""
    if popt_model7 is None or df_clean.empty or 'V_wave_volts' not in df_clean.columns:
        ax.text(0.5, 0.5, "Model 7 parity data unavailable.", ha='center', va='center', transform=ax.transAxes)
        return

    valid = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'V_wave_volts', 'L_eff_m']).copy()
    
    f_val = valid['f_weighted_Hz'].values
    v_val = valid['V_sc_volts'].values
    vw_val = valid['V_wave_volts'].values
    l_obs = valid['L_eff_m'].values

    l_pred = analytical_leff_model7_physics((f_val, v_val, vw_val), *popt_model7)

    hb = ax.hexbin(l_pred, l_obs, gridsize=45, cmap='plasma', mincnt=1, bins='log', extent=[0, 30, 0, 30])
    plt.colorbar(hb, ax=ax).set_label('Point Density (log10)')

    ref = np.linspace(0, 30, 100)
    ax.plot(ref, ref, 'r--', linewidth=2.0, label=r'1:1 Perfect Parity ($y = x$)')
    ax.fill_between(ref, ref - 2.0, ref + 2.0, color='red', alpha=0.10, label=r'$\pm 2\mathrm{m}$ Tolerance Band')

    r2_str = f"{r2_model7:.4f}" if r2_model7 is not None else "N/A"
    ax.text(0.05, 0.82, rf"Model 7 ($\delta V_{{\mathrm{{raw}}}}$ Damping)" + "\n" + rf"$R^2 = {r2_str}$" + "\n" + rf"$N = {len(valid):,}$ pts", 
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_xlabel(r"Model 7 Predicted $L_{\mathrm{eff}}$ (m)", fontweight='bold')
    ax.set_ylabel(r"Observed $L_{\mathrm{eff}}$ from Resonance (m)", fontweight='bold')
    ax.set_title(r"Model 7 Accuracy: Observed vs. Predicted $L_{\mathrm{eff}}$", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)

def draw_model7_surface_and_overlay(ax, df_clean, popt_model7, min_events=5):
    """PANEL: Empirical Data Map with Overlaid Model 7 Contours."""
    if df_clean.empty or popt_model7 is None or 'V_wave_volts' not in df_clean.columns:
        return

    valid = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'V_wave_volts', 'L_eff_m']).copy()

    ret_med_L = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['L_eff_m'], statistic='median', bins=[35, 35])
    ret_med_V = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['V_wave_volts'], statistic='median', bins=[35, 35])
    ret_cnt   = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['L_eff_m'], statistic='count', bins=[35, 35])

    f_edges, v_edges = ret_med_L.x_edge, ret_med_L.y_edge
    bin_f, bin_v = 0.5 * (f_edges[:-1] + f_edges[1:]), 0.5 * (v_edges[:-1] + v_edges[1:])
    mesh_f, mesh_v = np.meshgrid(f_edges, v_edges)
    X_c, Y_c = np.meshgrid(bin_f, bin_v)

    surf_L, surf_V, cnt_data = ret_med_L.statistic.T, ret_med_V.statistic.T, ret_cnt.statistic.T
    surf_L[cnt_data < min_events] = np.nan

    pcm = ax.pcolormesh(mesh_f, mesh_v, surf_L, cmap='plasma', vmin=0, vmax=20, shading='flat')
    plt.colorbar(pcm, ax=ax).set_label(rf'Empirical Median $L_{{eff}}$ (m) [$N \geq {min_events}$]')

    V_grid_filled = np.nan_to_num(surf_V, nan=0.01)
    Z_model_2d = analytical_leff_model7_physics((X_c, Y_c, V_grid_filled), *popt_model7)

    contours = ax.contour(X_c, Y_c, Z_model_2d, levels=[2.0, 3.5, 5.0, 7.5, 10.0, 15.0], colors='cyan', linewidths=1.8, linestyles='--')
    labels = ax.clabel(contours, fmt='%.1fm', inline=True, fontsize=9.5, inline_spacing=5)
    plt.setp(labels, path_effects=[path_effects.withStroke(linewidth=2.5, foreground='black')], color='cyan', fontweight='bold')

    ax.set_xlabel("Wave Frequency (Hz)", fontweight='bold')
    ax.set_ylabel(r"Spacecraft Potential $V_{sc}$ (V)", fontweight='bold')
    ax.set_title(r"Data Map with Overlaid Model 7 Contours ($\delta V_{\mathrm{raw}}$ Damped)", fontweight='bold')
    ax.grid(True, alpha=0.3)

def draw_model7_2d_residuals(ax, df_clean, popt_model7, min_events=5):
    """PANEL: 2D Model 7 Residual Map (Data - Model 7)."""
    if df_clean.empty or popt_model7 is None or 'V_wave_volts' not in df_clean.columns:
        return

    valid = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'V_wave_volts', 'L_eff_m']).copy()

    ret_med_L = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['L_eff_m'], statistic='median', bins=[35, 35])
    ret_med_V = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['V_wave_volts'], statistic='median', bins=[35, 35])
    ret_cnt   = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['L_eff_m'], statistic='count', bins=[35, 35])

    f_edges, v_edges = ret_med_L.x_edge, ret_med_L.y_edge
    bin_f, bin_v = 0.5 * (f_edges[:-1] + f_edges[1:]), 0.5 * (v_edges[:-1] + v_edges[1:])
    mesh_f, mesh_v = np.meshgrid(f_edges, v_edges)
    X_c, Y_c = np.meshgrid(bin_f, bin_v)

    surf_L, surf_V, cnt_data = ret_med_L.statistic.T, ret_med_V.statistic.T, ret_cnt.statistic.T

    V_grid_filled = np.nan_to_num(surf_V, nan=0.01)
    Z_model_2d = analytical_leff_model7_physics((X_c, Y_c, V_grid_filled), *popt_model7)

    residual_matrix = surf_L - Z_model_2d
    residual_matrix[cnt_data < min_events] = np.nan

    pcm = ax.pcolormesh(mesh_f, mesh_v, residual_matrix, cmap='coolwarm', vmin=-5.0, vmax=5.0, shading='flat')
    plt.colorbar(pcm, ax=ax).set_label(r'Residual $\Delta L_{\mathrm{eff}}$ (Data - Model 7) [m]')

    V0_val = popt_model7[2] if popt_model7 is not None else 5.50
    ax.axhline(V0_val, color='black', linestyle=':', label=rf'$V_0 = {V0_val:.2f}\mathrm{{V}}$')
    ax.set_xlabel("Wave Frequency (Hz)", fontweight='bold')
    ax.set_ylabel(r"Spacecraft Potential $V_{sc}$ (V)", fontweight='bold')
    ax.set_title(r"Model 7 Residual Map: $L_{\mathrm{data}} - L_{\mathrm{model7}}$", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

def draw_model_evolution_comparison_slide(df_clean, popt_vsc, popt_bandpass, popt_model7, r2_vsc, r2_bandpass, r2_model7, output_path):
    if df_clean.empty or popt_vsc is None or popt_bandpass is None or popt_model7 is None: return
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(r"Physics Progression: Residual Error Reduction ($\Delta L_{\mathrm{eff}} = L_{\mathrm{data}} - L_{\mathrm{model}}$)", fontweight='bold', fontsize=14, y=0.98)
    valid = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'V_wave_volts', 'L_eff_m'])
    f_vals, v_vals, vw_vals, l_vals = valid['f_weighted_Hz'].values, valid['V_sc_volts'].values, valid['V_wave_volts'].values, valid['L_eff_m'].values
    
    ret_med = binned_statistic_2d(f_vals, v_vals, l_vals, statistic='median', bins=[30, 30])
    ret_cnt = binned_statistic_2d(f_vals, v_vals, l_vals, statistic='count', bins=[30, 30])
    f_edges, v_edges = ret_med.x_edge, ret_med.y_edge
    bin_f, bin_v = 0.5 * (f_edges[:-1] + f_edges[1:]), 0.5 * (v_edges[:-1] + v_edges[1:])
    mesh_f, mesh_v = np.meshgrid(f_edges, v_edges)
    X_c, Y_c = np.meshgrid(bin_f, bin_v)
    surf_data, cnt_data = ret_med.statistic.T, ret_cnt.statistic.T

    z_m2 = analytical_leff_model_f_vsc((X_c, Y_c), *popt_vsc)
    res_m2 = surf_data - z_m2
    res_m2[cnt_data < 5] = np.nan
    ax1.pcolormesh(mesh_f, mesh_v, res_m2, cmap='coolwarm', vmin=-6, vmax=6)
    ax1.set_title(rf"1. Voltage Model ($R^2 = {r2_vsc:.4f}$)", fontweight='bold')
    ax1.set_ylabel(r"Spacecraft Potential $V_{sc}$ (V)", fontweight='bold')
    ax1.set_xlabel("Wave Frequency (Hz)", fontweight='bold')
    ax1.grid(True, alpha=0.3)

    z_m4 = analytical_leff_model_physics_bandpass((X_c, Y_c), *popt_bandpass)
    res_m4 = surf_data - z_m4
    res_m4[cnt_data < 5] = np.nan
    ax2.pcolormesh(mesh_f, mesh_v, res_m4, cmap='coolwarm', vmin=-6, vmax=6)
    ax2.set_title(rf"2. Physics Bandpass ($R^2 = {r2_bandpass:.4f}$)", fontweight='bold')
    ax2.set_xlabel("Wave Frequency (Hz)", fontweight='bold')
    ax2.grid(True, alpha=0.3)

    ret_vw = binned_statistic_2d(f_vals, v_vals, vw_vals, statistic='median', bins=[30, 30])
    VW_grid = ret_vw.statistic.T
    z_m7 = analytical_leff_model7_physics((X_c, Y_c, np.nan_to_num(VW_grid, nan=0.01)), *popt_model7)
    res_m7 = surf_data - z_m7
    res_m7[cnt_data < 5] = np.nan
    pcm = ax3.pcolormesh(mesh_f, mesh_v, res_m7, cmap='coolwarm', vmin=-6, vmax=6)
    ax3.set_title(rf"3. Full Physics + $\delta V_{{\mathrm{{raw}}}}$ Damping ($R^2 = {r2_model7:.4f}$)", fontweight='bold')
    ax3.set_xlabel("Wave Frequency (Hz)", fontweight='bold')
    ax3.grid(True, alpha=0.3)

    fig.colorbar(pcm, ax=[ax1, ax2, ax3], orientation='vertical', fraction=0.02, pad=0.02).set_label(r'Residual Error $\Delta L_{\mathrm{eff}}$ (Data - Model) [m]', fontweight='bold')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def draw_model8_parity(ax, df_clean, popt_model8, r2_model8=None):
    """PANEL: Observed vs. Model 8 Predicted L_eff Parity Plot."""
    if popt_model8 is None or df_clean.empty: return
    valid = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'B_wave_nT', 'L_eff_m']).copy()

    r_col = next((col for col in ['r_au', 'r_AU', 'radial_distance_AU', 'r_sun_au', 'r_sun'] if col in valid.columns), None)
    r_au_vals = valid[r_col].values if r_col else np.ones(len(valid))
    valid['f_RC_corner_Hz'] = compute_in_situ_f_RC(valid['V_sc_volts'].values, r_au_vals)

    bx_col = next((col for col in ['theta_BX_deg', 'theta_bx_deg', 'theta_B_X', 'theta_B_X_deg'] if col in valid.columns), None)
    valid['sin2_theta_bx'] = np.sin(np.radians(np.nan_to_num(valid[bx_col].values, nan=90.0)))**2 if bx_col else np.ones(len(valid))

    b_perp = valid['B_wave_nT'].values * np.sin(np.radians(np.nan_to_num(valid['theta_kB_deg'].values, nan=90.0))) if 'theta_kB_deg' in valid.columns else valid['B_wave_nT'].values
    frc = valid['f_RC_corner_Hz'].values
    sbx = valid['sin2_theta_bx'].values

    l_pred = analytical_leff_model8_wake((valid['f_weighted_Hz'].values, valid['V_sc_volts'].values, b_perp, frc, sbx), *popt_model8)

    hb = ax.hexbin(l_pred, valid['L_eff_m'].values, gridsize=45, cmap='plasma', mincnt=1, bins='log', extent=[0, 30, 0, 30])
    plt.colorbar(hb, ax=ax).set_label('Point Density (log10)')

    ref = np.linspace(0, 30, 100)
    ax.plot(ref, ref, 'r--', linewidth=2.0, label=r'1:1 Parity ($y = x$)')
    ax.fill_between(ref, ref - 2.0, ref + 2.0, color='red', alpha=0.10, label=r'$\pm 2\mathrm{m}$ Tolerance Band')

    r2_str = f"{r2_model8:.4f}" if r2_model8 is not None else "N/A"
    ax.text(0.05, 0.82, rf"Model 8 (1/r² Photoflux + Wake)" + "\n" + rf"$R^2 = {r2_str}$" + "\n" + rf"$N = {len(valid):,}$ pts", 
            transform=ax.transAxes, fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_box_aspect(1)
    ax.set_xlabel(r"Model 8 Predicted $L_{eff}$ (m)", fontweight='bold')
    ax.set_ylabel(r"Observed $L_{eff}$ from Resonance (m)", fontweight='bold')
    ax.set_title(r"Model 8 Accuracy: Observed vs. Predicted $L_{eff}$", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)


def draw_model8_surface_and_overlay(ax, df_clean, popt_model8, min_events=5):
    """PANEL: Empirical Data Map with Overlaid Model 8 Contours."""
    if df_clean.empty or popt_model8 is None: return
    valid = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'B_wave_nT', 'L_eff_m']).copy()

    r_col = next((col for col in ['r_au', 'r_AU', 'radial_distance_AU', 'r_sun_au', 'r_sun'] if col in valid.columns), None)
    r_au_vals = valid[r_col].values if r_col else np.ones(len(valid))
    valid['f_RC_corner_Hz'] = compute_in_situ_f_RC(valid['V_sc_volts'].values, r_au_vals)

    bx_col = next((col for col in ['theta_BX_deg', 'theta_bx_deg', 'theta_B_X', 'theta_B_X_deg'] if col in valid.columns), None)
    valid['sin2_theta_bx'] = np.sin(np.radians(np.nan_to_num(valid[bx_col].values, nan=90.0)))**2 if bx_col else np.ones(len(valid))

    b_perp = valid['B_wave_nT'].values * np.sin(np.radians(np.nan_to_num(valid['theta_kB_deg'].values, nan=90.0))) if 'theta_kB_deg' in valid.columns else valid['B_wave_nT'].values
    valid['B_perp_nT'] = b_perp

    ret_med_L = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['L_eff_m'], statistic='median', bins=[35, 35])
    ret_med_B = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['B_perp_nT'], statistic='median', bins=[35, 35])
    ret_med_RC = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['f_RC_corner_Hz'], statistic='median', bins=[35, 35])
    ret_med_BX = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['sin2_theta_bx'], statistic='median', bins=[35, 35])
    ret_cnt   = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['L_eff_m'], statistic='count', bins=[35, 35])

    f_edges, v_edges = ret_med_L.x_edge, ret_med_L.y_edge
    mesh_f, mesh_v = np.meshgrid(f_edges, v_edges)
    X_c, Y_c = np.meshgrid(0.5 * (f_edges[:-1] + f_edges[1:]), 0.5 * (v_edges[:-1] + v_edges[1:]))
    surf_L = ret_med_L.statistic.T
    surf_L[ret_cnt.statistic.T < min_events] = np.nan
    pcm = ax.pcolormesh(mesh_f, mesh_v, surf_L, cmap='plasma', vmin=0, vmax=20, shading='flat')
    plt.colorbar(pcm, ax=ax).set_label(rf'Empirical Median $L_{{eff}}$ (m) [$N \geq {min_events}$]')

    RC_grid = np.nan_to_num(ret_med_RC.statistic.T, nan=10.0)
    B_grid = np.nan_to_num(ret_med_B.statistic.T, nan=1.0)
    BX_grid = np.nan_to_num(ret_med_BX.statistic.T, nan=1.0)

    contours = ax.contour(X_c, Y_c, analytical_leff_model8_wake((X_c, Y_c, B_grid, RC_grid, BX_grid), *popt_model8), levels=[2.0, 3.5, 5.0, 7.5, 10.0, 15.0], colors='cyan', linewidths=1.8, linestyles='--')
    labels = ax.clabel(contours, fmt='%.1fm', inline=True, fontsize=9.5, inline_spacing=5)
    plt.setp(labels, path_effects=[path_effects.withStroke(linewidth=2.5, foreground='black')], color='cyan', fontweight='bold')

    ax.set_xlabel("Wave Frequency (Hz)", fontweight='bold')
    ax.set_ylabel(r"Spacecraft Potential $V_{sc}$ (V)", fontweight='bold')
    ax.set_title(r"Data Map with Overlaid Model 8 Contours (EUV + Wake)", fontweight='bold')
    ax.grid(True, alpha=0.3)


def draw_model8_2d_residuals(ax, df_clean, popt_model8, min_events=5):
    """PANEL: 2D Model 8 Residual Map (Data - Model 8)."""
    if df_clean.empty or popt_model8 is None: return
    valid = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'B_wave_nT', 'L_eff_m']).copy()

    r_col = next((col for col in ['r_au', 'r_AU', 'radial_distance_AU', 'r_sun_au', 'r_sun'] if col in valid.columns), None)
    r_au_vals = valid[r_col].values if r_col else np.ones(len(valid))
    valid['f_RC_corner_Hz'] = compute_in_situ_f_RC(valid['V_sc_volts'].values, r_au_vals)

    bx_col = next((col for col in ['theta_BX_deg', 'theta_bx_deg', 'theta_B_X', 'theta_B_X_deg'] if col in valid.columns), None)
    valid['sin2_theta_bx'] = np.sin(np.radians(np.nan_to_num(valid[bx_col].values, nan=90.0)))**2 if bx_col else np.ones(len(valid))

    b_perp = valid['B_wave_nT'].values * np.sin(np.radians(np.nan_to_num(valid['theta_kB_deg'].values, nan=90.0))) if 'theta_kB_deg' in valid.columns else valid['B_wave_nT'].values
    valid['B_perp_nT'] = b_perp

    ret_med_L = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['L_eff_m'], statistic='median', bins=[35, 35])
    ret_med_B = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['B_perp_nT'], statistic='median', bins=[35, 35])
    ret_med_RC = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['f_RC_corner_Hz'], statistic='median', bins=[35, 35])
    ret_med_BX = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['sin2_theta_bx'], statistic='median', bins=[35, 35])
    ret_cnt   = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['L_eff_m'], statistic='count', bins=[35, 35])

    f_edges, v_edges = ret_med_L.x_edge, ret_med_L.y_edge
    mesh_f, mesh_v = np.meshgrid(f_edges, v_edges)
    X_c, Y_c = np.meshgrid(0.5 * (f_edges[:-1] + f_edges[1:]), 0.5 * (v_edges[:-1] + v_edges[1:]))

    RC_grid = np.nan_to_num(ret_med_RC.statistic.T, nan=10.0)
    B_grid = np.nan_to_num(ret_med_B.statistic.T, nan=1.0)
    BX_grid = np.nan_to_num(ret_med_BX.statistic.T, nan=1.0)

    residual_matrix = ret_med_L.statistic.T - analytical_leff_model8_wake((X_c, Y_c, B_grid, RC_grid, BX_grid), *popt_model8)
    residual_matrix[ret_cnt.statistic.T < min_events] = np.nan

    pcm = ax.pcolormesh(mesh_f, mesh_v, residual_matrix, cmap='coolwarm', vmin=-5.0, vmax=5.0, shading='flat')
    plt.colorbar(pcm, ax=ax).set_label(r'Residual $\Delta L_{eff}$ (Data - Model 8) [m]')
    ax.axhline(popt_model8[3], color='black', linestyle=':', label=rf'$V_0 = {popt_model8[3]:.2f}\mathrm{{V}}$')
    ax.set_xlabel("Wave Frequency (Hz)", fontweight='bold')
    ax.set_ylabel(r"Spacecraft Potential $V_{sc}$ (V)", fontweight='bold')
    ax.set_title(r"Model 8 Residual Map: $L_{\mathrm{data}} - L_{\mathrm{model8}}$", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')


def draw_leff_vs_deltaB_with_model8(ax, df_clean, popt_model8):
    """PANEL: AC Sheath Damping vs. deltaB_perp using V_sc Slices for Model 8."""
    if 'B_wave_nT' not in df_clean.columns or 'V_sc_volts' not in df_clean.columns or popt_model8 is None:
        ax.text(0.5, 0.5, "Missing required columns for Model 8 damping plot.", ha='center', va='center', transform=ax.transAxes)
        return

    valid = df_clean.dropna(subset=['B_wave_nT', 'V_sc_volts', 'f_weighted_Hz', 'L_eff_m']).copy()
    valid = valid[(valid['B_wave_nT'] > 0) & (valid['L_eff_m'] > 0)]

    r_col = next((col for col in ['r_au', 'r_AU', 'radial_distance_AU', 'r_sun_au', 'r_sun'] if col in valid.columns), None)
    r_au_vals = valid[r_col].values if r_col else np.ones(len(valid))
    valid['f_RC_corner_Hz'] = compute_in_situ_f_RC(valid['V_sc_volts'].values, r_au_vals)

    bx_col = next((col for col in ['theta_BX_deg', 'theta_bx_deg', 'theta_B_X', 'theta_B_X_deg'] if col in valid.columns), None)
    valid['sin2_theta_bx'] = np.sin(np.radians(np.nan_to_num(valid[bx_col].values, nan=90.0)))**2 if bx_col else np.ones(len(valid))

    b_perp = valid['B_wave_nT'].values * np.sin(np.radians(np.nan_to_num(valid['theta_kB_deg'].values, nan=90.0))) if 'theta_kB_deg' in valid.columns else valid['B_wave_nT'].values
    valid['B_perp_nT'] = b_perp

    valid['vsc_slice'] = pd.cut(
        valid['V_sc_volts'], 
        bins=[-np.inf, 1.0, 3.5, 6.5, np.inf], 
        labels=[r'Low $V_{sc} \leq 1.0\mathrm{V}$', r'Mod $1.0 < V_{sc} \leq 3.5\mathrm{V}$', r'Peak $3.5 < V_{sc} \leq 6.5\mathrm{V}$', r'High $V_{sc} > 6.5\mathrm{V}$']
    )
    colors_list = ['purple', 'teal', 'darkorange', 'crimson']
    b_eval = np.logspace(-3, 1.3, 100)

    for (slice_name, group), col in zip(valid.groupby('vsc_slice', observed=False), colors_list):
        if len(group) < 15: continue
        b_bins = np.logspace(np.log10(group['B_perp_nT'].min()), np.log10(group['B_perp_nT'].max()), 10)
        group_copy = group.copy()
        group_copy['b_bin'] = pd.cut(group_copy['B_perp_nT'], b_bins)
        b_centers = 0.5 * (b_bins[:-1] + b_bins[1:])
        
        binned_med = group_copy.groupby('b_bin', observed=False)['L_eff_m'].median()
        binned_q25 = group_copy.groupby('b_bin', observed=False)['L_eff_m'].quantile(0.25)
        binned_q75 = group_copy.groupby('b_bin', observed=False)['L_eff_m'].quantile(0.75)
        cnt_leff   = group_copy.groupby('b_bin', observed=False)['L_eff_m'].count()
        
        med_vals, q25_vals, q75_vals = binned_med.values.copy(), binned_q25.values.copy(), binned_q75.values.copy()
        mask_valid = cnt_leff.values >= 3
        med_vals[~mask_valid], q25_vals[~mask_valid], q75_vals[~mask_valid] = np.nan, np.nan, np.nan
        
        ax.plot(b_centers, med_vals, color=col, linewidth=1.8, marker='o', markersize=4, label=f'{slice_name}')
        ax.fill_between(b_centers, q25_vals, q75_vals, color=col, alpha=0.15)
        
        v_med = group['V_sc_volts'].median()
        f_med = group['f_weighted_Hz'].median()
        r_med = group[r_col].median() if r_col else 1.0
        bx_med = group['sin2_theta_bx'].median()
        frc_med = compute_in_situ_f_RC(v_med, r_med)
        if np.isfinite(v_med) and np.isfinite(f_med):
            l_model = analytical_leff_model8_wake((f_med, v_med, b_eval, frc_med, bx_med), *popt_model8)
            ax.plot(b_eval, l_model, color=col, linestyle='--', linewidth=1.8, alpha=0.90)

    L_base_val = popt_model8[0] if popt_model8 is not None else 1.57
    ax.axhline(L_base_val, color='gray', linestyle=':', label=rf'$L_{{base}} = {L_base_val:.2f}\mathrm{{m}}$')
    ax.set_xscale('log')
    ax.set_xlabel(r"Perpendicular Wave Fluctuation $\delta B_\perp$ (nT)", fontweight='bold')
    ax.set_ylabel(r"Effective Antenna Length $L_{\mathrm{eff}}$ (m)", fontweight='bold')
    ax.set_title(r"Model 8: AC Sheath Damping vs. Perpendicular Wave Fluctuation $\delta B_\perp$", fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    
# =========================================================================
# MODEL 9 DRAWING FUNCTIONS (Mondal Tanh Transfer Function)
# =========================================================================

def draw_model9_parity(ax, df_clean, popt_model9, r2_model9=None):
    """PANEL: Observed vs. Model 9 Predicted L_eff Parity Plot."""
    if popt_model9 is None or df_clean.empty: return
    valid = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'B_wave_nT', 'L_eff_m']).copy()

    bx_col = next((col for col in ['theta_BX_deg', 'theta_bx_deg'] if col in valid.columns), None)
    valid['sin2_theta_bx'] = np.sin(np.radians(np.nan_to_num(valid[bx_col].values, nan=90.0)))**2 if bx_col else np.ones(len(valid))

    b_col = next((c for c in ['B_high_freq_nT', 'B_wave_nT'] if c in valid.columns), 'B_wave_nT')
    raw_b = valid[b_col].values
    b_perp = raw_b * np.sin(np.radians(np.nan_to_num(valid['theta_kB_deg'].values, nan=90.0))) if 'theta_kB_deg' in valid.columns else raw_b
    sbx = valid['sin2_theta_bx'].values

    l_pred = analytical_leff_model9_tanh((valid['f_weighted_Hz'].values, valid['V_sc_volts'].values, b_perp, sbx), *popt_model9)

    hb = ax.hexbin(l_pred, valid['L_eff_m'].values, gridsize=45, cmap='plasma', mincnt=1, bins='log', extent=[0, 30, 0, 30])
    plt.colorbar(hb, ax=ax).set_label('Point Density (log10)')

    ref = np.linspace(0, 30, 100)
    ax.plot(ref, ref, 'r--', linewidth=2.0, label=r'1:1 Parity ($y = x$)')
    ax.fill_between(ref, ref - 2.0, ref + 2.0, color='red', alpha=0.10, label=r'$\pm 2\mathrm{m}$ Tolerance Band')

    r2_str = f"{r2_model9:.4f}" if r2_model9 is not None else "N/A"
    ax.text(0.05, 0.82, rf"Model 9 (Mondal Tanh)" + "\n" + rf"$R^2 = {r2_str}$" + "\n" + rf"$N = {len(valid):,}$ pts", 
            transform=ax.transAxes, fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_box_aspect(1)
    ax.set_xlabel(r"Model 9 Predicted $L_{eff}$ (m)", fontweight='bold')
    ax.set_ylabel(r"Observed $L_{eff}$ from Resonance (m)", fontweight='bold')
    ax.set_title(r"Model 9 Accuracy: Observed vs. Predicted $L_{eff}$", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)


def draw_model9_surface_and_overlay(ax, df_clean, popt_model9, min_events=5):
    """PANEL: Empirical Data Map with Overlaid Model 9 Contours."""
    if df_clean.empty or popt_model9 is None: return
    valid = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'B_wave_nT', 'L_eff_m']).copy()

    bx_col = next((col for col in ['theta_BX_deg', 'theta_bx_deg'] if col in valid.columns), None)
    valid['sin2_theta_bx'] = np.sin(np.radians(np.nan_to_num(valid[bx_col].values, nan=90.0)))**2 if bx_col else np.ones(len(valid))

    b_col = next((c for c in ['B_high_freq_nT', 'B_wave_nT'] if c in valid.columns), 'B_wave_nT')
    raw_b = valid[b_col].values
    b_perp = raw_b * np.sin(np.radians(np.nan_to_num(valid['theta_kB_deg'].values, nan=90.0))) if 'theta_kB_deg' in valid.columns else raw_b
    valid['B_perp_nT'] = b_perp

    ret_med_L = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['L_eff_m'], statistic='median', bins=[35, 35])
    ret_med_B = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['B_perp_nT'], statistic='median', bins=[35, 35])
    ret_med_BX = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['sin2_theta_bx'], statistic='median', bins=[35, 35])
    ret_cnt   = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['L_eff_m'], statistic='count', bins=[35, 35])

    f_edges, v_edges = ret_med_L.x_edge, ret_med_L.y_edge
    mesh_f, mesh_v = np.meshgrid(f_edges, v_edges)
    X_c, Y_c = np.meshgrid(0.5 * (f_edges[:-1] + f_edges[1:]), 0.5 * (v_edges[:-1] + v_edges[1:]))
    surf_L = ret_med_L.statistic.T
    surf_L[ret_cnt.statistic.T < min_events] = np.nan
    pcm = ax.pcolormesh(mesh_f, mesh_v, surf_L, cmap='plasma', vmin=0, vmax=20, shading='flat')
    plt.colorbar(pcm, ax=ax).set_label(rf'Empirical Median $L_{{eff}}$ (m) [$N \geq {min_events}$]')

    B_grid = np.nan_to_num(ret_med_B.statistic.T, nan=1.0)
    BX_grid = np.nan_to_num(ret_med_BX.statistic.T, nan=1.0)

    contours = ax.contour(X_c, Y_c, analytical_leff_model9_tanh((X_c, Y_c, B_grid, BX_grid), *popt_model9), levels=[2.0, 3.5, 5.0, 7.5, 10.0, 15.0], colors='cyan', linewidths=1.8, linestyles='--')
    labels = ax.clabel(contours, fmt='%.1fm', inline=True, fontsize=9.5, inline_spacing=5)
    plt.setp(labels, path_effects=[path_effects.withStroke(linewidth=2.5, foreground='black')], color='cyan', fontweight='bold')

    ax.set_xlabel("Wave Frequency (Hz)", fontweight='bold')
    ax.set_ylabel(r"Spacecraft Potential $V_{sc}$ (V)", fontweight='bold')
    ax.set_title(r"Data Map with Overlaid Model 9 Contours (Mondal Tanh)", fontweight='bold')
    ax.grid(True, alpha=0.3)


def draw_model9_2d_residuals(ax, df_clean, popt_model9, min_events=5):
    """PANEL: 2D Model 9 Residual Map (Data - Model 9)."""
    if df_clean.empty or popt_model9 is None: return
    valid = df_clean.dropna(subset=['f_weighted_Hz', 'V_sc_volts', 'B_wave_nT', 'L_eff_m']).copy()

    bx_col = next((col for col in ['theta_BX_deg', 'theta_bx_deg'] if col in valid.columns), None)
    valid['sin2_theta_bx'] = np.sin(np.radians(np.nan_to_num(valid[bx_col].values, nan=90.0)))**2 if bx_col else np.ones(len(valid))

    b_col = next((c for c in ['B_high_freq_nT', 'B_wave_nT'] if c in valid.columns), 'B_wave_nT')
    raw_b = valid[b_col].values
    b_perp = raw_b * np.sin(np.radians(np.nan_to_num(valid['theta_kB_deg'].values, nan=90.0))) if 'theta_kB_deg' in valid.columns else raw_b
    valid['B_perp_nT'] = b_perp

    ret_med_L = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['L_eff_m'], statistic='median', bins=[35, 35])
    ret_med_B = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['B_perp_nT'], statistic='median', bins=[35, 35])
    ret_med_BX = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['sin2_theta_bx'], statistic='median', bins=[35, 35])
    ret_cnt   = binned_statistic_2d(valid['f_weighted_Hz'], valid['V_sc_volts'], valid['L_eff_m'], statistic='count', bins=[35, 35])

    f_edges, v_edges = ret_med_L.x_edge, ret_med_L.y_edge
    mesh_f, mesh_v = np.meshgrid(f_edges, v_edges)
    X_c, Y_c = np.meshgrid(0.5 * (f_edges[:-1] + f_edges[1:]), 0.5 * (v_edges[:-1] + v_edges[1:]))

    B_grid = np.nan_to_num(ret_med_B.statistic.T, nan=1.0)
    BX_grid = np.nan_to_num(ret_med_BX.statistic.T, nan=1.0)

    residual_matrix = ret_med_L.statistic.T - analytical_leff_model9_tanh((X_c, Y_c, B_grid, BX_grid), *popt_model9)
    residual_matrix[ret_cnt.statistic.T < min_events] = np.nan

    pcm = ax.pcolormesh(mesh_f, mesh_v, residual_matrix, cmap='coolwarm', vmin=-5.0, vmax=5.0, shading='flat')
    plt.colorbar(pcm, ax=ax).set_label(r'Residual $\Delta L_{eff}$ (Data - Model 9) [m]')
    ax.axhline(popt_model9[3], color='black', linestyle=':', label=rf'$V_0 = {popt_model9[3]:.2f}\mathrm{{V}}$')
    ax.set_xlabel("Wave Frequency (Hz)", fontweight='bold')
    ax.set_ylabel(r"Spacecraft Potential $V_{sc}$ (V)", fontweight='bold')
    ax.set_title(r"Model 9 Residual Map: $L_{\mathrm{data}} - L_{\mathrm{model9}}$", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')


def draw_leff_vs_deltaB_with_model9(ax, df_clean, popt_model9):
    """PANEL: AC Sheath Damping vs. deltaB_perp using V_sc Slices for Model 9."""
    if 'B_wave_nT' not in df_clean.columns or 'V_sc_volts' not in df_clean.columns or popt_model9 is None:
        ax.text(0.5, 0.5, "Missing required columns for Model 9 damping plot.", ha='center', va='center', transform=ax.transAxes)
        return

    valid = df_clean.dropna(subset=['B_wave_nT', 'V_sc_volts', 'f_weighted_Hz', 'L_eff_m']).copy()
    valid = valid[(valid['B_wave_nT'] > 0) & (valid['L_eff_m'] > 0)]

    bx_col = next((col for col in ['theta_BX_deg', 'theta_bx_deg'] if col in valid.columns), None)
    valid['sin2_theta_bx'] = np.sin(np.radians(np.nan_to_num(valid[bx_col].values, nan=90.0)))**2 if bx_col else np.ones(len(valid))

    b_col = next((c for c in ['B_high_freq_nT', 'B_wave_nT'] if c in valid.columns), 'B_wave_nT')
    raw_b = valid[b_col].values
    b_perp = raw_b * np.sin(np.radians(np.nan_to_num(valid['theta_kB_deg'].values, nan=90.0))) if 'theta_kB_deg' in valid.columns else raw_b
    valid['B_perp_nT'] = b_perp

    valid['vsc_slice'] = pd.cut(
        valid['V_sc_volts'], 
        bins=[-np.inf, 1.0, 3.5, 6.5, np.inf], 
        labels=[r'Low $V_{sc} \leq 1.0\mathrm{V}$', r'Mod $1.0 < V_{sc} \leq 3.5\mathrm{V}$', r'Peak $3.5 < V_{sc} \leq 6.5\mathrm{V}$', r'High $V_{sc} > 6.5\mathrm{V}$']
    )
    colors_list = ['purple', 'teal', 'darkorange', 'crimson']
    b_eval = np.logspace(-3, 1.3, 100)

    for (slice_name, group), col in zip(valid.groupby('vsc_slice', observed=False), colors_list):
        if len(group) < 15: continue
        b_bins = np.logspace(np.log10(group['B_perp_nT'].min()), np.log10(group['B_perp_nT'].max()), 10)
        group_copy = group.copy()
        group_copy['b_bin'] = pd.cut(group_copy['B_perp_nT'], b_bins)
        b_centers = 0.5 * (b_bins[:-1] + b_bins[1:])
        
        binned_med = group_copy.groupby('b_bin', observed=False)['L_eff_m'].median()
        binned_q25 = group_copy.groupby('b_bin', observed=False)['L_eff_m'].quantile(0.25)
        binned_q75 = group_copy.groupby('b_bin', observed=False)['L_eff_m'].quantile(0.75)
        cnt_leff   = group_copy.groupby('b_bin', observed=False)['L_eff_m'].count()
        
        med_vals, q25_vals, q75_vals = binned_med.values.copy(), binned_q25.values.copy(), binned_q75.values.copy()
        mask_valid = cnt_leff.values >= 3
        med_vals[~mask_valid], q25_vals[~mask_valid], q75_vals[~mask_valid] = np.nan, np.nan, np.nan
        
        ax.plot(b_centers, med_vals, color=col, linewidth=1.8, marker='o', markersize=4, label=f'{slice_name}')
        ax.fill_between(b_centers, q25_vals, q75_vals, color=col, alpha=0.15)
        
        v_med = group['V_sc_volts'].median()
        f_med = group['f_weighted_Hz'].median()
        bx_med = group['sin2_theta_bx'].median()
        if np.isfinite(v_med) and np.isfinite(f_med):
            l_model = analytical_leff_model9_tanh((f_med, v_med, b_eval, bx_med), *popt_model9)
            ax.plot(b_eval, l_model, color=col, linestyle='--', linewidth=1.8, alpha=0.90)

    L_base_val = popt_model9[0] if popt_model9 is not None else 1.57
    ax.axhline(L_base_val, color='gray', linestyle=':', label=rf'$L_{{base}} = {L_base_val:.2f}\mathrm{{m}}$')
    ax.set_xscale('log')
    ax.set_xlabel(r"Perpendicular Wave Fluctuation $\delta B_\perp$ (nT)", fontweight='bold')
    ax.set_ylabel(r"Effective Antenna Length $L_{\mathrm{eff}}$ (m)", fontweight='bold')
    ax.set_title(r"Model 9: AC Sheath Damping vs. Perpendicular Wave Fluctuation $\delta B_\perp$", fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    
def generate_ascii_param_box(args_dict, title="INPUT PARAMETERS (Hammerhead Diagnostics)"):
    keys = sorted(args_dict.keys())
    max_k_len, max_v_len = max(len(str(k)) for k in keys), max(len(str(args_dict[k])) for k in keys)
    k_width, v_width = max(max_k_len, 23), max(max_v_len, 31)
    total_width = k_width + v_width + 5
    border_mid = f"+{'-' * (k_width + 2)}+{'-' * (v_width + 2)}+"
    lines = [f"+{'-' * (total_width - 2)}+", f"| {title.center(total_width - 4)} |", border_mid]
    for k in keys: lines.append(f"| {k:<{k_width}} | {str(args_dict[k]):<{v_width}} |")
    lines.append(border_mid)
    return "\n".join(lines)

# =========================================================================
# MAIN DIAGNOSTIC GENERATOR
# =========================================================================

def generate_diagnostic_plots(csv_file, gap_limit=120, min_events=15, fidelity_limit=0.0, save_individual=False, debug_col=None):
    print(f"    [*] Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['epoch'], unit='s')
    df = df.sort_values('datetime').reset_index(drop=True)
    date_tag = df['datetime'].dt.strftime('%Y%m%d').iloc[0]
    
    print("    [*] Computing theoretical cold-plasma dispersion k_disp...")
    df['k_disp'] = calculate_cold_plasma_kdisp(df)

    df['f_res_thermal_Hz'] = (df['k_disp'] * df['v_th_para_ms']) / (2 * np.pi)
    delta_omega = df['f_bandwidth_Hz'] * 2 * np.pi
    k_delta_v = df['k_disp'] * df['v_th_para_ms']
    valid_scatter = (delta_omega > 0) & (k_delta_v > 0) & np.isfinite(delta_omega) & np.isfinite(k_delta_v)
    x_clean, y_clean = k_delta_v[valid_scatter], delta_omega[valid_scatter]

    # --- QUALITY & FIDELITY FILTERING ---
    valid_leff = (df['L_eff_m'] > 0) & (df['L_eff_m'] < 40)
    df_clean = df[valid_leff].dropna(subset=['f_weighted_Hz', 'L_eff_m']).copy()

    if fidelity_limit > 0.0:
        valid_kdisp = np.isfinite(df_clean['k_disp']) & (df_clean['k_disp'] > 0)
        df_clean = df_clean[valid_kdisp].copy()
        ratio = (df_clean['f_bandwidth_Hz'] * 2 * np.pi) / (df_clean['k_disp'] * df_clean['v_th_para_ms'])
        in_fidelity_window = (ratio >= (1.0 / fidelity_limit)) & (ratio <= fidelity_limit)
        n_before = len(df_clean)
        df_clean = df_clean[in_fidelity_window].copy()
        print(f"    [*] Fidelity Filter Active (Limit = {fidelity_limit:.2f}, via k_disp): Kept {len(df_clean):,} / {n_before:,} pts")

    # --- Spacecraft Charging Regime Audit ---
    if 'V_sc_volts' in df_clean.columns:
        n_neg_vsc = (df_clean['V_sc_volts'] <= 0).sum()
        pct_neg = (n_neg_vsc / len(df_clean)) * 100.0 if len(df_clean) > 0 else 0.0
        print(f"    [*] Spacecraft Potential Audit: {n_neg_vsc:,} / {len(df_clean):,} pts ({pct_neg:.2f}%) in negative charging regime (V_sc <= 0V)")
        
        n_outliers = (df_clean['V_sc_volts'] < -2.0).sum()
        pct_outliers = (n_outliers / len(df_clean)) * 100.0 if len(df_clean) > 0 else 0.0
        df_clean = df_clean[df_clean['V_sc_volts'] >= -2.0].copy()
        print(f"    [*] Probe Quality Gate: Stripped {n_outliers:,} points ({pct_outliers:.2f}%) with non-physical V_sc < -2.0V")
        
    # =========================================================================
    # EXPORT SLOW-WIND TIMESTAMPS (< 350 km/s) FOR VDF / MOMENT AUDIT
    # =========================================================================
    if 'v_para_sc_ms' in df_clean.columns:
        slow_mask = (df_clean['v_para_sc_ms'] < 350000.0) & np.isfinite(df_clean['v_para_sc_ms'])
        slow_df = df_clean[slow_mask].copy()

        if not slow_df.empty:
            out_slow_txt = csv_file.replace('.csv', '_slow_wind_lt350kms_timestamps.txt')
            
            if 'iso_time' in slow_df.columns:
                timestamps = slow_df['iso_time'].dropna().tolist()
            else:
                timestamps = pd.to_datetime(slow_df['epoch'], unit='s', utc=True).dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ').tolist()

            with open(out_slow_txt, 'w') as f_out:
                for ts in timestamps:
                    f_out.write(f"{ts}\n")

            print(f"    [*] Audit Export: Wrote {len(timestamps):,} slow-wind timestamps (< 350 km/s) to:")
            print(f"        -> {out_slow_txt}")
        else:
            print(f"    [*] Audit Notice: No points found with v_para_sc_ms < 350 km/s.")
    
    df_lit = load_literature_event_context(csv_file)
    
    popt_2d, r2_2d = fit_leff_surface_noVsc(df_clean)
    popt_vsc, r2_vsc = fit_leff_surface_f_vsc(df_clean)
    popt_3d, r2_3d = fit_leff_surface_3d(df_clean)
    popt_bandpass, r2_bandpass = fit_leff_surface_physics_bandpass(df_clean)
    popt_wavepower, r2_wavepower = fit_leff_surface_physics_wavepower(df_clean)
    popt_model6, r2_model6 = fit_leff_surface_model6(df_clean)
    popt_model7, r2_model7 = fit_leff_surface_model7(df_clean)
    popt_model8, r2_model8 = fit_leff_surface_model8(df_clean)
    popt_model9, r2_model9 = fit_leff_surface_model9(df_clean)

    # --- PRINT MASTER MODEL PERFORMANCE COMPARISON SUMMARY TABLE ---
    headers = ("Model Name", "Evaluated Pts", "R-squared (R²)")
    pts_str = f"{len(df_clean):,}"

    # Build active row data
    rows = []
    if r2_2d is not None:        rows.append(("1. 2D Debye Surface L(f, λ_D)", pts_str, f"{r2_2d:.4f}"))
    if r2_vsc is not None:       rows.append(("2. Streamlined L(f, V_sc)", pts_str, f"{r2_vsc:.4f}"))
    if r2_3d is not None:        rows.append(("3. 3D Surface L(f, λ_D, V_sc)", pts_str, f"{r2_3d:.4f}"))
    if r2_bandpass is not None:  rows.append(("4. Physics Bandpass L(f, V_sc)", pts_str, f"{r2_bandpass:.4f}"))
    if r2_wavepower is not None: rows.append(("5. Model 5 (Bandpass + dB Damping)", pts_str, f"{r2_wavepower:.4f}"))
    if r2_model6 is not None:    rows.append(("6. Model 6 (Power-Law Roll-Off)", pts_str, f"{r2_model6:.4f}"))
    if r2_model7 is not None:    rows.append(("7. Model 7 (Bandpass + dV_raw Damping)", pts_str, f"{r2_model7:.4f}"))
    if r2_model8 is not None:    rows.append(("8. Model 8 (1/r^2 EUV flux and asymmetric wake)", pts_str, f"{r2_model8:.4f}"))
    if r2_model9 is not None:    rows.append(("9. Model 9 (Model 8 with hyperbolic tanh(f))", pts_str, f"{r2_model8:.4f}"))

    if rows:
        # Calculate dynamic column widths based on longest string in each column
        col1_w = max(len(headers[0]), max(len(r[0]) for r in rows))
        col2_w = max(len(headers[1]), max(len(r[1]) for r in rows))
        col3_w = max(len(headers[2]), max(len(r[2]) for r in rows))

        # Calculate border width (" | " adds 6 characters total)
        total_w = col1_w + col2_w + col3_w + 6

        # Render Table
        print("\n" + "=" * total_w)
        print("EXECUTIVE MODEL PERFORMANCE COMPARISON SUMMARY".center(total_w))
        print("=" * total_w)
        print(f"{headers[0]:<{col1_w}} | {headers[1]:<{col2_w}} | {headers[2]:<{col3_w}}")
        print("-" * total_w)
        for name, pts, r2 in rows:
            print(f"{name:<{col1_w}} | {pts:<{col2_w}} | {r2:<{col3_w}}")
        print("=" * total_w + "\n")

    # --- EXECUTIVE DASHBOARDS ---
    # Model 5 diagnostic plot
    fig_m5, axes_m5 = plt.subplots(2, 3, figsize=(18, 11))
    fig_m5.suptitle("PSP FIELDS Antenna Effective Length Calibration Dashboard (Model 5)", fontweight='bold', fontsize=16, y=0.98)
    draw_correlation(axes_m5[0, 0], x_clean, y_clean, fidelity_limit=fidelity_limit)
    draw_model5_surface_and_overlay(axes_m5[0, 1], df_clean, popt_wavepower, min_events=min_events)
    draw_model5_2d_residuals(axes_m5[0, 2], df_clean, popt_wavepower, min_events=min_events)
    draw_dispersion_validation(axes_m5[1, 0], df_clean, debug_col=debug_col)
    draw_leff_vs_deltaB_with_model5(axes_m5[1, 1], df_clean, popt_wavepower)
    draw_observed_vs_predicted_parity(axes_m5[1, 2], df_clean, popt_wavepower, r2_wavepower=r2_wavepower)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_plot_m5 = csv_file.replace('.csv', '_diagnostics_model5.png') if re.search(r'\d{8}', os.path.basename(csv_file)) else os.path.join(os.path.dirname(csv_file), f"{os.path.basename(csv_file).replace('.csv', '')}_{date_tag}_diagnostics.png")
    plt.savefig(out_plot_m5, dpi=300, bbox_inches='tight')
    plt.close(fig_m5)

    # Model 6 diagnostic plot
    fig_m6, axes_m6 = plt.subplots(2, 3, figsize=(18, 11))
    fig_m6.suptitle("PSP FIELDS Antenna Effective Length Calibration Dashboard (Model 6)", fontweight='bold', fontsize=16, y=0.98)
    draw_correlation(axes_m6[0, 0], x_clean, y_clean, fidelity_limit=fidelity_limit)
    draw_model6_surface_and_overlay(axes_m6[0, 1], df_clean, popt_model6, min_events=min_events)
    draw_model6_2d_residuals(axes_m6[0, 2], df_clean, popt_model6, min_events=min_events)
    draw_dispersion_validation(axes_m6[1, 0], df_clean, debug_col=debug_col)
    draw_model6_parity(axes_m6[1, 1], df_clean, popt_model6, r2_model6=r2_model6)
    draw_literature_comparison(axes_m6[1, 2], df_clean, popt_model6, popt_wavepower=popt_wavepower, popt_model7=popt_model7, popt_model8=popt_model8, df_lit=df_lit)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_plot_m6 = csv_file.replace('.csv', '_diagnostics_model6.png')
    plt.savefig(out_plot_m6, dpi=300, bbox_inches='tight')
    plt.close(fig_m6)

    # Model 7 diagnostic plot
    fig_m7, axes_m7 = plt.subplots(2, 3, figsize=(18, 11))
    fig_m7.suptitle("PSP FIELDS Antenna Effective Length Calibration Dashboard (Model 7)", fontweight='bold', fontsize=16, y=0.98)
    draw_correlation(axes_m7[0, 0], x_clean, y_clean, fidelity_limit=fidelity_limit)
    draw_model7_surface_and_overlay(axes_m7[0, 1], df_clean, popt_model7, min_events=min_events)
    draw_model7_2d_residuals(axes_m7[0, 2], df_clean, popt_model7, min_events=min_events)
    draw_dispersion_validation(axes_m7[1, 0], df_clean, debug_col=debug_col)
    draw_model7_parity(axes_m7[1, 1], df_clean, popt_model7, r2_model7=r2_model7)
    draw_literature_comparison(axes_m7[1, 2], df_clean, popt_model6, popt_wavepower=popt_wavepower, popt_model7=popt_model7, popt_model8=popt_model8, df_lit=df_lit)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_plot_m7 = csv_file.replace('.csv', '_diagnostics_model7.png')
    plt.savefig(out_plot_m7, dpi=300, bbox_inches='tight')
    plt.close(fig_m7)
    
    # Model 8 diagnostic plot
    fig_m8, axes_m8 = plt.subplots(2, 3, figsize=(18, 11))
    fig_m8.suptitle("PSP FIELDS Antenna Effective Length Calibration Dashboard (Model 8 Canonical)", fontweight='bold', fontsize=16, y=0.98)
    draw_correlation(axes_m8[0, 0], x_clean, y_clean, fidelity_limit=fidelity_limit)
    draw_model8_surface_and_overlay(axes_m8[0, 1], df_clean, popt_model8, min_events=min_events)
    draw_model8_2d_residuals(axes_m8[0, 2], df_clean, popt_model8, min_events=min_events)
    draw_dispersion_validation(axes_m8[1, 0], df_clean, debug_col=debug_col)
    draw_model8_parity(axes_m8[1, 1], df_clean, popt_model8, r2_model8=r2_model8)
    draw_literature_comparison(axes_m8[1, 2], df_clean, popt_model6, popt_wavepower=popt_wavepower, popt_model7=popt_model7, popt_model8=popt_model8, df_lit=df_lit)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_plot_m8 = csv_file.replace('.csv', '_diagnostics_model8.png')
    plt.savefig(out_plot_m8, dpi=300, bbox_inches='tight')
    plt.close(fig_m8)
    
    # Model 9 diagnostic plot
    fig_m9, axes_m9 = plt.subplots(2, 3, figsize=(18, 11))
    fig_m9.suptitle("PSP FIELDS Antenna Effective Length Calibration Dashboard (Model 9 Mondal Tanh)", fontweight='bold', fontsize=16, y=0.98)
    draw_correlation(axes_m9[0, 0], x_clean, y_clean, fidelity_limit=fidelity_limit)
    draw_model9_surface_and_overlay(axes_m9[0, 1], df_clean, popt_model9, min_events=min_events)
    draw_model9_2d_residuals(axes_m9[0, 2], df_clean, popt_model9, min_events=min_events)
    draw_dispersion_validation(axes_m9[1, 0], df_clean, debug_col=debug_col)
    draw_model9_parity(axes_m9[1, 1], df_clean, popt_model9, r2_model9=r2_model9)
    draw_literature_comparison(axes_m9[1, 2], df_clean, popt_model6, popt_wavepower=popt_wavepower, popt_model7=popt_model7, popt_model8=popt_model8, df_lit=df_lit)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_plot_m9 = csv_file.replace('.csv', '_diagnostics_model9.png')
    plt.savefig(out_plot_m9, dpi=300, bbox_inches='tight')
    plt.close(fig_m9)

    if save_individual:
        slides_dir = os.path.join(os.path.dirname(csv_file), f"{os.path.basename(csv_file).replace('.csv', '')}_slide_plots")
        os.makedirs(slides_dir, exist_ok=True)
        print(f"    [*] Exporting individual slide panels to: {slides_dir}/")

        panels = [
            ("01_frequencies.png", lambda ax: draw_frequencies(ax, df)),
            ("02_kinematics.png", lambda ax: draw_kinematics(ax, df)),
            ("03_k_parallel.png", lambda ax: draw_k_parallel(ax, df)),
            ("04_resonance_loglog.png", lambda ax: draw_correlation(ax, x_clean, y_clean, fidelity_limit=fidelity_limit)),
            ("05_debye_surface_map.png", lambda ax: draw_surface_map(ax, df_clean, min_events=min_events)),
            ("06_debye_slices.png", lambda ax: draw_debye_slices(ax, df_clean, popt_bandpass=popt_bandpass, min_events=min_events)),
            ("07_debye_surface_3d.png", lambda ax: draw_surface_map_3d(ax, df_clean, min_events=min_events)),
            ("08_vsc_surface_map.png", lambda ax: draw_vsc_surface_map(ax, df_clean, min_events=min_events)),
            ("09_vsc_slices.png", lambda ax: draw_vsc_slices(ax, df_clean, popt_bandpass=popt_bandpass, min_events=min_events)),
            ("10_vsc_surface_map_3d.png", lambda ax: draw_vsc_surface_map_3d(ax, df_clean, min_events=min_events)),
            ("11_vsc_fit_map_2d.png", lambda ax: draw_vsc_analytical_surface_2d(ax, df_clean, popt_vsc=popt_vsc)),
            ("12_vsc_fit_map_3d.png", lambda ax: draw_vsc_analytical_surface_3d(ax, df_clean, popt_vsc=popt_vsc)),
            ("13_Leff_wavePower_controlled.png", lambda ax: draw_leff_vs_deltaB_controlled(ax, df_clean)),
            ("14_bandpass_fit_map_2d.png", lambda ax: draw_bandpass_analytical_surface_2d(ax, df_clean, popt_bandpass=popt_bandpass)),
            ("15_bandpass_fit_map_3d.png", lambda ax: draw_bandpass_analytical_surface_3d(ax, df_clean, popt_bandpass=popt_bandpass)),
            ("16_vsc_vs_density_proof.png", lambda ax: draw_vsc_vs_density(ax, df_clean, popt_bandpass=popt_bandpass)),
            ("17_vsc_vs_debye_proof.png", lambda ax: draw_vsc_vs_debye(ax, df_clean, popt_bandpass=popt_bandpass)),
            ("18_instrument_rc_bandwidth.png", lambda ax: draw_instrument_rc_bandwidth(ax, df_clean)),
            ("19_data_model_contours_overlay.png", lambda ax: draw_overlay_data_and_model_contours(ax, df_clean, popt_bandpass, min_events=5)),
            ("20_2d_model_residuals.png", lambda ax: draw_2d_model_residuals(ax, df_clean, popt_bandpass, min_events=5)),
            ("21_bfieldModel_parity.png", lambda ax: draw_observed_vs_predicted_parity(ax, df_clean, popt_wavepower, r2_wavepower=r2_wavepower)),
            ("22_Leff_vs_deltaB_model.png", lambda ax: draw_leff_vs_deltaB_with_model5(ax, df_clean, popt_wavepower)),
            ("23_model_evolution.png", None),
            ("24_BfieldModel_surface_overlay.png", lambda ax: draw_model5_surface_and_overlay(ax, df_clean, popt_wavepower, min_events=5)),
            ("25_BfieldModel_residuals.png", lambda ax: draw_model5_2d_residuals(ax, df_clean, popt_wavepower, min_events=5)), 
            ("26_dispersion_validation.png", lambda ax: draw_dispersion_validation(ax, df_clean, debug_col=debug_col)),
            ("27_antenna_bias_current_residual.png", lambda ax: draw_bias_current_residual_check(ax, df_clean, popt_wavepower)),
            ("28_literature_comparison.png", lambda ax: draw_literature_comparison(ax, df_clean, popt_model6, popt_wavepower=popt_wavepower, popt_model7=popt_model7, popt_model8=popt_model8, df_lit=df_lit)),
            ("29_model6_parity.png", lambda ax: draw_model6_parity(ax, df_clean, popt_model6, r2_model6=r2_model6)),
            ("30_model6_surface_overlay.png", lambda ax: draw_model6_surface_and_overlay(ax, df_clean, popt_model6, min_events=min_events)),
            ("31_model6_residuals.png", lambda ax: draw_model6_2d_residuals(ax, df_clean, popt_model6, min_events=min_events)),
            ("32_Leff_vs_deltaB_model6.png", lambda ax: draw_leff_vs_deltaB_with_model6(ax, df_clean, popt_model6)),
            ("33_model7_parity.png", lambda ax: draw_model7_parity(ax, df_clean, popt_model7, r2_model7=r2_model7)),
            ("34_model7_surface_overlay.png", lambda ax: draw_model7_surface_and_overlay(ax, df_clean, popt_model7, min_events=min_events)),
            ("35_model7_residuals.png", lambda ax: draw_model7_2d_residuals(ax, df_clean, popt_model7, min_events=min_events)),
            ("36_Leff_vs_Vwave_model7.png", lambda ax: draw_leff_vs_Vwave_with_model7(ax, df_clean, popt_model7)),
            ("37_model8_parity.png", lambda ax: draw_model8_parity(ax, df_clean, popt_model8, r2_model8=r2_model8)),
            ("38_model8_surface_overlay.png", lambda ax: draw_model8_surface_and_overlay(ax, df_clean, popt_model8, min_events=min_events)),
            ("39_model8_residuals.png", lambda ax: draw_model8_2d_residuals(ax, df_clean, popt_model8, min_events=min_events)),
            ("40_Leff_vs_deltaB_model8.png", lambda ax: draw_leff_vs_deltaB_with_model8(ax, df_clean, popt_model8)),
            ("41_model9_parity.png", lambda ax: draw_model9_parity(ax, df_clean, popt_model9, r2_model9=r2_model9)),
            ("42_model9_surface_overlay.png", lambda ax: draw_model9_surface_and_overlay(ax, df_clean, popt_model9, min_events=min_events)),
            ("43_model9_residuals.png", lambda ax: draw_model9_2d_residuals(ax, df_clean, popt_model9, min_events=min_events)),
            ("44_Leff_vs_deltaB_model9.png", lambda ax: draw_leff_vs_deltaB_with_model9(ax, df_clean, popt_model9)),
        ]

        for filename, draw_fn in panels:
            out_file_path = os.path.join(slides_dir, filename)
            if filename == "23_model_evolution.png":
                draw_model_evolution_comparison_slide(df_clean, popt_vsc, popt_bandpass, popt_model7, r2_vsc, r2_bandpass, r2_model7, out_file_path)
                continue
            fig_s = plt.figure(figsize=(10, 8)) if "_3d" in filename else plt.subplots(figsize=(8, 5))[0]
            ax_s = fig_s.add_subplot(111, projection='3d') if "_3d" in filename else fig_s.axes[0]
            draw_fn(ax_s)
            if any(k in filename for k in ["01_", "02_", "03_"]):
                apply_adaptive_date_ticks(ax_s, df['epoch'])
                ax_s.set_xlabel("Time (UTC)", fontweight='bold', labelpad=8)
            fig_s.savefig(out_file_path, dpi=300, bbox_inches='tight')
            plt.close(fig_s)
        print(f"    [+] Successfully exported {len(panels)} individual slide figures!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot Hammerhead Diagnostics")
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--gap_limit', type=int, default=20)
    parser.add_argument('--min_events', type=int, default=15)
    parser.add_argument('--fidelity_limit', type=float, default=2.0)
    parser.add_argument('--slides', action='store_true')
    parser.add_argument('--debug_col', type=str, default=None, 
                        help="Variable column to color-code dispersion validation scatter (e.g., 'v_para_sc_ms', 'theta_kB_deg', 'ne_lfr', 'r_au')")
    args = parser.parse_args()
    
    csv_target = args.csv if args.csv else max(glob.glob(os.path.join(config.get_drive_path(), "Research/PSP/Hammerheads/resonanceFiles/hammerhead_cal_*.csv")), key=os.path.getctime)
    summary_txt_path = csv_target.replace('.csv', '_summary.txt')
    
    logger = DualLogger(summary_txt_path)
    sys.stdout = logger
    print("\n" + generate_ascii_param_box(vars(args)) + "\n")
    
    generate_diagnostic_plots(csv_target, gap_limit=args.gap_limit, min_events=args.min_events, fidelity_limit=args.fidelity_limit, save_individual=args.slides, debug_col=args.debug_col)
    sys.stdout = logger.terminal
    logger.close()
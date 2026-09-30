import numpy as np
import scipy.constants as const
import scipy.ndimage as ndimage
import warnings

def get_k_magnitude_from_powers(f_sc, Power_E, Power_B, L_baseline, L_eff_dynamic):
    """Derives physical wavenumber magnitude |k| (rad/m) and Phase Velocity (km/s)."""
    Power_E_clean = np.where(Power_E > 0, Power_E, np.nan)
    Power_B_clean = np.where(Power_B > 0, Power_B, np.nan)
    
    V_amp = (np.sqrt(Power_E_clean) * 1e-3) * L_baseline
    E_amp_Vm = V_amp / L_eff_dynamic
    B_amp_T  = np.sqrt(Power_B_clean) * 1e-9
    
    with np.errstate(divide='ignore', invalid='ignore'):
        V_ph = E_amp_Vm / B_amp_T
        k_mag = (2 * np.pi * f_sc) / V_ph
        
    return k_mag, V_ph / 1000.0

def calculate_plasma_frame_physics(f_sc, k_hat_n, k_hat_p, k_hat_q, k_mag, V_sw_RTN, V_sc_RTN, B0_SC, cmat_SC_to_RTN):
    k_n = k_mag * k_hat_n
    k_p = k_mag * k_hat_p
    k_q = k_mag * k_hat_q
    
    B0_mag = np.linalg.norm(B0_SC, axis=-1, keepdims=True)
    N_hat_SC = B0_SC / B0_mag
    
    sc_y = np.array([0, -1, 0])
    P_temp = np.cross(N_hat_SC, sc_y)
    P_hat_SC = P_temp / np.linalg.norm(P_temp, axis=-1, keepdims=True)
    Q_hat_SC = np.cross(N_hat_SC, P_hat_SC)
    
    k_vec_SC = (k_n[:, :, np.newaxis] * N_hat_SC[:, np.newaxis, :]) + \
               (k_p[:, :, np.newaxis] * P_hat_SC[:, np.newaxis, :]) + \
               (k_q[:, :, np.newaxis] * Q_hat_SC[:, np.newaxis, :])
               
    k_vec_RTN = np.einsum('tij,tfj->tfi', cmat_SC_to_RTN, k_vec_SC)
    
    V_rel_RTN = (V_sw_RTN - V_sc_RTN) * 1000.0
    k_dot_V = np.sum(k_vec_RTN * V_rel_RTN[:, np.newaxis, :], axis=-1)
              
    f_plasma = f_sc - (k_dot_V / (2 * np.pi))
    k_parallel = k_n
    k_perp = np.sqrt(k_p**2 + k_q**2)
    
    return f_plasma, k_parallel, k_perp

def calculate_alfven_velocity(B0_mag, N_p, N_alpha, nominal_alpha_ratio=0.04):
    B_tesla = B0_mag * 1e-9
    N_a_clean = np.where(np.isfinite(N_alpha), N_alpha, nominal_alpha_ratio * N_p)
    
    rho_p = (N_p * 1e6) * const.m_p
    rho_a = (N_a_clean * 1e6) * (4 * const.m_p) 
    rho_total = rho_p + rho_a
    
    with np.errstate(divide='ignore', invalid='ignore'):
        v_A = B_tesla / np.sqrt(const.mu_0 * rho_total)
        
    return v_A / 1000.0

def calculate_integrated_powers(f_sc, Wave_Power, S_n, coh, ell, k_parallel, k_mag):
    f_1d = f_sc[0, :] if f_sc.ndim == 2 else f_sc
    df = np.mean(np.diff(f_1d))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = np.clip(np.abs(k_parallel) / k_mag, -1.0, 1.0)
        theta_k = np.degrees(np.arccos(cos_theta))
        
    def smooth2d(arr, size=(2, 20)):
        valid_mask = np.isfinite(arr).astype(float)
        arr_filled = np.where(np.isfinite(arr), arr, 0.0)
        s_arr = ndimage.uniform_filter(arr_filled, size=size, mode='nearest')
        s_weights = ndimage.uniform_filter(valid_mask, size=size, mode='nearest')
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(s_weights > 0, s_arr / s_weights, np.nan)

    coh_s = smooth2d(coh)
    ell_s = smooth2d(ell)
    theta_k_s = smooth2d(theta_k)
    
    base_mask = (coh_s >= 0.8) & (theta_k_s <= 25.0)
    mask_LH = base_mask & (ell_s <= -0.8)
    mask_RH = base_mask & (ell_s >= 0.8)
    mask_Sn_pos = base_mask & (S_n > 0)
    mask_Sn_neg = base_mask & (S_n < 0)
    
    int_LH = np.nansum(np.where(mask_LH, Wave_Power, 0.0), axis=1) * df
    int_RH = np.nansum(np.where(mask_RH, Wave_Power, 0.0), axis=1) * df
    int_Sn_pos = np.nansum(np.where(mask_Sn_pos, S_n, 0.0), axis=1) * df
    int_Sn_neg = np.nansum(np.where(mask_Sn_neg, S_n, 0.0), axis=1) * df
    
    return int_LH, int_RH, int_Sn_pos, int_Sn_neg, mask_LH, mask_RH, mask_Sn_pos, mask_Sn_neg
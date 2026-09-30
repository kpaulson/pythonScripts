import numpy as np
import scipy.constants as const

def calculate_resonant_energies(f_plasma, k_parallel, B0_mag):
    q_e = const.e
    m_p = const.m_p
    m_alpha = 4 * const.m_p
    
    B0_tesla = B0_mag * 1e-9 
    Omega_p = (q_e * B0_tesla) / m_p
    Omega_alpha = (2 * q_e * B0_tesla) / m_alpha
    
    Omega_p_grid = Omega_p[:, np.newaxis]
    Omega_alpha_grid = Omega_alpha[:, np.newaxis]
    omega_plasma = 2 * np.pi * f_plasma
    
    with np.errstate(divide='ignore', invalid='ignore'):
        v_res_p_n1 = (omega_plasma - Omega_p_grid) / k_parallel
        v_res_alpha_n1 = (omega_plasma - Omega_alpha_grid) / k_parallel
        E_res_p_n1 = (0.5 * m_p * v_res_p_n1**2) / q_e
        E_res_alpha_n1 = (0.5 * m_alpha * v_res_alpha_n1**2) / q_e
        
        v_res_landau = omega_plasma / k_parallel
        E_res_landau_p = (0.5 * m_p * v_res_landau**2) / q_e
        
        v_res_p_n_minus1 = (omega_plasma + Omega_p_grid) / k_parallel
        E_res_p_n_minus1 = (0.5 * m_p * v_res_p_n_minus1**2) / q_e
        
        v_res_alpha_n_minus1 = (omega_plasma + Omega_alpha_grid) / k_parallel
        E_res_alpha_n_minus1 = (0.5 * m_alpha * v_res_alpha_n_minus1**2) / q_e
        
    return E_res_p_n1, E_res_alpha_n1, E_res_landau_p, E_res_p_n_minus1, E_res_alpha_n_minus1
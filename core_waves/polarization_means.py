import numpy as np
import scipy.ndimage as ndimage
import warnings

def smooth1_1d(arr, size):
    """
    Replicates Autoplot's smooth1() boxcar moving average across the frequency axis.
    Size defines the number of frequency bins to average over.
    Handles NaNs smoothly by weighting against valid data points.
    """
    valid_mask = np.isfinite(arr).astype(float)
    arr_filled = np.where(np.isfinite(arr), arr, 0.0)
    
    # Apply uniform filter across the frequency axis (axis=1)
    s_arr = ndimage.uniform_filter1d(arr_filled, size=size, axis=1, mode='nearest')
    s_weights = ndimage.uniform_filter1d(valid_mask, size=size, axis=1, mode='nearest')
    
    with np.errstate(divide='ignore', invalid='ignore'):
        smoothed = np.where(s_weights > 0, s_arr / s_weights, np.nan)
        
    return smoothed

def calculate_polarization(Bn_fft, Bp_fft, Bq_fft, Nx, Ny, Nz, wave_power_min, En_fft=None, Ep_fft=None, Eq_fft=None):
    """
    Polarization analysis performed on spectral components based on [Means 1972].
    
    Parameters:
        B(n,p,q)_fft: rank3 complex numpy arrays (Time, Freq, Component)
        N(x,y,z): rank1 real numpy arrays (Time)
        wave_power_min: float
        E(n,p,q)_fft: rank3 complex numpy arrays (Time, Freq, Component) [Optional]
        
    Returns:
        dict containing physical parameters mapped identically to Jython outputs.
    """
    
    results = {}
    
    # --- 1. DEFINE J-PRIME (INPUT) MATRIX ELEMENTS ---
    
    # In Python, we can just use the absolute square for the real diagonal
    Jxx_prime = np.abs(Bp_fft)**2
    Jyy_prime = np.abs(Bq_fft)**2
    Jzz_prime = np.abs(Bn_fft)**2

    # Cross terms: Jxy = Bp * conj(Bq)
    Jxy_prime = Bp_fft * np.conj(Bq_fft)
    Jxy_prime_real = np.real(Jxy_prime)
    Jxy_prime_img  = np.imag(Jxy_prime)

    Jxz_prime = Bp_fft * np.conj(Bn_fft)
    Jxz_prime_real = np.real(Jxz_prime)
    Jxz_prime_img  = np.imag(Jxz_prime)

    Jyz_prime = Bn_fft * np.conj(Bq_fft)
    Jyz_prime_real = np.real(Jyz_prime)
    Jyz_prime_img  = np.imag(Jyz_prime)
    
    if En_fft is not None:
        E_Jxx_prime = np.abs(Ep_fft)**2
        E_Jyy_prime = np.abs(Eq_fft)**2
        E_Jzz_prime = np.abs(En_fft)**2
        
        E_Jxy_prime = Ep_fft * np.conj(Eq_fft)
        E_Jxy_prime_real = np.real(E_Jxy_prime)
        E_Jxy_prime_img  = np.imag(E_Jxy_prime)
        
        E_Jxz_prime = Ep_fft * np.conj(En_fft)
        E_Jxz_prime_real = np.real(E_Jxz_prime)
        E_Jxz_prime_img  = np.imag(E_Jxz_prime)
        
        E_Jyz_prime = En_fft * np.conj(Eq_fft)
        E_Jyz_prime_real = np.real(E_Jyz_prime)
        E_Jyz_prime_img  = np.imag(E_Jyz_prime)

    # --- 2. DEFINE K-HAT VECTOR (MEANS - IMAGINARY COMPONENTS) ---
    
    img_mag = np.sqrt(Jxy_prime_img**2 + Jxz_prime_img**2 + Jyz_prime_img**2)
    trace = Jxx_prime + Jyy_prime + Jzz_prime
    
    kp = np.zeros_like(Jxx_prime)
    kq = np.zeros_like(Jxx_prime)
    kn = np.zeros_like(Jxx_prime)
    
    # Masking logic
    r = np.abs(img_mag) < 1E-5
    r0 = ~r
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Linear polarizations
        kp[r] = np.sqrt(Jxx_prime[r] / trace[r])
        kq[r] = Jxy_prime_real[r] / (trace[r] * Jxx_prime[r])
        kn[r] = Jxz_prime_real[r] / (trace[r] * Jxx_prime[r])
        
        # Elliptical / Circular polarizations
        kp[r0] =  Jyz_prime_img[r0] / img_mag[r0]
        kq[r0] = -Jxz_prime_img[r0] / img_mag[r0]
        kn[r0] =  Jxy_prime_img[r0] / img_mag[r0]

    # Expand 1D N-vectors to 2D matching the FFT grid
    Nx_2d = np.zeros_like(kp)  # Actually Np=0
    Ny_2d = np.zeros_like(kp)  # Actually Nq=0
    Nz_2d = np.ones_like(kp)   # Actually Nn=1
    
    # Ensure k vector direction aligns with N vector
    r_flip = kn < 0
    kp[r_flip] = -kp[r_flip]
    kq[r_flip] = -kq[r_flip]
    kn[r_flip] = -kn[r_flip]
    
    results['kp'] = kp
    results['kq'] = kq
    results['kn'] = kn

    # --- 3. DEFINE WAVE-NORMAL COORDINATES & ROTATE J-PRIME ---
    
    theta = np.arccos(np.clip(kn, -1.0, 1.0))
    factor = np.sin(theta)
    
    # Force theta <= PI/2
    theta[theta > (np.pi / 2)] = np.pi - theta[theta > (np.pi / 2)]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        Rx = ((kq * Nz_2d) - (kn * Ny_2d)) / factor
        Ry = ((kn * Nx_2d) - (kp * Nz_2d)) / factor
        Rz = ((kp * Ny_2d) - (kq * Nx_2d)) / factor
        
        Sx = (kq * Rz) - (kn * Ry)
        Sy = (kn * Rx) - (kp * Rz)
        Sz = (kp * Ry) - (kq * Rx)

    # Matrix Rotation
    Jxx = (Jxx_prime * Rx**2) + (Jyy_prime * Ry**2) + (Jzz_prime * Rz**2) + \
          2 * (Jxy_prime_real*Rx*Ry + Jxz_prime_real*Rx*Rz + Jyz_prime_real*Ry*Rz)
          
    Jyy = (Jxx_prime * Sx**2) + (Jyy_prime * Sy**2) + (Jzz_prime * Sz**2) + \
          2 * (Jxy_prime_real*Sx*Sy + Jxz_prime_real*Sx*Sz + Jyz_prime_real*Sy*Sz)
          
    Jzz = (Jxx_prime * kp**2) + (Jyy_prime * kq**2) + (Jzz_prime * kn**2) + \
          2 * (Jxy_prime_real*kp*kq + Jxz_prime_real*kp*kn + Jyz_prime_real*kq*kn)

    Jxy_real = (Jxx_prime * Rx * Sx) + (Jyy_prime * Ry * Sy) + (Jzz_prime * Rz * Sz) \
             + (Jxy_prime_real * (Rx * Sy + Ry * Sx)) \
             + (Jxz_prime_real * (Rx * Sz + Rz * Sx)) \
             + (Jyz_prime_real * (Ry * Sz + Rz * Sy))

    Jxy_img  = (Jxy_prime_img * (Rx * Sy - Ry * Sx)) \
             + (Jxz_prime_img * (Rx * Sz - Rz * Sx)) \
             + (Jyz_prime_img * (Ry * Sz - Rz * Sy))
             
    Wave_Power = Jxx + Jyy
    
    if En_fft is not None:
        E_Jxx = (E_Jxx_prime * Rx**2) + (E_Jyy_prime * Ry**2) + (E_Jzz_prime * Rz**2) + \
                2 * (E_Jxy_prime_real*Rx*Ry + E_Jxz_prime_real*Rx*Rz + E_Jyz_prime_real*Ry*Rz)
                
        E_Jyy = (E_Jxx_prime * Sx**2) + (E_Jyy_prime * Sy**2) + (E_Jzz_prime * Sz**2) + \
                2 * (E_Jxy_prime_real*Sx*Sy + E_Jxz_prime_real*Sx*Sz + E_Jyz_prime_real*Sy*Sz)
                
        E_Jxy_real = (E_Jxx_prime * Rx * Sx) + (E_Jyy_prime * Ry * Sy) + (E_Jzz_prime * Rz * Sz) \
                   + (E_Jxy_prime_real * (Rx * Sy + Ry * Sx)) \
                   + (E_Jxz_prime_real * (Rx * Sz + Rz * Sx)) \
                   + (E_Jyz_prime_real * (Ry * Sz + Rz * Sy))

        E_Jxy_img  = (E_Jxy_prime_img * (Rx * Sy - Ry * Sx)) \
                   + (E_Jxz_prime_img * (Rx * Sz - Rz * Sx)) \
                   + (E_Jyz_prime_img * (Ry * Sz - Rz * Sy))

    # --- 4. FREQUENCY SMOOTHING ---
    
    freq_avg = 3 if Jxx.shape[1] < 5 else 5
    
    Jxx      = smooth1_1d(Jxx, freq_avg)
    Jyy      = smooth1_1d(Jyy, freq_avg)
    Jxy_real = smooth1_1d(Jxy_real, freq_avg)
    Jxy_img  = smooth1_1d(Jxy_img, freq_avg)
    
    J_det = (Jxx * Jyy) - (Jxy_real**2 + Jxy_img**2)
    
    if En_fft is not None:
        E_Jxx      = smooth1_1d(E_Jxx, freq_avg)
        E_Jyy      = smooth1_1d(E_Jyy, freq_avg)
        E_Jxy_real = smooth1_1d(E_Jxy_real, freq_avg)
        E_Jxy_img  = smooth1_1d(E_Jxy_img, freq_avg)
        E_J_det = (E_Jxx * E_Jyy) - (E_Jxy_real**2 + E_Jxy_img**2)

    # --- 5. POWER & POLARIZATION PARAMETERS ---
    
    results['Power_compressional'] = Jzz_prime
    results['Power_perp'] = np.sqrt(Jxx_prime**2 + Jyy_prime**2)
    results['Wave_Power'] = Wave_Power
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results['Deg_Polarization'] = np.sqrt(np.clip(1 - ((4 * J_det) / ((Jxx + Jyy)**2)), 0, None))
        results['Coherency'] = np.sqrt(np.clip((Jxy_real**2 + Jxy_img**2) / (Jxx * Jyy), 0.0, 1.0))
        results['Angle_Polarization'] = np.degrees(0.5 * np.arctan2(2 * Jxy_real, Jxx - Jyy))
        
        Handedness = (2 * Jxy_img) / np.sqrt((Jxx + Jyy)**2 - (4 * J_det))
        results['Handedness'] = Handedness
        results['Ellipticity'] = np.tan(0.5 * np.arcsin(np.clip(Handedness, -1.0, 1.0)))
        
    results['Angle_Normal'] = np.degrees(theta)
    
    if En_fft is not None:
        results['E_Power_compressional'] = E_Jzz_prime
        results['E_Power_perp'] = np.sqrt(E_Jxx_prime**2 + E_Jyy_prime**2)
        results['E_Wave_Power'] = E_Jxx + E_Jyy
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results['E_Deg_Polarization'] = np.sqrt(np.clip(1 - ((4 * E_J_det) / ((E_Jxx + E_Jyy)**2)), 0, None))
            results['E_Coherency'] = np.sqrt(np.clip((E_Jxy_real**2 + E_Jxy_img**2) / (E_Jxx * E_Jyy), 0.0, 1.0))
            results['E_Angle_Polarization'] = np.degrees(0.5 * np.arctan2(2 * E_Jxy_real, E_Jxx - E_Jyy))
            
            E_Handedness = (2 * E_Jxy_img) / np.sqrt((E_Jxx + E_Jyy)**2 - (4 * E_J_det))
            results['E_Ellipticity'] = np.tan(0.5 * np.arcsin(np.clip(E_Handedness, -1.0, 1.0)))

    # --- 6. APPLY NOISE GATE ---
    # Bypasses the filter if wavePowerMin is set to 0.0
    if wave_power_min > 0.0:
        mask = Wave_Power < wave_power_min
        results['Angle_Normal'][mask] = -1.0
        results['Ellipticity'][mask] = -2.0
        results['Coherency'][mask] = -1.0
    
        if En_fft is not None:
            results['E_Angle_Normal'] = np.where(mask, -1, results.get('E_Angle_Normal', -1))
            results['E_Ellipticity']  = np.where(mask, -2, results['E_Ellipticity'])
            results['E_Coherency']    = np.where(mask, -1, results['E_Coherency'])

    return results
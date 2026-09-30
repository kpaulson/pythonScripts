import numpy as np
from scipy.signal import butter, sosfiltfilt

def align_to_background(Bx, By, Bz, Rx, Ry, Rz, Bx_bg=None, By_bg=None, Bz_bg=None, fs=None, trend_seconds=None):
    """
    Transforms raw magnetic field components into the Field-Aligned Coordinate System (NPQ).
    
    This function supports two modes for determining the background field:
      1. Explicit Background: Provide Bx_bg, By_bg, Bz_bg (e.g., interpolated 1-minute L2 data).
      2. Self-Smoothing: Provide 'fs' and 'trend_seconds' to generate the background on the 
         fly using a drift-free zero-phase Butterworth lowpass filter.
    
    Parameters:
        B(x,y,z)      : rank1 numpy arrays (High-Cadence Raw Magnetic Field)
        R(x,y,z)      : rank1 numpy arrays (Radial s/c position, synchronized to B)
        B(x,y,z)_bg   : rank1 numpy arrays (Optional explicit background)
        fs            : float (Sampling frequency in Hz, required if self-smoothing)
        trend_seconds : float (Seconds for background trend, required if self-smoothing)
        
    Returns:
        Bn, Bp, Bq  : Magnetic field components in the NPQ system.
        Nx, Ny, Nz  : Unit vector components of the parallel direction (N-hat).
        Px, Py, Pz  : Unit vector components of the quasi-tangential direction (P-hat).
        Qx, Qy, Qz  : Unit vector components of the quasi-normal direction (Q-hat).
    """
    
    # --- 1. DETERMINE BACKGROUND FIELD ---
    if Bx_bg is not None and By_bg is not None and Bz_bg is not None:
        # Mode 1: Use explicitly provided background vectors
        bg_x, bg_y, bg_z = Bx_bg, By_bg, Bz_bg
        
    elif fs is not None and trend_seconds is not None:
        # Mode 2: Self-smooth the high-cadence data using a zero-phase lowpass filter
        cutoff_freq = 1.0 / trend_seconds
        nyq = 0.5 * fs
        normal_cutoff = cutoff_freq / nyq
        
        # 2nd-order Butterworth filter (applied forward and backward via sosfiltfilt)
        sos = butter(2, normal_cutoff, btype='low', output='sos')
        
        bg_x = sosfiltfilt(sos, Bx)
        bg_y = sosfiltfilt(sos, By)
        bg_z = sosfiltfilt(sos, Bz)
    else:
        raise ValueError("Must provide either explicit background vectors (Bx_bg...) OR 'fs' and 'trend_seconds' to self-smooth.")

    # --- 2. DEFINE N-HAT (Parallel) ---
    Btotal_bg = np.sqrt(bg_x**2 + bg_y**2 + bg_z**2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        Nx = bg_x / Btotal_bg
        Ny = bg_y / Btotal_bg
        Nz = bg_z / Btotal_bg

    # --- 3. DEFINE RADIAL VECTOR ---
    # Negated so NxR is westward, preserving right-handedness
    R_x, R_y, R_z = -Rx, -Ry, -Rz
    
    # --- 4. DEFINE P-HAT (Quasi-azimuthal / Tangential) ---
    # P = N x R
    TEMP_Px = (Ny * R_z) - (Nz * R_y)
    TEMP_Py = (Nz * R_x) - (Nx * R_z)
    TEMP_Pz = (Nx * R_y) - (Ny * R_x)
    
    Pmag = np.sqrt(TEMP_Px**2 + TEMP_Py**2 + TEMP_Pz**2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        Px = TEMP_Px / Pmag
        Py = TEMP_Py / Pmag
        Pz = TEMP_Pz / Pmag
        
    # --- 5. DEFINE Q-HAT (Quasi-radial / Normal) ---
    # Q = - (P x N) = N x P
    Qx = -(Py * Nz) + (Pz * Ny)
    Qy = -(Pz * Nx) + (Px * Nz)
    Qz = -(Px * Ny) + (Py * Nx)
    
    # --- 6. ROTATE HIGH-CADENCE FIELD ---
    Bn = (Bx * Nx) + (By * Ny) + (Bz * Nz)
    Bp = (Bx * Px) + (By * Py) + (Bz * Pz)
    Bq = (Bx * Qx) + (By * Qy) + (Bz * Qz)
    
    # Optional: Match the rounding from Jython to keep exact numerical equivalence
    Nx, Ny, Nz = np.round(Nx, 3), np.round(Ny, 3), np.round(Nz, 3)
    Px, Py, Pz = np.round(Px, 3), np.round(Py, 3), np.round(Pz, 3)
    Qx, Qy, Qz = np.round(Qx, 3), np.round(Qy, 3), np.round(Qz, 3)

    return Bn, Bp, Bq, Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz

def align_to_background_robust(Bx, By, Bz, Rx, Ry, Rz, Bx_bg=None, By_bg=None, Bz_bg=None, fs=None, trend_seconds=None, eps=1e-3):
    """
    Bespoke alignment function for QuickLook / Interactive exploration.
    Includes a reference vector fallback to prevent geometric singularities (NaNs) 
    during sharp field flips / switchbacks.
    """
    # 1. Determine background field (explicit or self-smoothed)
    if Bx_bg is not None and By_bg is not None and Bz_bg is not None:
        bg_x, bg_y, bg_z = Bx_bg, By_bg, Bz_bg
    elif fs is not None and trend_seconds is not None:
        cutoff_freq = 1.0 / trend_seconds
        nyq = 0.5 * fs
        normal_cutoff = cutoff_freq / nyq
        sos = butter(2, normal_cutoff, btype='low', output='sos')
        bg_x, bg_y, bg_z = sosfiltfilt(sos, Bx), sosfiltfilt(sos, By), sosfiltfilt(sos, Bz)
    else:
        raise ValueError("Must provide explicit background vectors or smoothing parameters.")

    # 2. Parallel unit vector N
    Btotal_bg = np.sqrt(bg_x**2 + bg_y**2 + bg_z**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        Nx, Ny, Nz = bg_x / Btotal_bg, bg_y / Btotal_bg, bg_z / Btotal_bg

    # 3. Primary Reference Vector R
    R_x, R_y, R_z = -Rx, -Ry, -Rz
    
    # Primary P = N x R
    TEMP_Px = (Ny * R_z) - (Nz * R_y)
    TEMP_Py = (Nz * R_x) - (Nx * R_z)
    TEMP_Pz = (Nx * R_y) - (Ny * R_x)
    Pmag = np.sqrt(TEMP_Px**2 + TEMP_Py**2 + TEMP_Pz**2)

    with np.errstate(divide='ignore', invalid='ignore'):
        Px, Py, Pz = TEMP_Px / Pmag, TEMP_Py / Pmag, TEMP_Pz / Pmag

    # 4. Dynamic Fallback for N || R alignment
    singular_mask = (Pmag < eps) | (~np.isfinite(Pmag))
    if np.any(singular_mask):
        # Substitute alternate reference vector R_alt = [1, 0, 0] (negated to [-1, 0, 0])
        R_alt_x, R_alt_y, R_alt_z = -1.0, 0.0, 0.0
        
        ALT_Px = (Ny * R_alt_z) - (Nz * R_alt_y)
        ALT_Py = (Nz * R_alt_x) - (Nx * R_alt_z)
        ALT_Pz = (Nx * R_alt_y) - (Ny * R_alt_x)
        ALT_Pmag = np.sqrt(ALT_Px**2 + ALT_Py**2 + ALT_Pz**2)

        Px[singular_mask] = ALT_Px[singular_mask] / ALT_Pmag[singular_mask]
        Py[singular_mask] = ALT_Py[singular_mask] / ALT_Pmag[singular_mask]
        Pz[singular_mask] = ALT_Pz[singular_mask] / ALT_Pmag[singular_mask]

    # 5. Q = N x P
    Qx = -(Py * Nz) + (Pz * Ny)
    Qy = -(Pz * Nx) + (Px * Nz)
    Qz = -(Px * Ny) + (Py * Nx)

    # 6. Transform high-res fields
    Bn = (Bx * Nx) + (By * Ny) + (Bz * Nz)
    Bp = (Bx * Px) + (By * Py) + (Bz * Pz)
    Bq = (Bx * Qx) + (By * Qy) + (Bz * Qz)

    return Bn, Bp, Bq, Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz
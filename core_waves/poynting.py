import numpy as np
import warnings

def Poyntingify(Bn_fft, Bp_fft, Bq_fft, En_fft, Ep_fft, Eq_fft, poynting_min):
	"""
	3D Poynting analysis performed on spectral components.
	
	Parameters:
		Bn_fft, Bp_fft, Bq_fft = rank 2 complex numpy arrays (Time, Freq)
		En_fft, Ep_fft, Eq_fft = rank 2 complex numpy arrays (Time, Freq)
		poynting_min           = minimum value of Poynting flux to be kept in mW/m/Hz 
								 (all lower values discarded).
								 
	Returns:
		Smag, Poynting_theta, Poynting_phi, Sn, Sp, Sq, TEMP_Sn, TEMP_Sp, TEMP_Sq
	"""
	
	# 1. Stack the individual components into 3D vectors
	# Shape becomes (Time, Frequency, 3) where the last axis is [n, p, q]
	B_vec = np.stack((Bn_fft, Bp_fft, Bq_fft), axis=-1)
	E_vec = np.stack((En_fft, Ep_fft, Eq_fft), axis=-1)
	
	# 2. Calculate the Poynting Flux Vector: S = 1/2 * Re(E x B*)
	# This single line perfectly replaces the 6 lines of TEMP_Sn/Sp/Sq real/imaginary math
	S_vec = 0.5 * np.real(np.cross(E_vec, np.conj(B_vec), axis=-1))
	
	# 3. Extract the un-normalized components (matching your old Jython variables)
	TEMP_Sn = S_vec[..., 0]
	TEMP_Sp = S_vec[..., 1]
	TEMP_Sq = S_vec[..., 2]
	
	# 4. Calculate Magnitude
	Smag = np.sqrt(TEMP_Sn**2 + TEMP_Sp**2 + TEMP_Sq**2)
	
	# 5. Normalize the vectors
	# (Using errstate to silently handle bins with zero wave power, avoiding console spam)
	with np.errstate(divide='ignore', invalid='ignore'):
		Sn = TEMP_Sn / Smag
		Sp = TEMP_Sp / Smag
		Sq = TEMP_Sq / Smag
		
	# 6. Calculate Angles (in degrees)
	with np.errstate(invalid='ignore'):
		# Theta: Angle from the background magnetic field (n-hat)
		Poynting_theta = np.degrees(np.arccos(np.clip(Sn, -1.0, 1.0)))
		
		# Phi: Clock angle in the transverse plane (p-q plane)
		Poynting_phi = np.degrees(np.arctan2(Sq, Sp))
		
	# 7. Apply the noise gate (step function for low wave power)
	# Bypasses the filter if poynting_min is set to 0.0
	if poynting_min > 0.0:
		mask = Smag < poynting_min
		
		Poynting_theta[mask] = -1
		Poynting_phi[mask]   = -181
		Sn[mask]             = -1e38
		Sp[mask]             = -1e38
		Sq[mask]             = -1e38
		TEMP_Sn[mask]        = -1e38
		TEMP_Sp[mask]        = -1e38
		TEMP_Sq[mask]        = -1e38
	
	return Smag, Poynting_theta, Poynting_phi, Sn, Sp, Sq, TEMP_Sn, TEMP_Sp, TEMP_Sq
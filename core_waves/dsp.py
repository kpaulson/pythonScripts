import numpy as np
from scipy.signal import stft

def Hanningfft(ds, fs, freq_min, freq_max, window, slide):
    """
    FFT routine employing a sliding Hanning window function.
    
    The divisor 1.732 (sqrt(3)) is the theoretical Power Spectral Density (PSD) energy 
    correction factor required to compensate for the Equivalent Noise Bandwidth (ENBW) 
    of the Hanning window.
    
    Parameters:
        ds       = The 1D time-series input data.
        fs       = Sampling frequency of the input data (Hz).
        freq_min = Lower end of frequency range to be preserved.
        freq_max = Upper end of frequency range.
        window   = Length of window to be taken measured in number of data points.
        slide    = Window overlap size, expressed as a reciprocal fraction of the length 
                   (e.g., 8 means steps of window/8, or 87.5% overlap).
        
    Returns:
        fft_time       = 1D array of time centers for the FFT windows (in seconds from start).
        frequency_axis = 1D array of frequencies (cropped to freq_min and freq_max).
        waveform_fft   = 2D complex numpy array of shape (Time, Frequency).
    """
    
    # Calculate overlap based on the slide factor
    step_size = int(window / slide)
    noverlap = window - step_size
    
    # SciPy's STFT handles the sliding window, padding, and FFT efficiently in C.
    # Note: 'return_onesided=True' automatically handles the factor of 2 for the 
    # positive frequencies that you previously calculated manually.
    frequency_axis, fft_time, waveform_fft = stft(
        ds, 
        fs=fs, 
        window='hann', 
        nperseg=window, 
        noverlap=noverlap,
        return_onesided=True
    )
    
    # SciPy outputs shape (Freq, Time). Transpose to (Time, Freq) to match your workflow.
    waveform_fft = waveform_fft.T
    
    # Apply your specific PSD energy correction factor for the Hanning window 
    # to perfectly match your old Jython output amplitudes.
    waveform_fft = waveform_fft * np.sqrt(window / fs) / 1.732
    
    # Identify the frequency range to keep
    valid_freqs = (frequency_axis >= freq_min) & (frequency_axis <= freq_max)
    
    # Crop the frequency axis and the FFT array, then return
    return fft_time, frequency_axis[valid_freqs], waveform_fft[:, valid_freqs]


def process_all_ffts(Bn, Bp, Bq, En, Ep, Eq, fs, freq_min, freq_max, window, slide):
    """
    Helper function to process all magnetic and electric components cleanly in one shot.
    
    Returns:
        fft_time, frequency_axis, Bn_fft, Bp_fft, Bq_fft, En_fft, Ep_fft, Eq_fft
    """
    # Calculate B-field FFTs
    fft_time, frequency_axis, Bn_fft = Hanningfft(Bn, fs, freq_min, freq_max, window, slide)
    _, _, Bp_fft = Hanningfft(Bp, fs, freq_min, freq_max, window, slide)
    _, _, Bq_fft = Hanningfft(Bq, fs, freq_min, freq_max, window, slide)
    
    # Calculate E-field FFTs (if valid data was passed)
    if En is not None:
        _, _, En_fft = Hanningfft(En, fs, freq_min, freq_max, window, slide)
        _, _, Ep_fft = Hanningfft(Ep, fs, freq_min, freq_max, window, slide)
        _, _, Eq_fft = Hanningfft(Eq, fs, freq_min, freq_max, window, slide)
    else:
        En_fft, Ep_fft, Eq_fft = None, None, None
        
    return fft_time, frequency_axis, Bn_fft, Bp_fft, Bq_fft, En_fft, Ep_fft, Eq_fft
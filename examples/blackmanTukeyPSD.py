import matplotlib.pyplot as plt 
import numpy as np 
from scipy import signal 
from scipy.signal import windows as win

from spectrum.correlation import CORRELATION, xcorr 
from spectrum import CORRELOGRAMPSD

# Composite signal 
sample_intv = 1 
sample_freq = 1/sample_intv 
t = np.linspace(1, 1000, 1000) 
count = t.size

def make_signal(t, amplitude, period, phase=0):
    return amplitude * np.sin(2 * np.pi * t/period - phase)

x_time = (make_signal(t, amplitude=2.0, period=50) +
          make_signal(t, amplitude=1.0, period=15) +
          make_signal(t, amplitude=0.5, period=5))

lags = signal.correlation_lags(count, count) 
valid = (np.abs(lags) < count) 
lags = lags[valid]

unscaled_xcorr = signal.correlate(x_time, x_time, mode='full')[valid] 
corr = unscaled_xcorr/(count - np.abs(lags)) 
corr_test,lags_test = xcorr(x_time,x_time,999,norm='biased')

def blakman_tukey(xcorr, sample_freq, lags=None, window='boxcar'):

    # Calculate peridogram resolution (Stoica, 2005, p.37-38)
    #f_res = 1/xcorr.size
    f_res = len(xcorr)
    #f_max = (f_res-1) / f_res
    f_full = np.linspace(0, (f_res-1)/(f_res), f_res)
    over_half = f_full > .5
    f_half = f_full[~over_half]

    # Frequency-domain representation of the data
    weighted_xcorr = xcorr * win.get_window(window, len(xcorr))
    
    if lags is None:
        x_freq = np.fft.fft(weighted_xcorr)
        #x_freq = np.fft.fftshift(x_freq)
    else:
        xcorr_2d, f_2d = np.meshgrid(weighted_xcorr, f_full)
        lags_2d, f_2d = np.meshgrid(lags, f_full)
        x_freq = np.sum(xcorr_2d * np.exp(-1j * 2 * np.pi * f_2d * lags_2d/sample_freq), axis=1)

    # Calculate the PSD from the frequency-domain representation    
    psd = (np.abs(x_freq)) # 
    psd = psd[~over_half]
    psd[1:] *= 2

    return f_half, psd

f_per, x_per = signal.periodogram(x_time) 
f_a, psd_a = blakman_tukey(corr_test, sample_freq) 
f_b, psd_b = blakman_tukey(corr_test, sample_freq, 999)

psd_spec = CORRELOGRAMPSD(x_time, norm='unbiased', lag=999, NFFT=1000, window='rectangle') 
f_spec = np.linspace(0, 1, psd_spec.size) 
psd_spec = psd_spec[f_spec <= .5] 
f_spec = f_spec[f_spec <= .5] 
psd_spec[1:] *=2

plt.figure(1)
plt.plot(f_per, x_per)

plt.figure(2)
plt.plot(f_spec, psd_spec)

plt.figure(3)
plt.plot(f_a,psd_a)

plt.figure(4)
plt.plot(f_b,psd_b)
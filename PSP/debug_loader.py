import os
from datetime import datetime, timezone
import cdflib
import numpy as np

# Import your tools directly from the main script
from psp_interactivePlot_generator import load_cdf_data, PLOT_VARS, ANALYSIS_ROOT, VERSION

print("--- INITIALIZING DEBUGGER ---")

# Setup the exact target date
dt = datetime(2023, 10, 4)
y, m, d = dt.strftime('%Y'), dt.strftime('%m'), dt.strftime('%d')

# Construct the file paths manually
folder = os.path.join(ANALYSIS_ROOT, y, m)
files = [
    os.path.join(folder, f"PSP_WaveAnalysis_{y}-{m}-{d}_0000_{VERSION}.cdf"),
    os.path.join(folder, f"PSP_WaveAnalysis_{y}-{m}-{d}_0600_{VERSION}.cdf"),
    os.path.join(folder, f"PSP_WaveAnalysis_{y}-{m}-{d}_1200_{VERSION}.cdf"),
    os.path.join(folder, f"PSP_WaveAnalysis_{y}-{m}-{d}_1800_{VERSION}.cdf")
]

print("Target files:")
for f in files: print(f"  {f}")

# Extract
target_vars = PLOT_VARS['waveAnalysis_daily']
print("\n--- RUNNING LOAD_CDF_DATA ---")
t_b_raw, t_fft_raw, a_data, master_freq = load_cdf_data(files, target_vars)

print("\n--- DEBUG OUTPUT ---")
if a_data is None:
    print("CRITICAL: a_data is NONE! The loader failed completely.")
else:
    print(f"master_freq shape: {master_freq.shape if master_freq is not None else 'None'}")
    print(f"t_b_raw length: {len(t_b_raw) if t_b_raw is not None else 0}")
    print(f"t_fft_raw length: {len(t_fft_raw) if t_fft_raw is not None else 0}")
    
    print("\nArray Structures in a_data:")
    for k, v in a_data.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:15}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k:15}: type={type(v)}")

    print("\n--- TIME CONVERSION TEST ---")
    if t_b_raw is not None and len(t_b_raw) > 0:
        # Grab the very first timestamp in the array
        first_raw = t_b_raw[0]
        
        # Determine the target Unix boundaries
        start_unix = dt.replace(tzinfo=timezone.utc).timestamp()
        
        try:
            first_unix = cdflib.cdfepoch.unixtime(first_raw)
            dt_parsed = datetime.fromtimestamp(first_unix, tz=timezone.utc)
            
            print(f"Raw Epoch Value : {first_raw}")
            print(f"Decoded to Unix : {first_unix}")
            print(f"Decoded to UTC  : {dt_parsed}")
            print(f"Target Start    : {start_unix} ({dt.replace(tzinfo=timezone.utc)})")
            
            # This is the exact math your Hard Filter uses
            if first_unix >= start_unix:
                print(">> SUCCESS: The first point falls INSIDE the filter window!")
            else:
                print(f">> FAIL: The first point is {start_unix - first_unix} seconds OUTSIDE the window.")
                
        except Exception as e:
            print(f"Conversion crashed: {e}")
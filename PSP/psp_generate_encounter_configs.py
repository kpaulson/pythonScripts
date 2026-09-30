import numpy as np
from datetime import datetime, timedelta
from scipy.signal import find_peaks

# Import the ephemeris loader you already built!
from psp_interactivePlot_generator import load_ephemeris

print("Loading mission ephemeris from CDF...")
eph_funcs = load_ephemeris()

if not eph_funcs or 'RAD_AU' not in eph_funcs:
    print("[!] Error: Could not load RAD_AU from the ephemeris CDF.")
    exit()

rad_func = eph_funcs['RAD_AU']

# 1. Create a daily timeline from launch (Aug 2018) to Dec 2029
start_dt = datetime(2018, 8, 12)
end_dt = datetime(2029, 12, 31)
total_days = (end_dt - start_dt).days

days = [start_dt + timedelta(days=i) for i in range(total_days)]
unix_times = np.array([(d - datetime(1970, 1, 1)).total_seconds() for d in days])

# 2. Evaluate the spacecraft's distance for every single day
r_au = rad_func(unix_times)

# ==========================================
# ENCOUNTER CALCULATION (r_au < 0.25)
# ==========================================
under_thresh = np.where(r_au < 0.25)[0]
encounter_blocks = np.split(under_thresh, np.where(np.diff(under_thresh) > 20)[0] + 1)

# ==========================================
# ORBIT CALCULATION (Aphelion to Aphelion)
# ==========================================
# Find peaks in radial distance (aphelia). 
# We set distance=60 days since the shortest orbit is 88 days, preventing false positives.
aphelia_idx, _ = find_peaks(r_au, distance=60)

# Prepend the launch date (index 0) so Orbit 1 is [Launch -> Aphelion 1]
orbit_boundaries = [0] + list(aphelia_idx)


# ==========================================
# OUTPUT GENERATION
# ==========================================
print("\n=======================================================")
print("--- PASTE THIS INTO config.py (Python Backend) ---")
print("=======================================================")
print("ENCOUNTER_DATES = {")
for i, enc in enumerate(encounter_blocks):
    if len(enc) == 0: continue
    enc_id = i + 1
    s_date = days[enc[0]].strftime('%Y-%m-%d')
    e_date = days[enc[-1]].strftime('%Y-%m-%d')
    print(f"    {enc_id}: ('{s_date}', '{e_date}'),")
print("}\n")

print("ORBIT_DATES = {")
for i in range(len(orbit_boundaries) - 1):
    orb_id = i + 1
    s_date = days[orbit_boundaries[i]].strftime('%Y-%m-%d')
    e_date = days[orbit_boundaries[i+1]].strftime('%Y-%m-%d')
    print(f"    {orb_id}: ('{s_date}', '{e_date}'),")
print("}")


print("\n=======================================================")
print("--- PASTE THIS INTO config.js (Web Frontend) ---")
print("=======================================================")
print("const ENCOUNTER_DATES = {")
for i, enc in enumerate(encounter_blocks):
    if len(enc) == 0: continue
    enc_id = i + 1
    s_date = days[enc[0]].strftime('%Y-%m-%d')
    print(f"    {enc_id}: '{s_date}',")
print("};\n")

print("const ORBIT_DATES = {")
for i in range(len(orbit_boundaries) - 1):
    orb_id = i + 1
    s_date = days[orbit_boundaries[i]].strftime('%Y-%m-%d')
    print(f"    {orb_id}: '{s_date}',")
print("};")
print("\nDone!")
import gzip
import json

json_path = "/home/kpaulson/MyDrive/Research/PSP/JSON/team/E29/psp_mag_enc_29.json.gz"

with gzip.open(json_path, 'rt', encoding='utf-8') as f:
    payload = json.load(f)

times = payload.get('times', {}).get('mag', [])
b_r = payload.get('data', {}).get('b_r', [])
b_tot = payload.get('data', {}).get('b_tot', [])

valid_br_count = sum(1 for v in b_r if v is not None)
valid_btot_count = sum(1 for v in b_tot if v is not None)

print("\n================ JSON PAYLOAD DIAGNOSTIC ================")
print(f"Metadata Orbit Bounds: {payload.get('metadata', {}).get('orbit_start')} to {payload.get('metadata', {}).get('orbit_end')}")
print(f"Total Timestamps Exported : {len(times)}")
if times:
    print(f"First Timestamp           : {times[0]}")
    print(f"Last Timestamp            : {times[-1]}")
print(f"Total B_R Points          : {len(b_r)} (Non-Null: {valid_br_count})")
print(f"Total |B| Points          : {len(b_tot)} (Non-Null: {valid_btot_count})")

# Print non-null sample for Sep 02 date range if present
sep2_points = [(t, r) for t, r in zip(times, b_r) if t and "2026-09-02" in t and r is not None]
print(f"Sep 02 Non-Null Points    : {len(sep2_points)}")
if sep2_points:
    print(f"Sample Sep 02 Point       : Time={sep2_points[0][0]}, Br={sep2_points[0][1]} nT")
print("=========================================================\n")
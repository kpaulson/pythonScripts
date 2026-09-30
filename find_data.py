import os
import glob

# Point to the root of your Autoplot Cache
CACHE_ROOT = '/mnt/sweaparc/sa-home/kpaulson/MyDrive/Research/Data/AutoplotCache'

print("=== Autoplot Cache Hunter for 2026-01-06 ===\n")

# Recursively search the entire cache for any file containing the instrument and date
pattern = os.path.join(CACHE_ROOT, '**', '*mag_SC*20260106*.cdf')
matches = glob.glob(pattern, recursive=True)

if not matches:
	print("[-] No mag_SC files found for 20260106 anywhere in the cache.")
else:
	for f in sorted(matches):
		# Print the path relative to the cache root so we can see the server and protocol
		print(f"[+] {os.path.relpath(f, CACHE_ROOT)}")
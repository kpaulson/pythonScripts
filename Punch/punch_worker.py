import sys
import os
import glob
import numpy as np
from astropy.io import fits
from astropy.utils.data import download_file

def process_file(url, bin_path, meta_path):
    try:
        # --- THE JANITOR LOGIC ---
        # Get the directory and look for old punch_data_*.bin files
        temp_dir = os.path.dirname(bin_path)
        old_files = glob.glob(os.path.join(temp_dir, "punch_data_*.bin"))
        old_metas = glob.glob(os.path.join(temp_dir, "punch_meta_*.txt"))
        
        for f in old_files + old_metas:
            try:
                # We try to delete. If Autoplot has it locked, it will fail silently.
                # If Autoplot has released it, it gets cleaned up.
                os.remove(f)
            except OSError:
                pass 
        # -------------------------

        print(f"Downloading: {url}")
        tmp_fits = download_file(url, cache=True)
        
        with fits.open(tmp_fits) as hdul:
            image_data = hdul[1].data.astype('float32')
            image_data.tofile(bin_path)
            
            with open(meta_path, 'w') as f:
                f.write(f"{image_data.shape[1]},{image_data.shape[0]}")
        
        print(f"Success: Created {os.path.basename(bin_path)}")
            
    except Exception as e:
        with open(meta_path, 'w') as f:
            f.write(f"ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    process_file(sys.argv[1], sys.argv[2], sys.argv[3])
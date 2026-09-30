"""
===============================================================================
PSP ANALYSIS BROWSER - CONFIGURATION TEMPLATE
===============================================================================
INSTRUCTIONS FOR NEW USERS:
1. Save a copy of this file in the script folder as `config.py`.
2. Edit the 5 string variables in Section 1 below to match your local paths
   or SPEDAS/Autoplot cache directories.
3. You're done! The browser will read these functions automatically.
===============================================================================
"""

import os

# =============================================================================
# 1. BASE PATH DEFINITIONS (EDIT THESE STRINGS)
# =============================================================================

# Root directory for your PSP research / Google Drive folder
# (Windows example: "G:/My Drive" or "C:/Users/Name/Research")
# (Linux example:   "/home/username/Research")
DRIVE_ROOT = "C:/Users/YourName/Research"

# Directory where SWEAP data lives (SPAN-I, SPC, etc.)
# Expected subfolder layout: .../spi/L2/spi_sf00/YYYY/MM/
SWEAP_DIR = "C:/Users/YourName/Research/Data/AutoplotCache/https/w3sweap.cfa.harvard.edu/data/sci"

# Directory where FIELDS/MAG data lives
# Expected subfolder layout: .../fields/l2/mag_SC_4_Sa_per_Cyc/YYYY/MM/
FIELDS_DIR = "C:/Users/YourName/Research/Data/AutoplotCache/http/research.ssl.berkeley.edu/data/spp/data/sci"

# Directory for shared PSP team data (contains 15min_DATA_Internal CSV files like E26.csv)
SHARED_DRIVE_ROOT = "C:/Users/YourName/Research/SharedDrives/PSP_SWEAP"

# Temporary directory on fast local storage (NVMe/SSD) for staging CDF copies
TMP_DIR = "C:/tmp_psp"


# =============================================================================
# 2. REQUIRED PATH GETTER FUNCTIONS
# spani_vdfDynamicBrowser.py calls these functions directly.
# =============================================================================

def get_drive_path():
    """Returns the base path for your research drive."""
    return DRIVE_ROOT

def get_sharedDrivePSP_path():
    """Returns the base path for shared PSP team data."""
    return SHARED_DRIVE_ROOT

def get_sweapCacheData():
    """Returns the base directory for SWEAP CDF files."""
    return SWEAP_DIR

def get_berkeleyCacheData():
    """Returns the base directory for FIELDS/MAG CDF files."""
    return FIELDS_DIR

def get_tmpMirror():
    """Returns a local temporary staging directory."""
    os.makedirs(TMP_DIR, exist_ok=True)
    return TMP_DIR


# =============================================================================
# 3. MISSION ENCOUNTER BOUNDARIES
# Maps date ranges to Parker Solar Probe Encounter numbers.
# =============================================================================
ENCOUNTER_DATES = {
    1:  ('2018-11-01', '2018-11-11'),
    2:  ('2019-03-31', '2019-04-10'),
    3:  ('2019-08-28', '2019-09-07'),
    4:  ('2020-01-24', '2020-02-04'),
    5:  ('2020-06-02', '2020-06-13'),
    6:  ('2020-09-22', '2020-10-02'),
    7:  ('2021-01-13', '2021-01-23'),
    8:  ('2021-04-25', '2021-05-04'),
    9:  ('2021-08-05', '2021-08-15'),
    10: ('2021-11-17', '2021-11-26'),
    11: ('2022-02-21', '2022-03-02'),
    12: ('2022-05-28', '2022-06-07'),
    13: ('2022-09-02', '2022-09-11'),
    14: ('2022-12-07', '2022-12-16'),
    15: ('2023-03-13', '2023-03-23'),
    16: ('2023-06-17', '2023-06-27'),
    17: ('2023-09-23', '2023-10-03'),
    18: ('2023-12-25', '2024-01-03'),
    19: ('2024-03-26', '2024-04-04'),
    20: ('2024-06-26', '2024-07-05'),
    21: ('2024-09-26', '2024-10-05'),
    22: ('2024-12-20', '2024-12-29'),
    23: ('2025-03-19', '2025-03-27'),
    24: ('2025-06-15', '2025-06-24'),
    25: ('2025-09-11', '2025-09-20'),
    26: ('2025-12-09', '2025-12-18'),
    27: ('2026-03-07', '2026-03-16'),
    28: ('2026-06-04', '2026-06-13'),
    29: ('2026-08-31', '2026-09-09'),
    30: ('2026-11-28', '2026-12-06'),
    31: ('2027-02-24', '2027-03-05'),
    32: ('2027-05-23', '2027-06-01'),
    33: ('2027-08-20', '2027-08-29'),
    34: ('2027-11-16', '2027-11-25'),
    35: ('2028-02-13', '2028-02-22'),
    36: ('2028-05-11', '2028-05-20'),
    37: ('2028-08-08', '2028-08-16'),
    38: ('2028-11-04', '2028-11-13'),
}
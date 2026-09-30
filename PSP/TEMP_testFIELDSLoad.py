import os
import sys
import re
import urllib.request
import urllib.parse
from datetime import datetime

# Append software directory to load auth and config
sys.path.append("/home/kpaulson/MyDrive/Software/Python/githubScripts_python")
import psp_swp_browser_dataProcessor as dp

target_date_str = "20260906"
year_str = "2026"
month_str = "09"
relative_subpath = "fields/l2/mag_RTN_4_Sa_per_Cyc/"

print("\n================ BERKELEY HTTP SYNC DIAGNOSTIC ================")

# 1. Load Credentials
creds = dp.load_auth_credentials()
user, passw = creds.get('BERKELEY_USER'), creds.get('BERKELEY_PASS')
print(f"1. .auth File Check      : User = '{user}', Password Set = {bool(passw)}")

# 2. Derive Target Local AutoplotCache Path
base_http_url = "http://research.ssl.berkeley.edu/data/spp/data/sci/"
autoplot_base_dir = dp.get_autoplot_cache_dir(base_http_url, relative_subpath)
target_local_dir = os.path.join(autoplot_base_dir, year_str, month_str)
target_local_file = os.path.join(target_local_dir, f"psp_fld_l2_mag_RTN_4_Sa_per_Cyc_{target_date_str}_v00.cdf")

print(f"2. Target Local Directory : {target_local_dir}")
print(f"   Target Local File Path : {target_local_file}")
print(f"   Local File Exists Now  : {os.path.exists(target_local_file)}")

# 3. Query Remote Berkeley Directory
month_url = f"{base_http_url}{relative_subpath}{year_str}/{month_str}/"
print(f"3. Remote Directory URL   : {month_url}")

if not user or not passw:
    print("\n[!] ABORT: Credentials missing in .auth file.")
    sys.exit(1)

password_mgr = urllib.request.HTTPPasswordMgrWithPriorAuth()
password_mgr.add_password(None, base_http_url, user, passw, is_authenticated=True)
handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
opener = urllib.request.build_opener(handler)

try:
    req = urllib.request.Request(month_url, headers={'User-Agent': 'Mozilla/5.0'})
    with opener.open(req, timeout=12) as resp:
        http_code = resp.getcode()
        final_url = resp.geturl()
        html_data = resp.read().decode('utf-8', errors='ignore')
        
    print(f"   HTTP Response Code    : {http_code}")
    print(f"   Final Resolved URL    : {final_url}")
    print(f"   Raw HTML Bytes Read   : {len(html_data)} bytes")

    # 4. Parse Remote File Links
    all_cdfs = re.findall(r'href=["\']([^"\']+\.cdf)["\']', html_data, re.IGNORECASE)
    date_matches = [f for f in all_cdfs if target_date_str in f]

    print(f"\n4. Directory Link Scraping:")
    print(f"   Total CDFs in Month   : {len(all_cdfs)}")
    print(f"   Matches for {target_date_str}   : {date_matches}")

    if date_matches:
        target_remote_file = date_matches[-1]
        full_file_url = urllib.parse.urljoin(month_url, target_remote_file)
        print(f"\n[SUCCESS] Found remote file online!")
        print(f"   Remote Download URL   : {full_file_url}")
        print(f"   Intended Cache Path   : {os.path.join(target_local_dir, os.path.basename(target_remote_file))}")
    else:
        print(f"\n[FAIL] No CDF matching date '{target_date_str}' found in HTML listing.")
        if len(html_data) < 200:
            print(f"   HTML Snippet: {html_data}")

except Exception as e:
    print(f"\n[ERROR] HTTP Request Failed: {e}")

print("===============================================================\n")
import argparse
import os
import sys
import glob
import tempfile
import shutil
import platform
import numpy as np
import cdflib
from datetime import datetime, timedelta

# Look up one level to find config file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import config

DRIVE_ROOT = config.get_drive_path()
CUSTOM_TMP = config.get_tmpMirror()

# --- OS ROUTER LOGIC ---
if platform.system() == 'Windows':
    ACTIVE_TMP = None 
    io_message = "(via Local Windows Temp)"
else:
    ACTIVE_TMP = CUSTOM_TMP
    os.makedirs(ACTIVE_TMP, exist_ok=True)
    io_message = "(via Linux RAM disk)"
# -----------------------

def setup():
    parser = argparse.ArgumentParser(description="Upgrades WaveAnalysis CDFs by scaling E-field and Poynting data to account for effective antenna length.")
    parser.add_argument('--date', type=str, required=True, help="Format: 'YYYY-MM-DD through YYYY-MM-DD'")
    parser.add_argument('-v_in', type=str, default="1.3", help="Input WaveAnalysis version")
    parser.add_argument('-v_out', type=str, default="1.4", help="Output WaveAnalysis version")
    parser.add_argument('-L_eff', type=float, default=3.5, help="Effective Antenna Length in meters (Default: 14.0)")
    return parser.parse_args()

def main():
    args = setup()
    
    try:
        date_parts = args.date.split(' through ')
        start_date = datetime.strptime(date_parts[0].strip(), '%Y-%m-%d')
        end_date   = datetime.strptime(date_parts[1].strip(), '%Y-%m-%d')
    except Exception as e:
        print(f"[!] Error parsing --date string: {e}")
        return

    print(f"[*] Commencing Version Upgrade Loop: {start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}")
    print(f"[*] Scaling E-field data from Volts to mV/m using L_eff = {args.L_eff}m")
    
    ANALYSIS_IN_ROOT  = f'{DRIVE_ROOT}/Research/PSP/WaveAnalysis/WaveAnalysis_Files/v{args.v_in}/'
    ANALYSIS_OUT_ROOT = f'{DRIVE_ROOT}/Research/PSP/WaveAnalysis/WaveAnalysis_Files/v{args.v_out}/'

    # The scaling factors
    # Ex (V) * (1000 / L_eff) = Ex (mV/m)
    linear_scale = 1000.0 / args.L_eff
    power_scale  = linear_scale ** 2

    # Variables that scale linearly (Amplitudes and Poynting Flux)
    linear_vars = ['S_mag', 'Sn', 'Sp', 'Sq', 'En_fft', 'Ep_fft', 'Eq_fft']
    # Variables that scale quadratically (Power)
    power_vars  = ['Wave_Power_e', 'E_power_para', 'E_power_perp']

    current_date = start_date
    while current_date <= end_date:
        y, m, d = current_date.strftime('%Y'), current_date.strftime('%m'), current_date.strftime('%d')
        
        in_dir = os.path.join(ANALYSIS_IN_ROOT, y, m)
        out_dir = os.path.join(ANALYSIS_OUT_ROOT, y, m)
        os.makedirs(out_dir, exist_ok=True)
        
        # Grab all files for this specific day
        daily_files = glob.glob(os.path.join(in_dir, f"PSP_WaveAnalysis_{y}-{m}-{d}_*_v{args.v_in}.cdf"))
        
        if not daily_files:
            print(f"  [-] No v{args.v_in} files found for {y}-{m}-{d}. Skipping.")
            current_date += timedelta(days=1)
            continue
            
        print(f"\n[=========== Upgrading Day: {y}-{m}-{d} ===========]")
        
        for in_file in sorted(daily_files):
            basename = os.path.basename(in_file)
            out_basename = basename.replace(f"_v{args.v_in}.cdf", f"_v{args.v_out}.cdf")
            final_out_path = os.path.join(out_dir, out_basename)
            
            print(f"  -> Processing: {basename}")
            
            if os.path.exists(final_out_path):
                print(f"      [✓] Output already exists. Skipping.")
                continue
                
            # Safely copy to RAM disk/Temp to avoid network I/O lag during read/write
            fd1, tmp_in = tempfile.mkstemp(suffix='.cdf', dir=ACTIVE_TMP)
            os.close(fd1)
            fd2, tmp_out = tempfile.mkstemp(suffix='.cdf', dir=ACTIVE_TMP)
            os.close(fd2)
            os.remove(tmp_out) # cdfwrite wants to create the file itself
            
            try:
                shutil.copyfile(in_file, tmp_in)
                
                with cdflib.CDF(tmp_in) as cdf_in:
                    cdf_out = cdflib.cdfwrite.CDF(tmp_out)
                    
                    # 1. Copy and Reformat Global Attributes
                    raw_global_atts = cdf_in.globalattsget()
                    global_atts = {}
                    
                    # cdflib write_globalattrs requires the {0: 'value'} dictionary structure
                    for k, v in raw_global_atts.items():
                        if isinstance(v, dict):
                            global_atts[k] = v
                        elif isinstance(v, (list, np.ndarray)):
                            global_atts[k] = {i: val for i, val in enumerate(v)}
                        else:
                            global_atts[k] = {0: v}

                    # Safely modify the re-packed dictionary
                    if 'Data_version' in global_atts and 0 in global_atts['Data_version']:
                        global_atts['Data_version'][0] = f"v{args.v_out}"
                    else:
                        global_atts['Data_version'] = {0: f"v{args.v_out}"}
                        
                    if 'TEXT' in global_atts and 0 in global_atts['TEXT']:
                        global_atts['TEXT'][0] = str(global_atts['TEXT'][0]) + f" | Scaled to mV/m using L_eff = {args.L_eff}m."
                    else:
                        global_atts['TEXT'] = {0: f"Scaled to mV/m using L_eff = {args.L_eff}m."}
                        
                    cdf_out.write_globalattrs(global_atts)
                    
                    # 2. Loop through every variable and copy/scale
                    # Access zVariables via attribute (.zVariables) instead of dictionary key
                    for var_name in cdf_in.cdf_info().zVariables:
                        v_data = cdf_in.varget(var_name)
                        v_atts = cdf_in.varattsget(var_name)
                        v_info = cdf_in.varinq(var_name)
                        
                        # Apply Scaling
                        if var_name in linear_vars:
                            v_data = v_data * linear_scale
                        elif var_name in power_vars:
                            v_data = v_data * power_scale
                            
                        # Rebuild Var Spec using object attributes (.Data_Type, etc.)
                        var_spec = {
                            'Variable': var_name,
                            'Data_Type': v_info.Data_Type,
                            'Num_Elements': v_info.Num_Elements,
                            'Rec_Vary': v_info.Rec_Vary,
                            'Dim_Sizes': v_info.Dim_Sizes,
                            'Data': v_data
                        }
                        
                        cdf_out.write_var(var_spec, var_attrs=v_atts, var_data=v_data)
                    
                    # 3. Add the new effective length variable
                    leff_data = np.array([args.L_eff], dtype=np.float32)
                    leff_spec = {
                        'Variable': 'efieldAntenna_effectiveLength',
                        'Data_Type': 44, # CDF_FLOAT
                        'Num_Elements': 1,
                        'Rec_Vary': False, # It's a scalar constant for the file
                        'Dim_Sizes': [],
                        'Data': leff_data
                    }
                    leff_atts = {
                        'CATDESC': 'Effective Antenna Length used to scale Volts to mV/m',
                        'FIELDNAM': 'Effective Antenna Length',
                        'UNITS': 'meters',
                        'FORMAT': 'F6.2'
                    }
                    cdf_out.write_var(leff_spec, var_attrs=leff_atts, var_data=leff_data)
                    
                    cdf_out.close()
                
                # Move finished file back to the network drive
                shutil.move(tmp_out, final_out_path)
                print(f"      [+] Success: Written to v{args.v_out}")
                
            except Exception as e:
                print(f"      [!] Failed to process {basename}: {e}")
                
            finally:
                if os.path.exists(tmp_in): os.remove(tmp_in)
                if os.path.exists(tmp_out): os.remove(tmp_out)

        current_date += timedelta(days=1)
        
    print("\n[+] Upgrade Complete!")

if __name__ == "__main__":
    main()
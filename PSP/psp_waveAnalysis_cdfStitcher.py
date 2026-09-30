import cdflib
import numpy as np
import sys
import os
import time
import shutil
import tempfile
from datetime import datetime, timezone

def run_stitcher(config_path):
    manifest_found = False
    for i in range(15): 
        if os.path.exists(config_path):
            manifest_found = True
            break
        time.sleep(0.5) 

    if not manifest_found: raise FileNotFoundError(f"Manifest not found: {config_path}")
        
    with open(config_path, 'r') as f:
        lines = f.readlines()

    header = lines[0].strip().split('|')
    output_path, time_var, done_flag_path = header[0], header[1], header[2]
    
    global_params_raw = lines[1].strip().split(';')
    input_params = {f'PARAM_{p.split("=")[0]}': {0: p.split("=")[1]} for p in global_params_raw if '=' in p}

    # =========================================================================
    # --- GOOGLE DRIVE / WINDOWS SYNC LOCK AVOIDANCE ---
    # =========================================================================
    is_windows = (os.name == 'nt')
    if is_windows:
        temp_dir = tempfile.gettempdir()
        working_path = os.path.join(temp_dir, os.path.basename(output_path))
        print(f"    --- Python: Windows detected. Writing locally to {working_path} first...")
    else:
        working_path = output_path

    if os.path.exists(working_path): os.remove(working_path)
    cdf = cdflib.cdfwrite.CDF(working_path)
    
    # =========================================================================
    # --- EDITABLE GLOBAL ATTRIBUTES SECTION ---
    # =========================================================================
    # Automatically get the filename without the '.cdf' for Logical_file_id
    file_name_no_ext = os.path.splitext(os.path.basename(output_path))[0]

    all_global_attrs = {
        'Project':          {0: 'PSP>Parker Solar Probe'},
        'Source_name':      {0: 'PSP_FLD>Parker Solar Probe FIELDS'},
        'Discipline':       {0: 'Space Physics>Interplanetary Studies'},
        'Data_type':        {0: 'L4>Level 4 Data'},
        'Descriptor':       {0: 'WAVEANALYSIS>Wave Analysis'},
        'Data_version':     {0: 'v1.3'},
        'Logical_file_id':  {0: file_name_no_ext},
        'Logical_source':   {0: 'PSP_WaveAnalysis'},
        'PI_name':          {0: 'S. Bale (UC Berkeley)'}, # Edit this
        'PI_affiliation':   {0: 'UC Berkeley'},          # Edit this
        'Instrument_type':  {0: 'Magnetic Fields (space)'},
        'TEXT':             {0: 'PSP WaveAnalysis Pipeline (Autoplot + Python)'},
        'Generated_by':     {0: 'PSP_WaveAnalysis.jy'},
        **input_params  # <--- This automatically merges your Jython parameters!
    }
    
    cdf.write_globalattrs(all_global_attrs)
    # =========================================================================
    

    # --- PASS 1: Find True Time Lengths ---
    time_lengths = {}
    for line in lines[2:]:
        parts = line.strip().split('|')
        if len(parts) < 5: continue
        name, bin_path, units, is_time_str, dims_str = parts[0:5]
        if is_time_str == '1' and os.path.exists(bin_path):
            data = np.fromfile(bin_path, dtype='<f8')
            time_lengths[name] = data.size

    # --- PASS 2: Variable Processing Loop ---
    for line in lines[2:]:
        parts = line.strip().split('|')
        if len(parts) < 5: continue
        
        name, bin_path, units, is_time_str, dims_str = parts[0:5]
        extra_attrs_str = parts[5] if len(parts) > 5 else ""
        
        is_time = (is_time_str == '1')
        is_freq = (name == 'Frequencies')
        dims = [int(d) for d in dims_str.split(',')] if dims_str else []

        if not os.path.exists(bin_path): continue
        data = np.fromfile(bin_path, dtype='<f8')
        
        # --- THE MAGIC COLUMN EXTRACTOR ---
        if not (is_time or is_freq):
            depend_0 = 'FFT_time' if len(dims) >= 1 else 'Mag_time'
            true_n = time_lengths.get(depend_0, 1)
            
            # Math out the extra columns Autoplot injected
            elements_per_record = data.size // true_n
            points_per_record = dims[0] if dims else 1
            columns = elements_per_record // points_per_record
            
            # Reshape to [Time, Freq, Columns]
            data = data.reshape(true_n, points_per_record, columns)
            
            if columns == 2:    # [Time, Value]
                data = data[:, :, 1]
            elif columns == 3:  # [Time, Freq, Value]
                data = data[:, :, 2]
            elif columns == 6:  # [Time, Freq, Real, Imag, Mag, Phase]
                data = data[:, :, 2:4] # Grab Real and Imag
            
            # Collapse dummy frequency axis for 1D time series (Bn, Bp, Bq)
            if not dims:
                data = data.reshape(true_n)
        elif is_freq:
            data = data.reshape(dims)

        data = np.ascontiguousarray(data)

        # --- DATA TYPE ALIGNMENT ---
        if is_time:
            tt2000_fill = -9223372036854775808
            bad_mask = np.isnan(data) | (data < -2e18) | (data > 2e18)
            
            valid_data = data[~bad_mask]
            if len(valid_data) > 0:
                sample = np.nanmedian(valid_data)
                
                if sample > 1e17:
                    # 1. NATIVE TT2000
                    # The data is already perfect! Do absolutely nothing.
                    tt2000_epochs = valid_data
                    
                elif sample > 1e14:
                    # 2. AUTOPLOT us2000 
                    # Microseconds since 2000-01-01. Let cdflib safely apply leap seconds.
                    unix_seconds = (valid_data / 1000000.0) + 946684800.0
                    dt_list = []
                    for ts in unix_seconds:
                        try:
                            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                            dt_list.append([dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, int(dt.microsecond / 1000), int(dt.microsecond % 1000), 0])
                        except Exception:
                            dt_list.append([2000, 1, 1, 12, 0, 0, 0, 0, 0])
                    tt2000_epochs = cdflib.cdfepoch.compute_tt2000(dt_list)
                    
                else:
                    # 3. CORRUPTED OR RELATIVE
                    tt2000_epochs = valid_data
            else:
                tt2000_epochs = np.array([])
                
            # Rebuild array safely
            final_data = np.full(data.shape, tt2000_fill, dtype=np.int64)
            final_data[~bad_mask] = tt2000_epochs
            data = final_data
            
            var_type = 33  # CDF_TIME_TT2000
            fill_val = tt2000_fill
            units = 'ns'
        else:
            data = data.astype(np.float64)
            var_type = 45 
            fill_val = -1e31
            data[~np.isfinite(data)] = fill_val

        var_attrs = {
            'CATDESC':  name, 
            'FIELDNAM': name, # FIELDNAM is standard for CDFs
            'UNITS':    units, 
            'FILLVAL':  fill_val,
            'VAR_TYPE': 'support_data' if (is_time or is_freq) else 'data',
            'FORMAT':   'I22' if is_time else 'E12.4'
        }

        if not (is_time or is_freq):
            if len(dims) >= 1:
                var_attrs['DEPEND_0'] = 'FFT_time'
                var_attrs['DEPEND_1'] = 'Frequencies'
            else:
                var_attrs['DEPEND_0'] = 'Mag_time'
        
        # --- Inject Jython Attributes ---
        if extra_attrs_str:
            for pair in extra_attrs_str.split(';'):
                if '=' in pair:
                    key, val = pair.split('=', 1)
                    
                    # Ensure ISTP limits are stored as numbers, not strings
                    if key in ['VALIDMIN', 'VALIDMAX']:
                        if not is_time: # BLOCK old microsecond limits from corrupting TT2000
                            try:
                                var_attrs[key] = float(val)
                            except ValueError:
                                pass # Skip if somehow un-parseable
                    else:
                        var_attrs[key] = val
        
        # Add Dimensional ISTP Attributes
        if not (is_time or is_freq):
            if len(dims) >= 1: # 2D or 3D Data
                var_attrs['DEPEND_0'] = 'FFT_time'
                var_attrs['DEPEND_1'] = 'Frequencies'
                var_attrs['DISPLAY_TYPE'] = 'spectrogram'
            else: # 1D Data
                var_attrs['DEPEND_0'] = 'Mag_time'
                var_attrs['DISPLAY_TYPE'] = 'time_series'
                
            # Fallback VALIDMIN/VALIDMAX only if Jython didn't provide them
            if 'VALIDMIN' not in var_attrs: var_attrs['VALIDMIN'] = float(-1e30)
            if 'VALIDMAX' not in var_attrs: var_attrs['VALIDMAX'] = float(1e30)
            
        elif is_freq:
            var_attrs['VALIDMIN'] = 0.0
            var_attrs['VALIDMAX'] = 1000.0 # Or whatever your max frequency is
            var_attrs['SCALETYP'] = 'log'   # Useful if frequencies are logarithmically spaced

        var_spec = {
            'Variable':     name, 
            'Data_Type':    var_type, 
            'Num_Elements': 1,
            'Rec_Vary':     not is_freq, 
            'Dim_Sizes':    dims,
            'Data':         data 
        }

        print(f"    --- Python: Writing {name} ({data.shape[0]} records)...")
        cdf.write_var(var_spec, var_attrs=var_attrs, var_data=data)
        os.remove(bin_path)

    cdf.close()
    
    # --- Move from temp to final destination if on Windows ---
    if is_windows:
        print(f"    --- Python: Moving finalized CDF to {output_path}...")
        # shutil.move handles the transfer and deletes the temp file automatically
        shutil.move(working_path, output_path)

    with open(done_flag_path, 'w') as f: f.write('done')
    print(f"    --- Python: Successfully finalized {output_path}")

if __name__ == "__main__":
    run_stitcher(sys.argv[1])
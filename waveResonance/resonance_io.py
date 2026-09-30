import numpy as np
import pandas as pd
import cdflib
import tempfile
import shutil
import os
import platform
from contextlib import contextmanager
from scipy.interpolate import interp1d
import sys
from datetime import datetime, timezone

# Look up one level to find config file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import config

CUSTOM_TMP = os.path.join(config.get_tmpMirror(), 'waveResonance')

# --- OS ROUTER LOGIC ---
if platform.system() == 'Windows':
    ACTIVE_TMP = None 
    io_message = "(via Local Windows Temp)"
else:
    ACTIVE_TMP = CUSTOM_TMP
    os.makedirs(ACTIVE_TMP, exist_ok=True)
    io_message = "(via Linux RAM disk)"

@contextmanager
def smart_cdf(filepath):
    """Safely copies a CDF to a local temp dir before reading to bypass network drive latency."""
    fd, temp_path = tempfile.mkstemp(suffix='.cdf', dir=ACTIVE_TMP)
    os.close(fd) 
    try:
        shutil.copyfile(filepath, temp_path)
        with cdflib.CDF(temp_path) as cdf:
            yield cdf
    finally:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass

def read_2d_waveResonance_cdf(cdf_path):
    """Reads an existing v2.X 2D WaveResonance CDF file and extracts 2D arrays for 1D reprocessing."""
    d_out = {}
    physics_out = {}
    
    with smart_cdf(cdf_path) as cdf:
        d_out['Time']         = cdflib.cdfepoch.unixtime(cdf.varget('Epoch'))
        d_out['f_sc']         = cdf.varget('frequencies')

        # Fallbacks for background fields (re-hydrated downstream by fetch_plasma_data if missing)
        zvars = cdf.cdf_info().zVariables
        d_out['B0']           = cdf.varget('B0_SC') if 'B0_SC' in zvars else np.full((len(d_out['Time']), 3), np.nan)
        d_out['N_p']          = cdf.varget('n_p') if 'n_p' in zvars else np.full(len(d_out['Time']), np.nan)
        d_out['N_alpha']      = cdf.varget('n_alpha') if 'n_alpha' in zvars else np.full(len(d_out['Time']), np.nan)
        d_out['V_sw']         = cdf.varget('v_sw_RTN') if 'v_sw_RTN' in zvars else np.full((len(d_out['Time']), 3), np.nan)
        d_out['Density_Flag'] = cdf.varget('density_flag') if 'density_flag' in zvars else np.full(len(d_out['Time']), 0, dtype=np.int8)

        # 2D Physics Spectrograms
        physics_out['f_plasma']   = cdf.varget('plasma_frequency')
        physics_out['V_ph']       = cdf.varget('phase_velocity')
        
        k_bundled                 = cdf.varget('wavenumbers')
        physics_out['k_parallel'] = k_bundled[..., 0]
        physics_out['k_perp']     = k_bundled[..., 1]
        
        resp_bundled              = cdf.varget('resonances_proton')
        physics_out['E_res_p']        = resp_bundled[..., 0]
        physics_out['E_res_landau_p'] = resp_bundled[..., 1]
        physics_out['E_res_anom_p']   = resp_bundled[..., 2]
        
        resa_bundled              = cdf.varget('resonances_alpha')
        physics_out['E_res_alpha']      = resa_bundled[..., 0]
        physics_out['E_res_anom_alpha'] = resa_bundled[..., 1]
        
        masks_pol                 = cdf.varget('masks_polarization')
        physics_out['mask_LH']    = masks_pol[..., 0].astype(bool)
        physics_out['mask_RH']    = masks_pol[..., 1].astype(bool)
        
    return d_out, physics_out

@contextmanager
def managed_waveResonance_cdf(output_path, v_out, source_wave_file, source_mag_file, data_type, logical_source):
    """Context manager to handle boilerplate CDF creation, global attributes, and OS file locks."""
    is_windows = (platform.system() == 'Windows')
    if is_windows:
        working_path = os.path.join(tempfile.gettempdir(), os.path.basename(output_path))
    else:
        working_path = output_path

    if os.path.exists(working_path): os.remove(working_path)
    os.makedirs(os.path.dirname(working_path), exist_ok=True)
    
    cdf = cdflib.cdfwrite.CDF(working_path)
    
    file_name_no_ext = os.path.splitext(os.path.basename(output_path))[0]
    cdf.write_globalattrs({
        'Project':          {0: 'PSP>Parker Solar Probe'},
        'Source_name':      {0: 'PSP_FLD_SWEAP>Parker Solar Probe FIELDS and SWEAP'},
        'Discipline':       {0: 'Space Physics>Interplanetary Studies'},
        'Data_type':        {0: data_type},
        'Descriptor':       {0: 'WAVERESONANCE>Wave Resonance and Kinematics'},
        'Data_version':     {0: f'v{v_out}'},
        'Logical_file_id':  {0: file_name_no_ext},
        'Logical_source':   {0: logical_source},
        'PI_name':          {0: 'S. Bale (UCB FIELDS), M. Stevens (CfA SWEAP)'}, 
        'PI_affiliation':   {0: 'UC Berkeley, Harvard-Smithsonian CfA'},  
        'Instrument_type':  {0: 'Fluxgate Magnetometer, iESA and Faraday Cup'},
        'TEXT':             {0: 'PSP Wave Resonance Pipeline. SPC Data: K. Paulson, SPANi: R. Livi, 15m merged: S. Badman.'},
        'Parents':          {0: f"WaveAnalysis: {source_wave_file} | Mag: {source_mag_file}"},
        'Generated_by':     {0: 'PSP_WaveResonance Pipeline (Python 3)'}
    })

    def add_var(name, data, var_type, depends, units, label, dims, extra_attrs=None):
        data = np.ascontiguousarray(data)
        
        if var_type == 45:       
            var_type = 44        
            data = data.astype(np.float32)
            
        var_attrs = {
            'CATDESC':  name, 'FIELDNAM': name, 'UNITS': units, 'LABLAXIS': label,
            'FILLVAL':  -9223372036854775808 if var_type == 33 else (-128 if var_type == 41 else -1e31),
            'VAR_TYPE': 'support_data' if 'time' in name.lower() or 'freq' in name.lower() else 'data',
            'FORMAT':   'I22' if var_type == 33 else ('I4' if var_type == 41 else 'E12.4')
        }
        
        if len(depends) == 1:
            var_attrs['DEPEND_0'] = depends[0]
            if var_type != 33 and 'freq' not in name.lower():
                var_attrs['DISPLAY_TYPE'] = 'time_series'
        elif len(depends) == 2:
            var_attrs['DEPEND_0'] = depends[0]
            var_attrs['DEPEND_1'] = depends[1]
            if 'frequencies' in depends:
                var_attrs['DISPLAY_TYPE'] = 'spectrogram'
            else:
                var_attrs['DISPLAY_TYPE'] = 'time_series'
        elif len(depends) == 3:
            var_attrs['DEPEND_0'] = depends[0]
            var_attrs['DEPEND_1'] = depends[1]
            var_attrs['DEPEND_2'] = depends[2]
            if 'frequencies' in depends:
                var_attrs['DISPLAY_TYPE'] = 'spectrogram'
            else:
                var_attrs['DISPLAY_TYPE'] = 'time_series'

        if extra_attrs: var_attrs.update(extra_attrs)
        if var_type not in (33, 41): data[~np.isfinite(data)] = var_attrs['FILLVAL']

        cdf.write_var({
            'Variable': name, 'Data_Type': var_type, 'Num_Elements': 1,
            'Rec_Vary': (name != 'frequencies'), 'Dim_Sizes': dims, 'Data': data 
        }, var_attrs=var_attrs, var_data=data)

    try:
        yield cdf, add_var
    finally:
        cdf.close()
        if is_windows:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.move(working_path, output_path)

def load_smart_csv(csv_path):
    """Safely copies, parses, and cleans the SWEAP CSV."""
    fd, temp_path = tempfile.mkstemp(suffix='.csv', dir=ACTIVE_TMP)
    os.close(fd) 
    try:
        shutil.copyfile(csv_path, temp_path)
        df = pd.read_csv(temp_path, index_col=0)
        df.columns = df.columns.str.strip()
        if 'Times' in df.columns:
            df['Times'] = pd.to_datetime(df['Times'], utc=True, errors='coerce').dt.tz_localize(None)
        for col in df.columns:
            if col != 'Times':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df.loc[df[col] < -1e30, col] = np.nan
        return df
    finally:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass

def fetch_wave_data(wave_file):
    """Loads L3 WaveAnalysis spectral, polarization, and antenna metadata."""
    data = {}
    print(f"    -> Loading Wave Analysis Data {io_message}...")
    with smart_cdf(wave_file) as cdf_w:
        data['f_sc']          = cdf_w.varget('frequencies')
        data['Power_E']       = cdf_w.varget('wave_power_E')
        data['Power_B']       = cdf_w.varget('wave_power_B')
        data['Power_B_para']  = cdf_w.varget('B_power_para')
        
        S_vec                 = cdf_w.varget('S_fieldAligned')
        data['S_n']           = S_vec[..., 0] 
        
        k_vec                 = cdf_w.varget('k_hat_fieldAligned')
        data['k_n']           = k_vec[..., 0]
        data['k_p']           = k_vec[..., 1]
        data['k_q']           = k_vec[..., 2]

        data['coherency_B']   = cdf_w.varget('coherency')
        data['ellipticity_B'] = cdf_w.varget('ellipticity')
        
        try:
            g_attrs = cdf_w.globalattrs()
            if 'Baseline_Antenna_Length' in g_attrs:
                val_raw = g_attrs['Baseline_Antenna_Length']
                if isinstance(val_raw, (dict, list)): val_raw = val_raw[0]
                val_str = str(val_raw).strip()
                data['efieldAntenna_baselineLength'] = 3.5 if 'unknown' in val_str.lower() else float(val_str.split()[0])
            else:
                data['efieldAntenna_baselineLength'] = 3.5
        except Exception:
            data['efieldAntenna_baselineLength'] = 3.5
            
        data['Time'] = cdflib.cdfepoch.unixtime(cdf_w.varget('fft_time'))

    if data['f_sc'].ndim == 1:
        data['f_sc'] = np.tile(data['f_sc'], (len(data['Time']), 1))
        
    return data

def fetch_plasma_data(target_unix_time, mag_file, csv_path, start_dt, end_dt, lfr_density_file=None):
    """Syncs background B0, solar wind velocity (V_sw), and ion densities (N_p, N_alpha)."""
    data = {}
    
    print(f"    -> Loading & Syncing Background B0 {io_message}...")
    with smart_cdf(mag_file) as cdf_m:
        var_names = cdf_m.cdf_info().zVariables
        b_var = 'psp_fld_l2_mag_SC_4_Sa_per_Cyc' if 'psp_fld_l2_mag_SC_4_Sa_per_Cyc' in var_names else 'psp_fld_l2_mag_SC_1min'
        t_var = 'epoch_mag_SC_4_Sa_per_Cyc' if 'epoch_mag_SC_4_Sa_per_Cyc' in var_names else 'epoch_mag_SC_1min'
        
        B0_raw = cdf_m.varget(b_var)
        mag_unix = cdflib.cdfepoch.unixtime(cdf_m.varget(t_var))

    B0_raw[B0_raw < -1e30] = np.nan
    f_interp_b = interp1d(mag_unix, B0_raw, axis=0, bounds_error=False, fill_value=np.nan)
    data['B0'] = f_interp_b(target_unix_time)
    
    print(f"    -> Loading and Syncing SWEAP CSV {io_message}...")
    df = load_smart_csv(csv_path)
    csv_mask = (df['Times'] >= start_dt) & (df['Times'] <= end_dt)
    df_win = df.loc[csv_mask]
    
    if df_win.empty:
        raise ValueError("CSV contains no data in the requested time range.")
    
    csv_unix = (df_win['Times'] - pd.Timestamp("1970-01-01")).dt.total_seconds().values
    
    vpr = df_win.get('Vpr-Parker', np.full(len(df_win), np.nan))
    vpt = df_win.get('Vpt-Parker', np.full(len(df_win), np.nan))
    vpn = df_win.get('Vpn-Parker', np.full(len(df_win), np.nan))
    V_sw_raw = np.column_stack((vpr, vpt, vpn))
    
    if not np.any(np.isfinite(V_sw_raw)):
        data['V_sw'] = np.full((len(target_unix_time), 3), np.nan)
    else:
        f_interp_sw = interp1d(csv_unix, V_sw_raw, axis=0, bounds_error=False, fill_value=np.nan)
        data['V_sw'] = f_interp_sw(target_unix_time)
        
    np_raw = df_win.get('Np_Parker', df_win.get('Np', np.full(len(df_win), np.nan)))
    if not np.any(np.isfinite(np_raw)): 
        data['N_p'] = np.full(len(target_unix_time), np.nan)
    else:
        f_interp_np = interp1d(csv_unix, np_raw, bounds_error=False, fill_value=np.nan)
        data['N_p'] = f_interp_np(target_unix_time)
        
    na_raw = df_win.get('Na_Parker', df_win.get('Na', np.full(len(df_win), np.nan)))
    if not np.any(np.isfinite(na_raw)): 
        data['N_alpha'] = np.full(len(target_unix_time), np.nan)
    else:
        f_interp_na = interp1d(csv_unix, na_raw, bounds_error=False, fill_value=np.nan)
        data['N_alpha'] = f_interp_na(target_unix_time)

    density_flag = np.full(len(target_unix_time), 3, dtype=np.int8)
    sweap_np_valid = np.isfinite(data['N_p'])
    sweap_na_valid = np.isfinite(data['N_alpha'])
    
    density_flag[sweap_np_valid & sweap_na_valid] = 0
    density_flag[sweap_np_valid & ~sweap_na_valid] = 1

    if lfr_density_file and os.path.exists(lfr_density_file) and np.any(np.isnan(data['N_p'])):
        print(f"      -> Filling SWEAP density gaps with LFR Mission Density {io_message}...")
        try:
            with smart_cdf(lfr_density_file) as cdf_lfr:
                lfr_unix = cdflib.cdfepoch.unixtime(cdf_lfr.varget('Epoch'))
                lfr_ne = cdf_lfr.varget('electronDensity')
                pad = 86400 * 2
                valid_mask = np.isfinite(lfr_ne) & (lfr_unix >= start_dt.timestamp() - pad) & (lfr_unix <= end_dt.timestamp() + pad)
                
                if np.any(valid_mask):
                    f_interp_lfr = interp1d(lfr_unix[valid_mask], lfr_ne[valid_mask], bounds_error=False, fill_value=np.nan)
                    lfr_interp = f_interp_lfr(target_unix_time)
                    nan_mask = np.isnan(data['N_p'])
                    data['N_p'][nan_mask] = lfr_interp[nan_mask] / 1.08
                    
                    alpha_nan_mask = np.isnan(data['N_alpha']) & nan_mask
                    data['N_alpha'][alpha_nan_mask] = 0.04 * data['N_p'][alpha_nan_mask]
                    
                    lfr_patched = nan_mask & np.isfinite(data['N_p'])
                    density_flag[lfr_patched] = 2
        except Exception:
            pass

    data['Density_Flag'] = density_flag
    return data

def fetch_and_align_data(wave_file, mag_file, csv_path, start_dt, end_dt, lfr_density_file=None):
    """Backward-compatible wrapper that loads wave and plasma data together."""
    d_wave = fetch_wave_data(wave_file)
    d_plasma = fetch_plasma_data(d_wave['Time'], mag_file, csv_path, start_dt, end_dt, lfr_density_file)
    return {**d_wave, **d_plasma}
        
def write_2d_waveResonance_cdf(output_path, d_out, physics_out, v_out, source_wave_file, source_mag_file):
    """Writes the heavy v2.X Wave Resonance 2D spectrograms directly to CDF."""
    print(f"    -> Formatting 2D Data for CDF...")
    
    unix_times = d_out['Time']
    dt_list = [[dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, int(dt.microsecond / 1000), int(dt.microsecond % 1000)*1000, 0] 
               for dt in (datetime.fromtimestamp(ts, tz=timezone.utc) for ts in unix_times)]
    tt2000_epochs = cdflib.cdfepoch.compute_tt2000(dt_list)
    
    freqs_1d = d_out['f_sc'][0, :].copy()
    freqs_1d[freqs_1d <= 0] = np.nan
    n_freqs = len(freqs_1d)

    with managed_waveResonance_cdf(output_path, v_out, source_wave_file, source_mag_file, 
                                   'L4>Level 4 Data 2D Spectrograms', 'PSP_WaveResonance_2D') as (cdf, add_var):
        
        f_min = float(np.nanmin(freqs_1d))
        f_max = float(np.nanmax(freqs_1d))
        add_var('Epoch', tt2000_epochs, 33, ['Epoch'], 'ns', 'Time', [])
        add_var('frequencies', freqs_1d, 45, [], 'Hz', 'Frequency!C', [n_freqs],
                {'SCALETYP': 'log', 'VALIDMIN': f_min, 'VALIDMAX': f_max,
                 'SCALEMIN': f_min, 'SCALEMAX': f_max})

        # --- SUPPORT INDICES & LABELS ---
        cdf.write_var({'Variable': 'index_3d', 'Data_Type': 41, 'Num_Elements': 1, 'Rec_Vary': False, 'Dim_Sizes': [3]}, var_data=np.array([1, 2, 3], dtype=np.int8))
        cdf.write_var({'Variable': 'index_2d', 'Data_Type': 41, 'Num_Elements': 1, 'Rec_Vary': False, 'Dim_Sizes': [2]}, var_data=np.array([1, 2], dtype=np.int8))
        
        cdf.write_var({'Variable': 'label_pol', 'Data_Type': 51, 'Num_Elements': 5, 'Rec_Vary': False, 'Dim_Sizes': [2]}, var_data=['LH', 'RH'])
        cdf.write_var({'Variable': 'label_dir', 'Data_Type': 51, 'Num_Elements': 5, 'Rec_Vary': False, 'Dim_Sizes': [2]}, var_data=['+Sn', '-Sn'])
        cdf.write_var({'Variable': 'label_k', 'Data_Type': 51, 'Num_Elements': 15, 'Rec_Vary': False, 'Dim_Sizes': [2]}, var_data=['k_parallel', 'k_perp'])
        cdf.write_var({'Variable': 'label_res_p', 'Data_Type': 51, 'Num_Elements': 20, 'Rec_Vary': False, 'Dim_Sizes': [3]}, var_data=['Cyclotron (n=1)', 'Landau (n=0)', 'Anomalous (n=-1)'])
        cdf.write_var({'Variable': 'label_res_alpha', 'Data_Type': 51, 'Num_Elements': 20, 'Rec_Vary': False, 'Dim_Sizes': [2]}, var_data=['Cyclotron (n=1)', 'Anomalous (n=-1)'])

        if 'L_eff_dynamic' in physics_out:
            add_var('effective_antenna_length', physics_out['L_eff_dynamic'], 45, ['Epoch', 'frequencies'], 'm', 'L_eff', [n_freqs],
                    {'SCALETYP': 'linear', 'VALIDMIN': 1.0, 'VALIDMAX': 50.0, 'TITLE': 'Dynamic Effective Antenna Length'})

        # Standard 2.x lowercase mappings
        add_var('plasma_frequency', physics_out['f_plasma'], 45, ['Epoch', 'frequencies'], 'Hz', 'Frequency!C(Plasma Frame)', [n_freqs],
                {'SCALETYP': 'linear', 'VALIDMIN': -1000.0, 'VALIDMAX': 1000.0, 'TITLE': 'Doppler-Shifted Plasma Frame Frequency'})
        add_var('phase_velocity', physics_out['V_ph'], 45, ['Epoch', 'frequencies'], 'km/s', 'V!Bph!N', [n_freqs],
                {'SCALETYP': 'log', 'VALIDMIN': 1.0, 'VALIDMAX': 3e8, 'TITLE': 'Wave Phase Velocity'})
        add_var('magnetic_compressibility', physics_out['compressibility'], 45, ['Epoch', 'frequencies'], '', 'c!BB!N', [n_freqs],
                {'SCALETYP': 'linear', 'VALIDMIN': 0.0, 'VALIDMAX': 1.0, 'TITLE': 'Magnetic Compressibility'})
        
        # Nested spectrogram directories
        k_bundled = np.stack([physics_out['k_parallel'], physics_out['k_perp']], axis=-1)
        add_var('wavenumbers', k_bundled, 45, ['Epoch', 'frequencies', 'index_2d'], 'rad/m', 'k Vectors', [n_freqs, 2],
                {'SCALETYP': 'linear', 'VALIDMIN': -1.0, 'VALIDMAX': 1.0, 'LABL_PTR_2': 'label_k'})
        
        resp_bundled = np.stack([physics_out['E_res_p'], physics_out['E_res_landau_p'], physics_out['E_res_anom_p']], axis=-1)
        add_var('resonances_proton', resp_bundled, 45, ['Epoch', 'frequencies', 'index_3d'], 'eV', 'Proton Resonances', [n_freqs, 3],
                {'SCALETYP': 'log', 'VALIDMIN': 1e-1, 'VALIDMAX': 1e8, 'LABL_PTR_2': 'label_res_p'})
        
        resa_bundled = np.stack([physics_out['E_res_alpha'], physics_out['E_res_anom_alpha']], axis=-1)
        add_var('resonances_alpha', resa_bundled, 45, ['Epoch', 'frequencies', 'index_2d'], 'eV', 'Alpha Resonances', [n_freqs, 2],
                {'SCALETYP': 'log', 'VALIDMIN': 1e-1, 'VALIDMAX': 1e8, 'LABL_PTR_2': 'label_res_alpha'})

        mask_pol_bundled = np.stack([physics_out['mask_LH'], physics_out['mask_RH']], axis=-1)
        add_var('masks_polarization', mask_pol_bundled, 41, ['Epoch', 'frequencies', 'index_2d'], '', 'Polarization Masks', [n_freqs, 2],
                {'SCALETYP': 'linear', 'VALIDMIN': 0, 'VALIDMAX': 1, 'LABL_PTR_2': 'label_pol'})
        
        mask_dir_bundled = np.stack([physics_out['mask_Sn_pos'], physics_out['mask_Sn_neg']], axis=-1)
        add_var('masks_poynting', mask_dir_bundled, 41, ['Epoch', 'frequencies', 'index_2d'], '', 'Poynting Masks', [n_freqs, 2],
                {'SCALETYP': 'linear', 'VALIDMIN': 0, 'VALIDMAX': 1, 'LABL_PTR_2': 'label_dir'})

def write_1d_waveResonance_cdf(output_path, d_out, physics_out, v_out, source_wave_file, source_mag_file):
    """Writes the reduced v3.X Wave Resonance 1D data directly to CDF with Autoplot-friendly 2D LH/RH splits."""
    print(f"    -> Formatting 1D Data for CDF...")
    
    unix_times = d_out['Time']
    dt_list = [[dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, int(dt.microsecond / 1000), int(dt.microsecond % 1000)*1000, 0] 
               for dt in (datetime.fromtimestamp(ts, tz=timezone.utc) for ts in unix_times)]
    tt2000_epochs = cdflib.cdfepoch.compute_tt2000(dt_list)

    with managed_waveResonance_cdf(output_path, v_out, source_wave_file, source_mag_file, 
                                   'L4>Level 4 Data 1D Timeseries', 'PSP_WaveResonance_1D') as (cdf, add_var):
        
        add_var('Epoch', tt2000_epochs, 33, ['Epoch'], 'ns', 'Time', [])
        
        # Spatial/Component Dimensions
        cdf.write_var({'Variable': 'index_3d_comp', 'Data_Type': 41, 'Num_Elements': 1, 'Rec_Vary': False, 'Dim_Sizes': [3]}, var_data=np.array([1, 2, 3], dtype=np.int8))
        cdf.write_var({'Variable': 'index_2d_comp', 'Data_Type': 41, 'Num_Elements': 1, 'Rec_Vary': False, 'Dim_Sizes': [2]}, var_data=np.array([1, 2], dtype=np.int8))
        cdf.write_var({'Variable': 'index_2d_pol',  'Data_Type': 41, 'Num_Elements': 1, 'Rec_Vary': False, 'Dim_Sizes': [2]}, var_data=np.array([1, 2], dtype=np.int8))

        # Labels
        cdf.write_var({'Variable': 'label_sc', 'Data_Type': 51, 'Num_Elements': 5, 'Rec_Vary': False, 'Dim_Sizes': [3]}, var_data=['B_x', 'B_y', 'B_z'])
        cdf.write_var({'Variable': 'label_rtn', 'Data_Type': 51, 'Num_Elements': 5, 'Rec_Vary': False, 'Dim_Sizes': [3]}, var_data=['V_R', 'V_T', 'V_N'])
        cdf.write_var({'Variable': 'label_pol', 'Data_Type': 51, 'Num_Elements': 5, 'Rec_Vary': False, 'Dim_Sizes': [2]}, var_data=['LH', 'RH'])
        cdf.write_var({'Variable': 'label_dir', 'Data_Type': 51, 'Num_Elements': 15, 'Rec_Vary': False, 'Dim_Sizes': [2]}, var_data=['Outward (+Sn)', 'Inward (-Sn)'])
        cdf.write_var({'Variable': 'label_k_comp', 'Data_Type': 51, 'Num_Elements': 15, 'Rec_Vary': False, 'Dim_Sizes': [2]}, var_data=['k_parallel', 'k_perp'])
        cdf.write_var({'Variable': 'label_res_p_comp', 'Data_Type': 51, 'Num_Elements': 20, 'Rec_Vary': False, 'Dim_Sizes': [3]}, var_data=['Cyclotron (n=1)', 'Landau (n=0)', 'Anomalous (n=-1)'])
        cdf.write_var({'Variable': 'label_res_alpha_comp', 'Data_Type': 51, 'Num_Elements': 20, 'Rec_Vary': False, 'Dim_Sizes': [2]}, var_data=['Cyclotron (n=1)', 'Anomalous (n=-1)'])

        # --- 1D BACKGROUND FIELDS ---
        add_var('B0_SC', d_out['B0'], 45, ['Epoch', 'index_3d_comp'], 'nT', 'B!B0!N (SC)', [3],
                {'SCALETYP': 'linear', 'VALIDMIN': -15000.0, 'VALIDMAX': 15000.0, 'LABL_PTR_1': 'label_sc', 'DISPLAY_TYPE': 'time_series'})
        add_var('v_sw_RTN', d_out['V_sw'], 45, ['Epoch', 'index_3d_comp'], 'km/s', 'V!Bsw!N (RTN)', [3],
                {'SCALETYP': 'linear', 'VALIDMIN': -2000.0, 'VALIDMAX': 2000.0, 'LABL_PTR_1': 'label_rtn', 'DISPLAY_TYPE': 'time_series'})
        add_var('n_p', d_out['N_p'], 45, ['Epoch'], 'cm^-3', 'Proton!CDensity', [],
                {'SCALETYP': 'log', 'VALIDMIN': 1e-3, 'VALIDMAX': 1e5})
        add_var('n_alpha', d_out['N_alpha'], 45, ['Epoch'], 'cm^-3', 'Alpha!CDensity', [],
                {'SCALETYP': 'log', 'VALIDMIN': 1e-4, 'VALIDMAX': 1e5})
        add_var('density_flag', d_out['Density_Flag'], 41, ['Epoch'], '', 'Density!CSource Flag', [],
                {'SCALETYP': 'linear', 'VALIDMIN': 0, 'VALIDMAX': 3, 'CATDESC': 'Density Source'})
        add_var('v_alfven', physics_out['V_A'], 45, ['Epoch'], 'km/s', 'Alfven!CVelocity', [],
                {'SCALETYP': 'log', 'VALIDMIN': 1e0, 'VALIDMAX': 1e7})

        # --- BUNDLED TIMESERIES (2D Integrations: Epoch x LH/RH or Direction) ---
        power_bundled = np.stack([physics_out['int_LH'], physics_out['int_RH']], axis=-1)
        add_var('integrated_wave_power', power_bundled, 45, ['Epoch', 'index_2d_pol'], 'nT^2', 'Wave Power', [2],
                {'SCALETYP': 'log', 'VALIDMIN': 1e-10, 'VALIDMAX': 1e5, 'LABL_PTR_1': 'label_pol', 'TITLE': 'Integrated Wave Power (LH/RH)', 'DISPLAY_TYPE': 'time_series'})
        
        poynting_bundled = np.stack([physics_out['int_Sn_pos'], physics_out['int_Sn_neg']], axis=-1)
        add_var('integrated_poynting_Sn', poynting_bundled, 45, ['Epoch', 'index_2d_pol'], 'nW/m^2', 'S_n', [2],
                {'SCALETYP': 'log', 'VALIDMIN': 1e-10, 'VALIDMAX': 1e5, 'LABL_PTR_1': 'label_dir', 'TITLE': 'Integrated Poynting Flux (+Sn/-Sn)', 'DISPLAY_TYPE': 'time_series'})

        # --- UNBUNDLED MULTI-COMPONENT AVERAGES (Split into 2D LH / RH Variables) ---
        
        # 1. Wavenumbers [Epoch, index_2d_comp]
        k_avg_LH = np.stack([physics_out['avg_k_parallel_LH'], physics_out['avg_k_perp_LH']], axis=-1)
        add_var('wavenumbers_avg_LH', k_avg_LH, 45, ['Epoch', 'index_2d_comp'], 'rad/m', 'Avg k (LH)', [2],
                {'LABL_PTR_1': 'label_k_comp', 'TITLE': 'Average Wavenumbers (LH)', 'DISPLAY_TYPE': 'time_series'})

        k_avg_RH = np.stack([physics_out['avg_k_parallel_RH'], physics_out['avg_k_perp_RH']], axis=-1)
        add_var('wavenumbers_avg_RH', k_avg_RH, 45, ['Epoch', 'index_2d_comp'], 'rad/m', 'Avg k (RH)', [2],
                {'LABL_PTR_1': 'label_k_comp', 'TITLE': 'Average Wavenumbers (RH)', 'DISPLAY_TYPE': 'time_series'})

        # 2. Proton Resonances [Epoch, index_3d_comp]
        res_p_LH = np.stack([physics_out['avg_E_res_p_LH'], physics_out['avg_E_res_landau_p_LH'], physics_out['avg_E_res_anom_p_LH']], axis=-1)
        add_var('resonances_proton_avg_LH', res_p_LH, 45, ['Epoch', 'index_3d_comp'], 'eV', 'Avg Proton Res. (LH)', [3],
                {'SCALETYP': 'log', 'VALIDMIN': 1e-1, 'VALIDMAX': 1e8, 'LABL_PTR_1': 'label_res_p_comp', 'TITLE': 'Average Proton Resonant Energies (LH)', 'DISPLAY_TYPE': 'time_series'})

        res_p_RH = np.stack([physics_out['avg_E_res_p_RH'], physics_out['avg_E_res_landau_p_RH'], physics_out['avg_E_res_anom_p_RH']], axis=-1)
        add_var('resonances_proton_avg_RH', res_p_RH, 45, ['Epoch', 'index_3d_comp'], 'eV', 'Avg Proton Res. (RH)', [3],
                {'SCALETYP': 'log', 'VALIDMIN': 1e-1, 'VALIDMAX': 1e8, 'LABL_PTR_1': 'label_res_p_comp', 'TITLE': 'Average Proton Resonant Energies (RH)', 'DISPLAY_TYPE': 'time_series'})

        # 3. Alpha Resonances [Epoch, index_2d_comp]
        res_a_LH = np.stack([physics_out['avg_E_res_alpha_LH'], physics_out['avg_E_res_anom_alpha_LH']], axis=-1)
        add_var('resonances_alpha_avg_LH', res_a_LH, 45, ['Epoch', 'index_2d_comp'], 'eV', 'Avg Alpha Res. (LH)', [2],
                {'SCALETYP': 'log', 'VALIDMIN': 1e-1, 'VALIDMAX': 1e8, 'LABL_PTR_1': 'label_res_alpha_comp', 'TITLE': 'Average Alpha Resonant Energies (LH)', 'DISPLAY_TYPE': 'time_series'})

        res_a_RH = np.stack([physics_out['avg_E_res_alpha_RH'], physics_out['avg_E_res_anom_alpha_RH']], axis=-1)
        add_var('resonances_alpha_avg_RH', res_a_RH, 45, ['Epoch', 'index_2d_comp'], 'eV', 'Avg Alpha Res. (RH)', [2],
                {'SCALETYP': 'log', 'VALIDMIN': 1e-1, 'VALIDMAX': 1e8, 'LABL_PTR_1': 'label_res_alpha_comp', 'TITLE': 'Average Alpha Resonant Energies (RH)', 'DISPLAY_TYPE': 'time_series'})

        # 4. Phase Velocity & Plasma Frequency
        v_ph_avg_data = np.stack([physics_out['avg_V_ph_LH'], physics_out['avg_V_ph_RH']], axis=-1)
        add_var('phase_velocity_avg', v_ph_avg_data, 45, ['Epoch', 'index_2d_pol'], 'km/s', 'Avg V!Bph!N', [2],
                {'SCALETYP': 'log', 'VALIDMIN': 1.0, 'VALIDMAX': 3e8, 'LABL_PTR_1': 'label_pol', 'TITLE': 'Average Phase Velocity (LH/RH)', 'DISPLAY_TYPE': 'time_series'})

        f_plasma_avg_data = np.stack([physics_out['avg_f_plasma_LH'], physics_out['avg_f_plasma_RH']], axis=-1)
        add_var('plasma_frequency_avg', f_plasma_avg_data, 45, ['Epoch', 'index_2d_pol'], 'Hz', 'Avg Plasma!CFrequency', [2],
                {'SCALETYP': 'linear', 'VALIDMIN': -1000.0, 'VALIDMAX': 1000.0, 'LABL_PTR_1': 'label_pol', 'TITLE': 'Average Plasma Frame Frequency (LH/RH)', 'DISPLAY_TYPE': 'time_series'})
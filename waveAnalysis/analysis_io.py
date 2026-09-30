import numpy as np
import cdflib
import tempfile
import shutil
import os
import platform
from contextlib import contextmanager
from datetime import datetime, timezone

@contextmanager
def managed_waveAnalysis_cdf(output_path, v_out, source_mag_file, source_efield_file, antenna_length='Unknown'):
	"""
	Context manager to handle boilerplate CDF creation, global attributes, 
	and OS file locks for the v1.4 Wave Analysis pipeline.
	"""
	is_windows = (platform.system() == 'Windows')
	if is_windows:
		working_path = os.path.join(tempfile.gettempdir(), os.path.basename(output_path))
	else:
		working_path = output_path

	if os.path.exists(working_path): 
		os.remove(working_path)
	os.makedirs(os.path.dirname(working_path), exist_ok=True)
	
	cdf = cdflib.cdfwrite.CDF(working_path)
	
	# Determine metadata string dynamically based on type
	if isinstance(antenna_length, (int, float)):
		antenna_meta_string = f"{antenna_length} meters"
	else:
		antenna_meta_string = str(antenna_length)

	# --- GLOBAL ATTRIBUTES ---
	file_name_no_ext = os.path.splitext(os.path.basename(output_path))[0]
	cdf.write_globalattrs({
		'Project':                  {0: 'PSP>Parker Solar Probe'},
		'Source_name':              {0: 'PSP_FLD>Parker Solar Probe FIELDS'},
		'Discipline':               {0: 'Space Physics>Interplanetary Studies'},
		'Data_type':                {0: 'L3>Level 3 Data'},
		'Descriptor':               {0: 'WAVEANALYSIS>Observational Wave Properties'},
		'Data_version':             {0: f'v{v_out}'},
		'Logical_file_id':          {0: file_name_no_ext},
		'Logical_source':           {0: 'PSP_WaveAnalysis'},
		'PI_name':                  {0: 'S. Bale (UCB FIELDS)'}, 
		'PI_affiliation':           {0: 'UC Berkeley'},  
		'Instrument_type':          {0: 'Fluxgate Magnetometer, SCM, and Electric Field'},
		'TEXT':                     {0: 'PSP Wave Analysis Pipeline (v1.4 Python Rewrite). Polarization: Means 1972.'},
		'Parents':                  {0: f"Mag: {source_mag_file} | E-field: {source_efield_file}"},
		'Generated_by':             {0: 'PSP_WaveAnalysis Pipeline (Python 3)'},
		'Baseline_Antenna_Length':  {0: antenna_meta_string}
	})

	# --- VARIABLE DEFINITION HELPER ---
	def add_var(name, data, var_type, depends, units, label, dims, extra_attrs=None):
		data = np.ascontiguousarray(data)
		
		# --- THE SILVER BULLET: Automatic Float32 Downcast ---
		if var_type == 45:       # If it's asking for CDF_DOUBLE (8-byte)
			var_type = 44        # Switch it to CDF_FLOAT (4-byte)
			data = data.astype(np.float32)
		
		# 33 = CDF_TIME_TT2000, 41 = CDF_BYTE/INT1, 44 = CDF_FLOAT
		var_attrs = {
			'CATDESC':  name, 
			'FIELDNAM': name, 
			'UNITS':    units, 
			'LABLAXIS': label,
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
			var_attrs['DISPLAY_TYPE'] = 'spectrogram'
		elif len(depends) == 3:
			var_attrs['DEPEND_0'] = depends[0]
			var_attrs['DEPEND_1'] = depends[1]
			var_attrs['DEPEND_2'] = depends[2]
			var_attrs['DISPLAY_TYPE'] = 'spectrogram'

		if extra_attrs: 
			var_attrs.update(extra_attrs)
			
		if var_type not in (33, 41): 
			data[~np.isfinite(data)] = var_attrs['FILLVAL']

		cdf.write_var({
			'Variable': name, 'Data_Type': var_type, 'Num_Elements': 1,
			'Rec_Vary': ('freq' not in name.lower()), 'Dim_Sizes': dims, 'Data': data 
		}, var_attrs=var_attrs, var_data=data)

	# --- YIELD TO WRITER ---
	try:
		yield cdf, add_var
	finally:
		cdf.close()
		if is_windows:
			os.makedirs(os.path.dirname(output_path), exist_ok=True)
			shutil.move(working_path, output_path)

def write_waveAnalysis_cdf(output_path, data, v_out, source_mag_file, source_efield_file, logy=True, verbose=False):
	"""
	Writes the optimized v1.4 Wave Analysis data directly to CDF.
	Expects 'data' to be a dictionary containing all physical arrays and time vectors.
	"""
	if verbose:
		print(f"    -> Formatting v1.4 WaveAnalysis Data for CDF...")
	
	# 1. Convert Time to TT2000 (Mag Time and FFT Time)
	def unix_to_tt2000(unix_array):
		dt_list = [[dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, 
					int(dt.microsecond / 1000), int(dt.microsecond % 1000)*1000, 0] 
				   for dt in (datetime.fromtimestamp(ts, tz=timezone.utc) for ts in unix_array)]
		return cdflib.cdfepoch.compute_tt2000(dt_list)

	mag_tt2000 = unix_to_tt2000(data['Mag_time'])
	fft_tt2000 = unix_to_tt2000(data['FFT_time'])
	
	freqs_1d = data['Frequencies']
	n_freqs = len(freqs_1d)

	# Determine scale type string dynamically based on pipeline configuration
	scale_type = 'log' if logy else 'linear'
	
	# Extract the antenna length safely
	antenna_len = data.get('Antenna_Length', 'Unknown')

	with managed_waveAnalysis_cdf(output_path, v_out, source_mag_file, source_efield_file, antenna_length=antenna_len) as (cdf, add_var):
		
		# --- DEFINE VECTOR COMPONENT LABELS (CDF_CHAR = 51) ---
		label_sets = {
			'label_B_FA': ['B_n', 'B_p', 'B_q'],
			'label_E_FA': ['E_n', 'E_p', 'E_q'],
			'label_S_vec': ['S_n', 'S_p', 'S_q'],
			'label_k_vec': ['k_n', 'k_p', 'k_q']
		}
		
		for name, strings in label_sets.items():
			cdf.write_var({
				'Variable': name, 'Data_Type': 51, 'Num_Elements': 3, 
				'Rec_Vary': False, 'Dim_Sizes': [3]
			}, var_data=strings)
			
		# --- DEFINE SPATIAL COMPONENT INDEX (For Autoplot Folders) ---
		cdf.write_var({
			'Variable': 'comp_index', 'Data_Type': 41, 'Num_Elements': 1, 
			'Rec_Vary': False, 'Dim_Sizes': [3]
		}, 
		var_attrs={'CATDESC': 'Spatial Component Index', 'FIELDNAM': 'Component Index', 'FORMAT': 'I2'},
		var_data=np.array([1, 2, 3], dtype=np.int8))
		
		# --- 1. SUPPORT DATA (Axes) ---
		f_min = float(np.nanmin(freqs_1d[freqs_1d > 0]))
		f_max = float(np.nanmax(freqs_1d))
		add_var('mag_time', mag_tt2000, 33, ['mag_time'], 'ns', 'Time (High Cadence)', [])
		add_var('fft_time', fft_tt2000, 33, ['fft_time'], 'ns', 'Time (FFT Windows)', [])
		add_var('frequencies', freqs_1d, 45, [], 'Hz', 'Frequency!C', [n_freqs],
				{'SCALETYP': scale_type, 'VALIDMIN': 0.01, 'VALIDMAX': 1000.0,
				 'SCALEMIN': f_min, 'SCALEMAX': f_max})
		# New support flag tracking background field angle relative to the physical whip plane
		if 'Whip_Tilt_Angle' in data:
			add_var('whip_tilt_angle', data['Whip_Tilt_Angle'], 45, ['fft_time'], 'deg', 'B0_Whip_Tilt', [],
					{'SCALETYP': 'linear', 'VALIDMIN': 0.0, 'VALIDMAX': 90.0, 
					 'TITLE': 'Angle of B0 out of the Spacecraft X-Y Antenna Plane',
					 'VAR_TYPE': 'support_data'})

		# 1. 1D Field-Aligned Vectors [Time, 3]
		add_var('B_fieldAligned', data['B_FA_vector'], 45, ['mag_time', 'comp_index'], 'nT', 'B!BfieldAligned!N!C!C', [3], 
				{'TITLE': 'Field-Aligned Magnetic Field', 'LABL_PTR_1': 'label_B_FA', 'DISPLAY_TYPE': 'time_series'})
				
		if data['E_FA_vector'] is not None:
			add_var('E_fieldAligned', data['E_FA_vector'], 45, ['mag_time', 'comp_index'], 'mV/m', 'E!BfieldAligned!N!C!C', [3], 
					{'TITLE': 'Field-Aligned Electric Field', 'LABL_PTR_1': 'label_E_FA', 'DISPLAY_TYPE': 'time_series'})

		# 2. 2D k-vectors (Re-bundled)
		if 'k_vector' in data:
			add_var('k_hat_fieldAligned', data['k_vector'], 45, ['fft_time', 'frequencies', 'comp_index'], '', 'k_hat', [n_freqs, 3], 
					{'TITLE': 'Wave Vector k', 'LABL_PTR_2': 'label_k_vec'})

		# --- 3. WAVE GEOMETRY (2D Spectrograms) ---
		add_var('ellipticity', data['Ellipticity'], 45, ['fft_time', 'frequencies'], '', 'Ellipticity', [n_freqs],
				{'SCALETYP': 'linear', 'VALIDMIN': -1.0, 'VALIDMAX': 1.0, 'TITLE': 'Magnetic Ellipticity'})
		add_var('wave_normal', data['Angle_Normal'], 45, ['fft_time', 'frequencies'], 'deg', 'Wave Normal!C', [n_freqs],
				{'SCALETYP': 'linear', 'VALIDMIN': 0.0, 'VALIDMAX': 90.0, 'TITLE': 'Wave Normal Angle'})
		add_var('coherency', data['Coherency'], 45, ['fft_time', 'frequencies'], '', 'Coherency', [n_freqs],
				{'SCALETYP': 'linear', 'VALIDMIN': 0.0, 'VALIDMAX': 1.0, 'TITLE': 'Magnetic Coherency'})
				
		# --- 4. WAVE POWER (2D Spectrograms) ---
		add_var('B_power_para', data['Power_compressional'], 45, ['fft_time', 'frequencies'], 'nT^2/Hz', 'PSD B_||!C', [n_freqs],
				{'SCALETYP': 'log', 'VALIDMIN': 1e-10, 'VALIDMAX': 1e5, 'TITLE': 'Compressional Magnetic Power'})
		add_var('B_power_perp', data['Power_perp'], 45, ['fft_time', 'frequencies'], 'nT^2/Hz', 'PSD B_perp!C', [n_freqs],
				{'SCALETYP': 'log', 'VALIDMIN': 1e-10, 'VALIDMAX': 1e5, 'TITLE': 'Transverse Magnetic Power'})
		add_var('wave_power_B', data['Wave_Power'], 45, ['fft_time', 'frequencies'], 'nT^2/Hz', 'Wave Power (B)!C', [n_freqs],
				{'SCALETYP': 'log', 'VALIDMIN': 1e-10, 'VALIDMAX': 1e5, 'TITLE': 'Magnetic Wave Power (Perp to k)'})
		
		if 'E_Power_compressional' in data:
			add_var('E_power_para', data['E_Power_compressional'], 45, ['fft_time', 'frequencies'], 'mV^2/m^2/Hz', 'PSD E_||!C', [n_freqs],
					{'SCALETYP': 'log', 'VALIDMIN': 1e-12, 'VALIDMAX': 1e5, 'TITLE': 'Compressional Electric Power'})
			add_var('E_power_perp', data['E_Power_perp'], 45, ['fft_time', 'frequencies'], 'mV^2/m^2/Hz', 'PSD E_perp!C', [n_freqs],
					{'SCALETYP': 'log', 'VALIDMIN': 1e-12, 'VALIDMAX': 1e5, 'TITLE': 'Transverse Electric Power'})
			add_var('wave_power_E', data['E_Wave_Power'], 45, ['fft_time', 'frequencies'], 'mV^2/m^2/Hz', 'Wave Power (E)!C', [n_freqs],
					{'SCALETYP': 'log', 'VALIDMIN': 1e-12, 'VALIDMAX': 1e5, 'TITLE': 'Electric Wave Power (Perp to k)'})

		# --- 5. POYNTING FLUX (Re-bundled) ---
		if 'S_mag' in data:
			add_var('S_mag', data['S_mag'], 45, ['fft_time', 'frequencies'], 'W/m^2/Hz', 'S_mag!C', [n_freqs],
					{'SCALETYP': 'log', 'VALIDMIN': 1e-12, 'VALIDMAX': 1e5, 'TITLE': 'Poynting Flux Magnitude'})
			add_var('S_theta', data['Poynting_theta'], 45, ['fft_time', 'frequencies'], 'deg', 'S_Theta', [n_freqs],
					{'SCALETYP': 'linear', 'VALIDMIN': 0.0, 'VALIDMAX': 180.0, 'TITLE': 'Poynting Flux Theta'})
			add_var('S_phi', data['Poynting_phi'], 45, ['fft_time', 'frequencies'], 'deg', 'S_Phi', [n_freqs],
					{'SCALETYP': 'linear', 'VALIDMIN': -180.0, 'VALIDMAX': 180.0, 'TITLE': 'Poynting Flux Phi'})
			
		if 'S_vector' in data:
			add_var('S_fieldAligned', data['S_vector'], 45, ['fft_time', 'frequencies', 'comp_index'], 'W/m^2/Hz', 'S', [n_freqs, 3], 
					{'TITLE': 'Poynting Flux Vector (FA)', 'LABL_PTR_2': 'label_S_vec'})
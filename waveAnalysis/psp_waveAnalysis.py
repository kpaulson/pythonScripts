import argparse
import os

# MUST BE SET BEFORE IMPORTING NUMPY/SCIPY
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import glob
import sys
import base64
import gc
import subprocess
import time
import numpy as np
import pandas as pd
import cdflib
from scipy.ndimage import gaussian_filter1d
from datetime import datetime, timedelta, timezone
import concurrent.futures
import resource
from multiprocessing import Manager
from rich.live import Live
from rich.table import Table
from rich.panel import Panel

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
psp_root = os.path.dirname(current_dir)
if psp_root not in sys.path:
	sys.path.insert(0, psp_root)

import config
from core_waves import data_loader, field_aligner, dsp, polarization_means, poynting
import analysis_io
import re
import urllib.request

# --- 1. DEFINE INGESTION ROOTS EXPLICITLY ---
DRIVE_ROOT = config.get_drive_path()

SPP_CACHE_ROOT = os.path.join(
	DRIVE_ROOT, 
	'Research/Data/AutoplotCache/http/research.ssl.berkeley.edu/data/spp/data/sci'
)

REMOTE_BASE_URL = "https://research.ssl.berkeley.edu/data/spp/data/sci/"

MAG_SC_ROOT   = os.path.join(SPP_CACHE_ROOT, 'fields/l2/mag_SC')
EFIELD_ROOT   = os.path.join(SPP_CACHE_ROOT, 'fields/l2/dfb_wf_dvdc')

def limit_memory(max_gb):
	"""Sets a hard limit on the RAM this specific Python process can use."""
	max_bytes = int(max_gb * 1024 * 1024 * 1024)
	soft, hard = resource.getrlimit(resource.RLIMIT_AS)
	resource.setrlimit(resource.RLIMIT_AS, (max_bytes, hard))
	
def set_terminal_tab_title(title_text):
	"""Writes OSC escape sequences directly to /dev/tty, bypassing
	sys.stdout and Rich redirection to avoid UI corruption."""
	try:
		if os.name == 'posix':
			with open('/dev/tty', 'w') as tty:
				tty.write(f"\033]0;{title_text}\007")
				tty.flush()
		elif os.name == 'nt':
			with open('CON', 'w') as tty:
				tty.write(f"\033]0;{title_text}\007")
				tty.flush()
	except Exception:
		pass
		
def load_auth_credentials(auth_path=None):
	"""
	Loads PSP FIELDS HTTP Basic Auth credentials from ~/MyDrive/.auth.
	Looks specifically for 'psp_flds_username' and 'psp_flds_password'.
	"""
	if auth_path is None:
		auth_path = os.path.expanduser('~/MyDrive/.auth')
		
	if not os.path.exists(auth_path):
		return None
		
	creds = {}
	try:
		with open(auth_path, 'r') as f:
			for line in f:
				line = line.strip()
				if line and '=' in line and not line.startswith('#'):
					k, v = line.split('=', 1)
					creds[k.strip()] = v.strip()
					
		user = creds.get('psp_flds_username')
		passwd = creds.get('psp_flds_password')
		
		return (user, passwd) if user and passwd else None
	except Exception:
		return None

def get_latest_file(pattern):
	files = sorted(glob.glob(pattern))
	return files[-1] if files else None
	
def fetch_missing_cdf(rel_subpath, y, m, file_pattern_regex, target_dir, auth=None):
	"""
	Scrapes remote SSL directory for matching files and streams the latest version
	directly into the local AutoplotCache structure.
	"""
	remote_dir_url = f"{REMOTE_BASE_URL}{rel_subpath}/{y}/{m}/"
	os.makedirs(target_dir, exist_ok=True)

	try:
		headers = {}
		if auth:
			user, passwd = auth
			b64_creds = base64.b64encode(f"{user}:{passwd}".encode()).decode()
			headers["Authorization"] = f"Basic {b64_creds}"

		req = urllib.request.Request(remote_dir_url, headers=headers)
		with urllib.request.urlopen(req, timeout=12) as response:
			html = response.read().decode('utf-8')

		matches = sorted(list(set(re.findall(file_pattern_regex, html))))
		if not matches:
			return None

		latest_filename = matches[-1]
		remote_file_url = f"{remote_dir_url}{latest_filename}"
		local_filepath = os.path.join(target_dir, latest_filename)

		dl_req = urllib.request.Request(remote_file_url, headers=headers)
		with urllib.request.urlopen(dl_req, timeout=60) as resp, open(local_filepath, 'wb') as out_f:
			out_f.write(resp.read())

		return local_filepath

	except Exception:
		return None

def print_boxed_params(args):
	"""Prints a clear, clean parameter checklist on pipeline startup."""
	title = "INPUT PARAMETERS (v1.4 Pipeline)"
	params = vars(args)
	items = sorted(params.items())
	kw = max([len(str(k)) for k in params.keys()]) + 2
	vw = max([len(str(v)) for v in params.values()]) + 2
	if (kw + vw + 1) < (len(title) + 2): vw = (len(title) + 2) - kw - 1
	iw = kw + vw + 1
	sep = "+" + "-" * kw + "+" + "-" * vw + "+"
	print("+" + "-" * iw + "+")
	print("| " + title.center(iw - 2) + " |")
	print(sep)
	for k, v in items:
		print("| %-*s| %-*s|" % (kw - 1, str(k), vw - 1, str(v)))
	print(sep)
	
def make_dashboard(worker_states, completed, total):
	"""Renders a dynamic live multi-row UI panel for Wave Analysis."""
	table = Table(expand=True)
	table.add_column("Worker", justify="center", style="cyan bold", no_wrap=True)
	table.add_column("Date", justify="center", style="green")
	table.add_column("Chunk", justify="center", style="yellow")
	table.add_column("Status", justify="left", style="white")
	table.add_column("Peak RAM", justify="right", style="blue")

	for slot_id in sorted(worker_states.keys()):
		state = worker_states[slot_id]
		table.add_row(
			f"Worker {slot_id + 1}",
			state.get('date', 'Idle'),
			state.get('chunk', '-'),
			state.get('status', 'Idle'),
			state.get('ram', '-')
		)
		
	pct = (completed / total * 100) if total > 0 else 0.0
	title_str = f"Parker Solar Probe Wave Analysis | Progress: {completed}/{total} Chunks ({pct:.1f}%)"
	return Panel(table, title=title_str, border_style="bold blue")
	
def parse_date_range(date_arg):
	"""
	Parses date ranges or single 6-hour chunk start timestamps:
	  - Single ISO Chunk: '2021-11-16T00:00' (Auto-expands 6 hours)
	  - Day Range       : '2019-02-20 through 2019-02-25'
	  - Explicit Window : '2019-02-20 12:00 through 2019-02-20 18:00'
	"""
	date_arg = date_arg.strip()
	
	# Handle single timestamp entries from batch files (e.g., '2021-11-16T00:00')
	if ' through ' not in date_arg:
		for fmt in ('%Y-%m-%dT%H:%M', '%Y-%m-%d %H:%M', '%Y-%m-%d'):
			try:
				dt = datetime.strptime(date_arg, fmt)
				if dt.hour not in [0, 6, 12, 18] or dt.minute != 0:
					raise ValueError(f"Timestamp '{date_arg}' must align to 6-hour boundaries (00:00, 06:00, 12:00, or 18:00).")
				return dt, dt + timedelta(hours=6)
			except ValueError as e:
				if "must align" in str(e):
					raise e
				pass
		raise ValueError(f"Unrecognized timestamp format: '{date_arg}'.")

	# Handle 'through' range strings
	parts = date_arg.split(' through ')
	start_str, end_str = parts[0].strip(), parts[1].strip()

	def parse_endpoint(s, is_end_day=False):
		for fmt in ('%Y-%m-%d %H:%M', '%Y-%m-%dT%H:%M'):
			try:
				dt = datetime.strptime(s, fmt)
				if dt.hour not in [0, 6, 12, 18] or dt.minute != 0:
					raise ValueError(f"Timestamp '{s}' must align to 6-hour boundaries (00:00, 06:00, 12:00, or 18:00).")
				return dt, True
			except ValueError as e:
				if "must align" in str(e):
					raise e
				pass

		try:
			dt = datetime.strptime(s, '%Y-%m-%d')
			if is_end_day:
				return dt + timedelta(days=1), False
			return dt, False
		except ValueError:
			raise ValueError(f"Unrecognized format: '{s}'. Use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'.")

	start_dt, start_has_time = parse_endpoint(start_str, is_end_day=False)
	end_dt, end_has_time = parse_endpoint(end_str, is_end_day=True)

	if start_has_time and end_has_time and start_dt == end_dt:
		end_dt = start_dt + timedelta(hours=6)

	if start_dt >= end_dt:
		raise ValueError(f"Start timestamp ({start_dt.strftime('%Y-%m-%d %H:%M')}) must be strictly before end timestamp ({end_dt.strftime('%Y-%m-%d %H:%M')}).")

	return start_dt, end_dt
	
def patch_telemetry_gaps(data_array, max_gap_samples=10):
	"""Linearly interpolates tiny raw telemetry NaNs before FFT processing."""
	if data_array is None:
		return None
	df = pd.DataFrame(data_array)
	df = df.interpolate(method='linear', limit=max_gap_samples, axis=0)
	return df.to_numpy()
	
def compute_local_background(B_raw, fs, window_sec=60.0):
	"""
	Computes a continuous background B-field using a NaN-safe normalized 
	Gaussian filter. Prevents brief telemetry dropouts or CDF fill values 
	from corrupting surrounding valid data.
	"""
	B_clean = B_raw.copy()
	
	# Clean CDF fill values (-1e31) and non-finite numbers to NaN
	invalid_mask = (np.abs(B_clean) > 1e5) | (~np.isfinite(B_clean))
	B_clean[invalid_mask] = np.nan
	
	sigma_samples = (window_sec * fs) / 2.0
	if sigma_samples < 1.0:
		sigma_samples = 1.0
		
	B_bg = np.zeros_like(B_clean)
	
	for i in range(3):
		val = B_clean[:, i]
		valid = np.isfinite(val).astype(float)
		val_zeroed = np.nan_to_num(val)
		
		smooth_val = gaussian_filter1d(val_zeroed, sigma=sigma_samples, mode='nearest')
		smooth_weight = gaussian_filter1d(valid, sigma=sigma_samples, mode='nearest')
		
		with np.errstate(divide='ignore', invalid='ignore'):
			bg_i = smooth_val / smooth_weight
			
		bg_i[smooth_weight < 0.01] = np.nan
		B_bg[:, i] = bg_i
		
	return B_bg

def process_chunk(chunk_tuple, slot_queue, status_queue, log_queue):
	start_dt, end_dt, params = chunk_tuple
	
	slot_id = slot_queue.get()
	y, m, d = start_dt.strftime('%Y'), start_dt.strftime('%m'), start_dt.strftime('%d')
	hr_str = start_dt.strftime('%H')
	date_str = f"{y}-{m}-{d}"
	chunk_str = f"{hr_str}:00"

	def safe_print(msg):
		log_queue.put(msg)

	def update_ui(status_text):
		peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
		ram_str = f"{peak_kb / (1024 * 1024):.2f} GB"
		status_queue.put({
			'slot_id': slot_id, 'date': date_str, 'chunk': chunk_str,
			'status': status_text, 'ram': ram_str
		})

	try:
		update_ui("Initializing")
		
		# --- 0. OVERWRITE SAFEGUARD ---
		out_dir = os.path.join(config.get_drive_path(), 'Research/PSP/WaveAnalysis/WaveAnalysis_Files', f"v{params['v_out']}", y, m)
		out_file = os.path.join(out_dir, f"PSP_WaveAnalysis_{y}-{m}-{d}_{hr_str}00_v{params['v_out']}.cdf")
		
		if not params['overwrite'] and os.path.exists(out_file):
			if params['verbose']:
				safe_print(f"    [*] File already exists. Skipping (overwrite=F): {os.path.basename(out_file)}")
			update_ui("Skipped (Exists)")
			return True
		
		# --- 1. LOCAL CACHE FILE DISCOVERY & FALLBACK DOWNLOAD ---
		update_ui("Resolving Files")
		mag_dir = os.path.join(MAG_SC_ROOT, y, m)
		ef_dir  = os.path.join(EFIELD_ROOT, y, m)
		
		mag_pattern = os.path.join(mag_dir, f"psp_fld_l2_mag_SC_{y}{m}{d}{hr_str}_v*.cdf")
		ef_pattern  = os.path.join(ef_dir, f"psp_fld_l2_dfb_wf_dvdc_{y}{m}{d}{hr_str}_v*.cdf")
			
		mag_file = get_latest_file(mag_pattern)
		ef_file  = get_latest_file(ef_pattern)

		auth = params.get('auth_credentials')

		if not mag_file:
			update_ui("Downloading Mag High-Res")
			reg = rf"psp_fld_l2_mag_SC_{y}{m}{d}{hr_str}_v\d+\.cdf"
			mag_file = fetch_missing_cdf("fields/l2/mag_SC", y, m, reg, mag_dir, auth=auth)

		if not ef_file:
			update_ui("Downloading E-Field")
			reg = rf"psp_fld_l2_dfb_wf_dvdc_{y}{m}{d}{hr_str}_v\d+\.cdf"
			ef_file = fetch_missing_cdf("fields/l2/dfb_wf_dvdc", y, m, reg, ef_dir, auth=auth)
		
		if params['verbose']:
			if mag_file: safe_print(f"    [+] RESOLVED High-Res Mag File: {mag_file}")
			else: safe_print(f"    [!] FAILED to find Mag High-Res file matching: {os.path.basename(mag_pattern)}")

			if ef_file: safe_print(f"    [+] RESOLVED E-Field File  : {ef_file}")
			else: safe_print(f"    [-] E-Field file optional/not found matching: {os.path.basename(ef_pattern)}")
		
		if not mag_file:
			if params['verbose']:
				safe_print(f"    [!] Missing critical B-field data for {start_dt}. Skipping.\n")
			update_ui("Skipped (Missing B)")
			return False
			
		# --- 2. LOAD & CLEAN HIGH-RES TELEMETRY ---
		update_ui("Loading Cache Data")
		with data_loader.smart_cdf(mag_file) as cdf_b:
			B_raw = cdf_b.varget('psp_fld_l2_mag_SC')
			t_B_unix = cdflib.cdfepoch.unixtime(cdf_b.varget('epoch_mag_SC'))
			
		if t_B_unix is not None:
			t_B_unix = np.atleast_1d(t_B_unix)
		if B_raw is not None:
			B_raw = np.atleast_1d(B_raw)
			
		if B_raw is None or len(t_B_unix) < params['windowWidth']:
			if params['verbose']:
				safe_print(f"    [!] Insufficient data points ({len(t_B_unix) if t_B_unix is not None else 0} < {params['windowWidth']}). Skipping chunk.")
			update_ui("Skipped (Sparse Data)")
			return False

		# Clean fill values and patch single-sample telemetry NaNs
		invalid_B = (np.abs(B_raw) > 1e5) | (~np.isfinite(B_raw))
		B_raw[invalid_B] = np.nan
		B_raw = patch_telemetry_gaps(B_raw, max_gap_samples=10)

		E_raw, t_E_unix = None, None
		if ef_file:
			try:
				with data_loader.smart_cdf(ef_file) as cdf_e:
					E_raw = cdf_e.varget('psp_fld_l2_dfb_wf_dVdc_sc')
					t_E_unix = cdflib.cdfepoch.unixtime(cdf_e.varget('epoch'))
					if params['verbose']:
						safe_print(f"    -> Converting raw differential voltage to mV/m by calculating [E_raw * 1000.0 / {params['antennaLength']}]")
					E_raw = (E_raw * 1000.0) / params['antennaLength']
					
					invalid_E = (np.abs(E_raw) > 1e5) | (~np.isfinite(E_raw))
					E_raw[invalid_E] = np.nan
					E_raw = patch_telemetry_gaps(E_raw, max_gap_samples=10)
			except Exception as e:
				safe_print(f"    [!] Failed to read E-field: {e}. Proceeding with B-field only.")
				ef_file = "None"
				
		fs = 1.0 / np.nanmedian(np.diff(t_B_unix))
		
		# --- 3. BACKGROUND FIELD & E-FIELD FILTERING ---
		update_ui("Syncing & Filtering")
		
		# Compute smooth, NaN-safe background directly from high-res B-field
		B_bg = compute_local_background(B_raw, fs, window_sec=params['trendSeconds'])
		
		nyquist_E = None
		if E_raw is not None:
			fs_E = 1.0 / np.nanmedian(np.diff(t_E_unix))
			nyquist_B = fs / 2.0
			nyquist_E = fs_E / 2.0
			safe_nyquist = min(nyquist_B, nyquist_E)
			highFreqELimit = safe_nyquist * 0.95 
			cutoff = min(highFreqELimit, params['maxFrequency'])
			
			if params['verbose']:
				safe_print(f"      [DEBUG] B-Nyq: {nyquist_B:.1f}Hz | E-Nyq: {nyquist_E:.1f}Hz | Filter Cutoff: {cutoff:.1f}Hz")
				
			Ex_filt = data_loader.apply_butterworth(E_raw[:, 0], fs_E, cutoff_freq=cutoff)
			Ey_filt = data_loader.apply_butterworth(E_raw[:, 1], fs_E, cutoff_freq=cutoff)
			
			Ex_sync = data_loader.sync_time_grids(t_E_unix, Ex_filt, t_B_unix)
			Ey_sync = data_loader.sync_time_grids(t_E_unix, Ey_filt, t_B_unix)
			
			with np.errstate(divide='ignore', invalid='ignore'):
				Ez_sync = -(Ex_sync * B_raw[:, 0] + Ey_sync * B_raw[:, 1]) / B_raw[:, 2]
			Ez_sync[~np.isfinite(Ez_sync)] = np.nan
		else:
			Ex_sync, Ey_sync, Ez_sync = None, None, None

		# --- 4. COORDINATE ALIGNMENT ---
		update_ui("Aligning Coordinates")
		if B_raw.ndim == 1:
			raise ValueError("High-res B_raw is 1D. Expecting a 2D array of shape (N, 3).")
			
		B_bg_mag = np.linalg.norm(B_bg, axis=-1)
		with np.errstate(divide='ignore', invalid='ignore'):
			sin_tilt = np.abs(B_bg[:, 2] / B_bg_mag)
			tilt_angle_deg = np.degrees(np.arcsin(sin_tilt))
			
		Rx = np.zeros(len(t_B_unix))
		Ry = np.full(len(t_B_unix), -1.0)
		Rz = np.zeros(len(t_B_unix))
		
		Bn, Bp, Bq, Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz = field_aligner.align_to_background(
			B_raw[:, 0], B_raw[:, 1], B_raw[:, 2], 
			Rx, Ry, Rz,
			Bx_bg=B_bg[:, 0], By_bg=B_bg[:, 1], Bz_bg=B_bg[:, 2]
		)
		
		if params['verbose']:	
			for name, vx, vy, vz in [('N', Nx, Ny, Nz), ('P', Px, Py, Pz), ('Q', Qx, Qy, Qz)]:
				norm = np.sqrt(vx**2 + vy**2 + vz**2)
				safe_print(f"    [DIAGNOSTIC] Norm of {name} vector: Min={np.nanmin(norm)}, Max={np.nanmax(norm)}")
		
		if Ex_sync is not None:
			En = (Ex_sync * Nx) + (Ey_sync * Ny) + (Ez_sync * Nz)
			Ep = (Ex_sync * Px) + (Ey_sync * Py) + (Ez_sync * Pz)
			Eq = (Ex_sync * Qx) + (Ey_sync * Qy) + (Ez_sync * Qz)
		else:
			En, Ep, Eq = None, None, None

		# --- 5. SIGNAL ANALYSIS & FFTS ---
		update_ui("Calculating FFTs")
		t_fft_sec, freqs, Bn_fft, Bp_fft, Bq_fft, En_fft, Ep_fft, Eq_fft = dsp.process_all_ffts(
			Bn, Bp, Bq, En, Ep, Eq, 
			fs, params['minFrequency'], params['maxFrequency'], params['windowWidth'], params['slideFactor']
		)
		
		t_fft_unix = t_B_unix[0] + t_fft_sec

		# --- 5.1 DYNAMIC HIGH-FREQUENCY FILL MASK ---
		from scipy.interpolate import interp1d
		dt_B = np.diff(t_B_unix)
		dt_B = np.append(dt_B, dt_B[-1]) 
		inst_nyquist_B = (0.5 / dt_B) * 1.05 
		
		nyq_interp = interp1d(t_B_unix, inst_nyquist_B, kind='nearest', bounds_error=False, fill_value='extrapolate')
		local_nyquist_B = nyq_interp(t_fft_unix)
		
		invalid_B_mask = freqs[None, :] > local_nyquist_B[:, None]
		
		Bn_fft[invalid_B_mask] = np.nan
		Bp_fft[invalid_B_mask] = np.nan
		Bq_fft[invalid_B_mask] = np.nan
		
		if En_fft is not None:
			invalid_E_mask = invalid_B_mask.copy()
			if nyquist_E is not None:
				invalid_E_mask |= (freqs[None, :] > nyquist_E)
				
			En_fft[invalid_E_mask] = np.nan
			Ep_fft[invalid_E_mask] = np.nan
			Eq_fft[invalid_E_mask] = np.nan

		# --- 6. GEOMETRIC POLARIZATION ANALYSIS ---
		update_ui("Polarization Math")
		results = polarization_means.calculate_polarization(
			Bn_fft, Bp_fft, Bq_fft, Nx, Ny, Nz, params['wavePowerMin'], En_fft, Ep_fft, Eq_fft
		)
		
		if En_fft is not None:
			update_ui("Poynting Vectors")
			S_mag, S_theta, S_phi, Sn, Sp, Sq, _, _, _ = poynting.Poyntingify(
				Bn_fft, Bp_fft, Bq_fft, En_fft, Ep_fft, Eq_fft, params['poyntingMin']
			)
			results.update({
				'S_mag': S_mag, 'Poynting_theta': S_theta, 'Poynting_phi': S_phi,
				'Sn': Sn, 'Sp': Sp, 'Sq': Sq
			})

		# --- 7. FILE PACKAGING & WRITE ---
		update_ui("Writing CDF")
		tilt_interp = interp1d(t_B_unix, tilt_angle_deg, kind='linear', bounds_error=False, fill_value='extrapolate')
		fft_whip_tilt = tilt_interp(t_fft_unix)
		
		B_FA_vector = np.stack((Bn, Bp, Bq), axis=-1)
		E_FA_vector = np.stack((En, Ep, Eq), axis=-1) if En is not None else None

		if 'Sn' in results:
			results['S_vector'] = np.stack((results['Sn'], results['Sp'], results['Sq']), axis=-1)
			
		if 'kn' in results:
			results['k_vector'] = np.stack((results['kn'], results['kp'], results['kq']), axis=-1)

		results.update({
			'Mag_time': t_B_unix, 'FFT_time': t_fft_unix, 'Frequencies': freqs,
			'B_FA_vector': B_FA_vector,
			'E_FA_vector': E_FA_vector,
			'Antenna_Length': params['antennaLength'],
			'Whip_Tilt_Angle': fft_whip_tilt
		})
		
		os.makedirs(out_dir, exist_ok=True)

		analysis_io.write_waveAnalysis_cdf(
			out_file, results, params['v_out'], os.path.basename(mag_file), os.path.basename(str(ef_file)), 
			logy=params['logy'], verbose=params['verbose']
		)
		
		# --- 8. AUTOMATIC SUBPROCESS PLOTTER ---
		if params['plotStuff']:
			update_ui("Plotting Spectrogram")
			plotter_script = os.path.join(psp_root, 'PSP', 'psp_waveAnalysis_plotter.py')
			start_str = start_dt.strftime('%Y-%m-%d %H:%M')
			end_str = end_dt.strftime('%Y-%m-%d %H:%M')
			
			cmd = [sys.executable, plotter_script, start_str, end_str, "--mode", "batch", "--v_out", params['v_out']]
			if params['logy']: cmd.append("--logy")
			cmd.append("--force")
			
			try:
				if params['verbose']:
					subprocess.run(cmd, check=True)
				else:
					subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
			except Exception as e:
				safe_print(f"    [!] Plotter execution failed: {e}")
				
		update_ui("Complete")
		return True

	except Exception as e:
		update_ui(f"Failed: {e}")
		safe_print(f"    [!] Fatal Chunk Error ({date_str} {chunk_str}): {e}")
		return False
		
	finally:
		gc.collect()
		slot_queue.put(slot_id)

def main():
	params = setup()

	if not params['date'] and not params['batch_file']:
		print("\n[!] ERROR: Must specify either --date or --batch_file.\n")
		sys.exit(1)

	chunk_tasks = []
	
	if params['batch_file']:
		if not os.path.exists(params['batch_file']):
			print(f"\n[!] ERROR: Batch file not found: '{params['batch_file']}'\n")
			sys.exit(1)
			
		with open(params['batch_file'], 'r') as f:
			for line in f:
				line = line.strip()
				if line and not line.startswith('#'):
					try:
						s_dt, e_dt = parse_date_range(line)
						chunk_tasks.append((s_dt, e_dt, params))
					except ValueError as err:
						print(f"[!] Warning: Skipping invalid batch file entry '{line}': {err}")
	else:
		try:
			start_dt, end_dt = parse_date_range(params['date'])
		except ValueError as err:
			print(f"\n[!] ERROR: {err}\n")
			sys.exit(1)

		curr = start_dt
		while curr < end_dt:
			chunk_end = curr + timedelta(hours=6)
			chunk_tasks.append((curr, chunk_end, params))
			curr = chunk_end

	total_chunks = len(chunk_tasks)
	if total_chunks == 0:
		print("[!] No work chunks generated for given range.")
		sys.exit(0)
		
	completed_chunks = 0
	
	manager = Manager()
	slot_queue = manager.Queue()
	status_queue = manager.Queue()
	log_queue = manager.Queue()
	
	for i in range(params['workers']):
		slot_queue.put(i)
		
	worker_states = {i: {'date': 'Idle', 'chunk': '-', 'status': 'Idle', 'ram': '-'} for i in range(params['workers'])}

	executor = concurrent.futures.ProcessPoolExecutor(max_workers=params['workers'])
	last_completed = -1

	try:
		with Live(make_dashboard(worker_states, 0, total_chunks), refresh_per_second=4) as live:
			futures = {
				executor.submit(process_chunk, task, slot_queue, status_queue, log_queue): task 
				for task in chunk_tasks
			}
			
			completed_futures = set()
			while completed_chunks < total_chunks:
				while not log_queue.empty():
					msg = log_queue.get()
					live.console.print(msg)

				while not status_queue.empty():
					msg = status_queue.get()
					slot = msg['slot_id']
					worker_states[slot].update(msg)

				done = [f for f in futures if f.done() and f not in completed_futures]
				for f in done:
					completed_futures.add(f)
					completed_chunks += 1

				if completed_chunks != last_completed:
					last_completed = completed_chunks
					pct = int((completed_chunks / total_chunks) * 100)
					set_terminal_tab_title(f"[{pct}%] WaveAnalysis ({completed_chunks}/{total_chunks})")

				live.update(make_dashboard(worker_states, completed_chunks, total_chunks))
				time.sleep(0.1)

	except KeyboardInterrupt:
		set_terminal_tab_title("[ABORTED] WaveAnalysis")
		print("\n\n[!] KeyboardInterrupt detected! Force-killing worker pool and releasing memory...")
		executor.shutdown(wait=False, cancel_futures=True)
		manager.shutdown()
		sys.exit(1)

	else:
		executor.shutdown(wait=True)
		manager.shutdown()
		set_terminal_tab_title("[DONE] WaveAnalysis Complete!")
		print("\n[+] Multiprocessing execution complete.")

def setup():
	parser = argparse.ArgumentParser(description="Multi-Mission Level 3 WaveAnalysis Pipeline (Python 3)")
	
	parser.add_argument('--date', type=str, default=None, help="Format: 'YYYY-MM-DD through YYYY-MM-DD'")
	parser.add_argument('-v_out', type=str, default="1.4", help="Data tracking minor release version")
	
	parser.add_argument('--minFrequency', type=float, default=0.1)
	parser.add_argument('--maxFrequency', type=float, default=64.0)
	parser.add_argument('--windowWidth', type=int, default=1024)
	parser.add_argument('--slideFactor', type=int, default=8)
	parser.add_argument('--trendSeconds', type=int, default=60)
	
	parser.add_argument('--poyntingMin', type=float, default=0.0, help="Noise gate power threshold")
	parser.add_argument('--wavePowerMin', type=float, default=0.0, help="Noise gate power threshold")
	
	parser.add_argument('--antennaLength', type=float, default=3.5, help="Baseline physical antenna length in meters")
	
	parser.add_argument('--no_plot', action='store_false', dest='plotStuff', help="Disable execution of matplotlib plotter child")
	parser.add_argument('--linear', action='store_false', dest='logy', help="Forces spectrogram Y-axes onto a linear scale")
	parser.add_argument('--removeReactionWheelTone', action='store_true', help="Filter out reaction wheel tones from spectrogram")
	
	parser.add_argument('--workers', type=int, default=2, help="Number of concurrent multiprocessing workers")
	parser.add_argument('--batch_file', type=str, default=None, help="Text file containing list of chunk ranges to re-run")
	parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose terminal output")
	parser.add_argument('--overwrite', '-o', action='store_true', help="Overwrite existing CDF files")
	
	args = parser.parse_args()
	print_boxed_params(args)
	
	params = vars(args)
	params['auth_credentials'] = load_auth_credentials()
	
	return params

if __name__ == "__main__":
	main()
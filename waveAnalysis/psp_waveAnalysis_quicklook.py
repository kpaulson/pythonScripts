import argparse
import os
import sys

# --- RDP / X11 WORKAROUND: Enable WebAgg Backend if requested or running over RDP ---
if "--web" in sys.argv or "XRDP_SESSION" in os.environ or "SSH_CONNECTION" in os.environ:
    import matplotlib
    matplotlib.use('WebAgg') # Serves interactive plot into browser tab, bypassing X11/RDP corruption!
	
import glob
import re
import base64
import urllib.request
import numpy as np
import cdflib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm, Normalize
from datetime import datetime, timedelta, timezone
from scipy.ndimage import gaussian_filter1d

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
psp_root = os.path.dirname(current_dir)
if psp_root not in sys.path:
	sys.path.insert(0, psp_root)

import config
from core_waves import data_loader, field_aligner, dsp, polarization_means, poynting
from psp_waveAnalysis import load_auth_credentials, fetch_missing_cdf, get_latest_file

# Set up global plot interactive style
plt.rcParams.update({'font.size': 10})

# Path constants
DRIVE_ROOT = config.get_drive_path()
SPP_CACHE_ROOT = os.path.join(
	DRIVE_ROOT, 
	'Research/Data/AutoplotCache/http/research.ssl.berkeley.edu/data/spp/data/sci'
)
MAG_SC_ROOT = os.path.join(SPP_CACHE_ROOT, 'fields/l2/mag_SC')
EFIELD_ROOT = os.path.join(SPP_CACHE_ROOT, 'fields/l2/dfb_wf_dvdc')


# -----------------------------------------------------------------------------
# 1. HELPER: Continuous Local Background B-field
# -----------------------------------------------------------------------------
def compute_local_background(B_raw, fs, window_sec=60.0):
	"""
	Computes a continuous background B-field using a NaN-safe normalized 
	Gaussian filter. Prevents brief telemetry dropouts or CDF fill values 
	from corrupting surrounding valid data.
	"""
	B_clean = B_raw.copy()
	
	# 1. Clean CDF fill values (-1e31) and non-finite numbers to NaN
	invalid_mask = (np.abs(B_clean) > 1e5) | (~np.isfinite(B_clean))
	B_clean[invalid_mask] = np.nan
	
	sigma_samples = (window_sec * fs) / 2.0
	if sigma_samples < 1.0:
		sigma_samples = 1.0
		
	B_bg = np.zeros_like(B_clean)
	
	# 2. Normalized Gaussian filter (NaN-safe smoothing)
	for i in range(3):
		val = B_clean[:, i]
		valid = np.isfinite(val).astype(float)
		val_zeroed = np.nan_to_num(val)
		
		# Convolve both the zeroed data and valid weights
		smooth_val = gaussian_filter1d(val_zeroed, sigma=sigma_samples, mode='nearest')
		smooth_weight = gaussian_filter1d(valid, sigma=sigma_samples, mode='nearest')
		
		# Divide to get exact weighted average of valid points
		with np.errstate(divide='ignore', invalid='ignore'):
			bg_i = smooth_val / smooth_weight
			
		# Only set to NaN if an entire window contains zero valid data points
		bg_i[smooth_weight < 0.01] = np.nan
		B_bg[:, i] = bg_i
		
	return B_bg


# -----------------------------------------------------------------------------
# 2. HELPER: High-Res Multi-File Data Ingestion
# -----------------------------------------------------------------------------
def load_highres_range(start_dt, end_dt, antenna_length=3.5, auth=None):
	"""
	Discovers, fetches, and concatenates high-res B and E field data across 
	whatever 6-hour boundaries are spanned by the requested time window.
	"""
	# Identify all 6-hour chunks covered by the range
	curr = start_dt.replace(minute=0, second=0, microsecond=0)
	curr = curr - timedelta(hours=curr.hour % 6)
	
	chunk_starts = []
	while curr < end_dt:
		chunk_starts.append(curr)
		curr += timedelta(hours=6)

	b_list, tb_list = [], []
	e_list, te_list = [], []

	for dt in chunk_starts:
		y, m, d, hr = dt.strftime('%Y'), dt.strftime('%m'), dt.strftime('%d'), dt.strftime('%H')
		mag_dir = os.path.join(MAG_SC_ROOT, y, m)
		ef_dir  = os.path.join(EFIELD_ROOT, y, m)

		mag_pat = os.path.join(mag_dir, f"psp_fld_l2_mag_SC_{y}{m}{d}{hr}_v*.cdf")
		ef_pat  = os.path.join(ef_dir, f"psp_fld_l2_dfb_wf_dvdc_{y}{m}{d}{hr}_v*.cdf")

		mag_file = get_latest_file(mag_pat) or fetch_missing_cdf("fields/l2/mag_SC", y, m, rf"psp_fld_l2_mag_SC_{y}{m}{d}{hr}_v\d+\.cdf", mag_dir, auth=auth)
		ef_file  = get_latest_file(ef_pat)  or fetch_missing_cdf("fields/l2/dfb_wf_dvdc", y, m, rf"psp_fld_l2_dfb_wf_dvdc_{y}{m}{d}{hr}_v\d+\.cdf", ef_dir, auth=auth)

		if mag_file:
			with data_loader.smart_cdf(mag_file) as cdf_b:
				b_list.append(cdf_b.varget('psp_fld_l2_mag_SC'))
				tb_list.append(cdflib.cdfepoch.unixtime(cdf_b.varget('epoch_mag_SC')))

		if ef_file:
			try:
				with data_loader.smart_cdf(ef_file) as cdf_e:
					e_raw = cdf_e.varget('psp_fld_l2_dfb_wf_dVdc_sc')
					te_raw = cdflib.cdfepoch.unixtime(cdf_e.varget('epoch'))
					e_raw = (e_raw * 1000.0) / antenna_length
					e_list.append(e_raw)
					te_list.append(te_raw)
			except Exception:
				pass

	if not b_list:
		raise ValueError(f"No high-res B-field data found for period {start_dt} to {end_dt}")

	# Concatenate arrays
	B_full = np.vstack(b_list)
	t_B_full = np.concatenate(tb_list)

	E_full, t_E_full = None, None
	if e_list:
		E_full = np.vstack(e_list)
		t_E_full = np.concatenate(te_list)

	# Slice strictly to user requested timestamps (with 60-second padding for FFT windows)
	pad_sec = 60.0
	t_start_padded = start_dt.replace(tzinfo=timezone.utc).timestamp() - pad_sec
	t_end_padded   = end_dt.replace(tzinfo=timezone.utc).timestamp() + pad_sec

	b_mask = (t_B_full >= t_start_padded) & (t_B_full <= t_end_padded)
	B_raw = B_full[b_mask]
	t_B = t_B_full[b_mask]

	if E_full is not None:
		e_mask = (t_E_full >= t_start_padded) & (t_E_full <= t_end_padded)
		E_raw = E_full[e_mask]
		t_E = t_E_full[e_mask]
	else:
		E_raw, t_E = None, None

	return t_B, B_raw, t_E, E_raw


# -----------------------------------------------------------------------------
# 3. INTERACTIVE PLOTTER (Customizable Panel Stack)
# -----------------------------------------------------------------------------
def plot_interactive_stack(t_B_datetime, B_FA, B_raw, t_fft_datetime, freqs, results, start_dt, end_dt, use_log_y=False):
	"""
	Generates a 6-panel interactive Matplotlib GUI stack with linked X-axes.
	"""
	fig = plt.figure(figsize=(13, 11))
	gs = gridspec.GridSpec(6, 1, height_ratios=[1.2, 1, 1, 1, 1, 1])
	plt.subplots_adjust(hspace=0.08, left=0.10, right=0.84, top=0.94, bottom=0.08)

	# --- Panel 0: Field-Aligned Magnetic Field ---
	ax0 = fig.add_subplot(gs[0])
	ax0.plot(t_B_datetime, B_FA[:, 0], color='black', lw=0.8, label=r'$B_{||}$')
	ax0.plot(t_B_datetime, B_FA[:, 1], color='red',   lw=0.8, label=r'$B_{\perp T}$')
	ax0.plot(t_B_datetime, B_FA[:, 2], color='blue',  lw=0.8, label=r'$B_{\perp N}$')
	
	# Optional: Overlay raw spacecraft components to verify telemetry continuity
	ax0.plot(t_B_datetime, B_raw[:, 0], color='gray',  lw=0.5, alpha=0.5, label=r'$B_x$')
	ax0.plot(t_B_datetime, B_raw[:, 1], color='pink',  lw=0.5, alpha=0.5, label=r'$B_y$')
	ax0.plot(t_B_datetime, B_raw[:, 2], color='cyan',  lw=0.5, alpha=0.5, label=r'$B_z$')
	
	ax0.set_ylabel('B (nT)', fontweight='bold')
	ax0.set_xlim(start_dt, end_dt)
	ax0.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False)
	ax0.set_title(f"PSP QuickLook WaveAnalysis | {start_dt.strftime('%Y-%m-%d %H:%M:%S')} to {end_dt.strftime('%H:%M:%S')} UTC", fontweight='bold')
	plt.setp(ax0.get_xticklabels(), visible=False)

	# Panel configuration maps
	spec_vars = [
		('Power_perp', 'turbo', LogNorm(1e-2, 1e3), r'PSD $B_{\perp}$\n($nT^2/Hz$)'),
		('Poynting_theta', 'RdBu_r', Normalize(0, 180), r'$S_{\theta}$' + '\n' + '(deg)'),
		('Ellipticity', 'RdBu_r', Normalize(-1, 1), 'Ellipticity\nRH       LH'),
		('Coherency', 'gray_r', Normalize(0, 1), 'Coherency'),
		('Angle_Normal', 'gray', Normalize(0, 90), 'Wave Normal\n' + r'$\theta_k$ (deg)')
	]

	axes = [ax0]
	f_max = np.nanmax(freqs)

	for i, (var_key, cmap, norm, label) in enumerate(spec_vars):
		ax = fig.add_subplot(gs[i+1], sharex=ax0)
		axes.append(ax)

		if var_key in results and results[var_key] is not None:
			z_data = results[var_key].T
			mesh = ax.pcolormesh(t_fft_datetime, freqs, z_data, cmap=cmap, norm=norm, shading='auto')
			
			pos = ax.get_position()
			cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.012, pos.height])
			cb = fig.colorbar(mesh, cax=cax)
			cb.set_label(label, fontweight='bold', rotation=270, labelpad=22)

		ax.set_ylabel('Hz', fontsize=9)
		if use_log_y:
			ax.set_yscale('log')
			ax.set_ylim(0.5, f_max)
		else:
			ax.set_ylim(0, f_max)

		if i < 4:
			plt.setp(ax.get_xticklabels(), visible=False)
		else:
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
			ax.set_xlabel('Time (UTC)', fontweight='bold')

	print("[+] QuickLook Window Opened. Use the toolbar at the bottom to zoom, pan, or save!")
	plt.show()


# -----------------------------------------------------------------------------
# 4. MAIN PIPELINE EXECUTION
# -----------------------------------------------------------------------------
def run_quicklook(start_str, end_str, bg_window=60.0, use_log_y=False):
	for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%dT%H:%M:%S'):
		try:
			start_dt = datetime.strptime(start_str, fmt)
			break
		except ValueError:
			pass

	for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%dT%H:%M:%S'):
		try:
			end_dt = datetime.strptime(end_str, fmt)
			break
		except ValueError:
			pass

	auth = load_auth_credentials()
	
	print(f"    [*] Ingesting High-Res Telemetry for {start_dt} -> {end_dt}...")
	t_B, B_raw, t_E, E_raw = load_highres_range(start_dt, end_dt, auth=auth)

	fs = 1.0 / np.nanmedian(np.diff(t_B))
	print(f"    [*] Magnetic Field Cadence: {fs:.1f} Hz ({len(t_B)} samples)")

	# Derive smooth background field on the fly
	print(f"    [*] Computing gap-free background B-field ({bg_window}s smoothing)...")
	B_bg = compute_local_background(B_raw, fs, window_sec=bg_window)

	# Handle E-field sync if available
	Ex_sync, Ey_sync, Ez_sync = None, None, None
	nyquist_E = None
	if E_raw is not None and len(E_raw) > 0:
		print("    [*] Processing Electric Field Data...")
		fs_E = 1.0 / np.nanmedian(np.diff(t_E))
		nyquist_B, nyquist_E = fs / 2.0, fs_E / 2.0
		cutoff = min(min(nyquist_B, nyquist_E) * 0.95, 64.0)

		Ex_filt = data_loader.apply_butterworth(E_raw[:, 0], fs_E, cutoff_freq=cutoff)
		Ey_filt = data_loader.apply_butterworth(E_raw[:, 1], fs_E, cutoff_freq=cutoff)

		Ex_sync = data_loader.sync_time_grids(t_E, Ex_filt, t_B)
		Ey_sync = data_loader.sync_time_grids(t_E, Ey_filt, t_B)

		with np.errstate(divide='ignore', invalid='ignore'):
			Ez_sync = -(Ex_sync * B_raw[:, 0] + Ey_sync * B_raw[:, 1]) / B_raw[:, 2]
		Ez_sync[~np.isfinite(Ez_sync)] = np.nan

	# Coordinate alignment
	print("    [*] Aligning to Local Field Coordinates (without Singularity Guard)...")
	Rx, Ry, Rz = np.zeros(len(t_B)), np.full(len(t_B), -1.0), np.zeros(len(t_B))
	Bn, Bp, Bq, Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz = field_aligner.align_to_background(
		B_raw[:, 0], B_raw[:, 1], B_raw[:, 2], Rx, Ry, Rz,
		Bx_bg=B_bg[:, 0], By_bg=B_bg[:, 1], Bz_bg=B_bg[:, 2]
	)
	# print("    [*] Aligning to Local Field Coordinates (with Singularity Guard)...")
	# Rx, Ry, Rz = np.zeros(len(t_B)), np.full(len(t_B), -1.0), np.zeros(len(t_B))
	# Bn, Bp, Bq, Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz = field_aligner.align_to_background_robust(
		# B_raw[:, 0], B_raw[:, 1], B_raw[:, 2], Rx, Ry, Rz,
		# Bx_bg=B_bg[:, 0], By_bg=B_bg[:, 1], Bz_bg=B_bg[:, 2]
	# )

	En, Ep, Eq = None, None, None
	if Ex_sync is not None:
		En = (Ex_sync * Nx) + (Ey_sync * Ny) + (Ez_sync * Nz)
		Ep = (Ex_sync * Px) + (Ey_sync * Py) + (Ez_sync * Pz)
		Eq = (Ex_sync * Qx) + (Ey_sync * Qy) + (Ez_sync * Qz)

	# Spectral Analysis
	print("    [*] Calculating FFT Spectrograms...")
	t_fft_sec, freqs, Bn_fft, Bp_fft, Bq_fft, En_fft, Ep_fft, Eq_fft = dsp.process_all_ffts(
		Bn, Bp, Bq, En, Ep, Eq, fs, 0.1, 64.0, 1024, 8
	)

	t_fft_unix = t_B[0] + t_fft_sec

	# Polarization Math
	print("    [*] Calculating Polarization Quantities...")
	results = polarization_means.calculate_polarization(
		Bn_fft, Bp_fft, Bq_fft, Nx, Ny, Nz, 0.0, En_fft, Ep_fft, Eq_fft
	)

	if En_fft is not None:
		S_mag, S_theta, S_phi, Sn, Sp, Sq, _, _, _ = poynting.Poyntingify(
			Bn_fft, Bp_fft, Bq_fft, En_fft, Ep_fft, Eq_fft, 0.0
		)
		results.update({'Poynting_theta': S_theta, 'S_mag': S_mag})

	# Convert timestamps to Python datetimes for Matplotlib plotting
	t_B_dt = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in t_B]
	t_fft_dt = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in t_fft_unix]
	B_FA = np.column_stack((Bn, Bp, Bq))

	# Launch Interactive Figure
	plot_interactive_stack(t_B_dt, B_FA, B_raw, t_fft_dt, freqs, results, start_dt, end_dt, use_log_y=use_log_y)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="PSP Interactive QuickLook Wave Analysis")
	parser.add_argument("start", type=str, help="Start time: 'YYYY-MM-DD HH:MM:SS'")
	parser.add_argument("end", type=str, help="End time: 'YYYY-MM-DD HH:MM:SS'")
	parser.add_argument("--bg_window", type=float, default=60.0, help="Smoothing window (in seconds) for local B0")
	parser.add_argument("--logy", action="store_true", help="Use log scale for frequency axes")
	parser.add_argument("--web", action="store_true", help="Display plot in a web browser (to fix x11 forwarding over rdp)")
	
	args = parser.parse_args()
	run_quicklook(args.start, args.end, bg_window=args.bg_window, use_log_y=args.logy)
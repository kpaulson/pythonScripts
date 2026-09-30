"""
psp_interactivePlot_generator.py
================================
Generates standalone, multi-panel interactive Plotly HTML figures and JSON payloads.
Imports shared backend data extraction from psp_swp_browser_dataProcessor.py.
"""

import os
import sys
import glob
import re
import time
from datetime import datetime, timedelta, timezone
import argparse
import tempfile
import shutil
import warnings
import pandas as pd
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- IMPORT HEADLESS BACKEND DATA PROCESSOR ---
import psp_swp_browser_dataProcessor as dp

VERBOSE = False
DYNAMIC_TICKS = False
JSON_OUT = False
NO_PLOT = False
ACCESS = 'team'
OVERWRITE = False
MIRROR_PUB = False
CSV_ROOT, CSV_URL = "", ""

def vprint(*args, **kwargs):
    if VERBOSE: print(*args, **kwargs)

# =============================================================================
# AXIS & EPHEMERIS FOOTER FORMATTER
# =============================================================================
def format_and_apply_xaxes(fig, start_dt, end_dt, is_encounter, num_rows, tick_times, minor_tick_times):
    """Applies standardized UTC date bounds, ticks, and the ephemeris text footer."""
    vprint("    -> Formatting X-Axes & Ephemeris Footer...")

    # Standardize date strings without trailing 'Z' so Plotly Python matches array tickvals
    start_utc_str = start_dt.strftime('%Y-%m-%d %H:%M:%S')
    end_utc_str = end_dt.strftime('%Y-%m-%d %H:%M:%S')
    tick_vals_std = [t.strftime('%Y-%m-%d %H:%M:%S') for t in tick_times]
    minor_vals_std = [t.strftime('%Y-%m-%d %H:%M:%S') for t in minor_tick_times]

    for i in range(1, num_rows + 1):
        fig.update_xaxes(
            type="date",
            range=[start_utc_str, end_utc_str],
            tickmode="array" if (is_encounter and not DYNAMIC_TICKS) else "auto", 
            tickvals=tick_vals_std if (is_encounter and not DYNAMIC_TICKS) else None, 
            showline=True, linecolor="black", 
            ticks="inside", ticklen=4, tickwidth=1, tickcolor="black", 
            minor=dict(tickvals=minor_vals_std, ticks="inside", ticklen=2, tickcolor="gray", tickwidth=1) if not DYNAMIC_TICKS else None, 
            row=i, col=1
        )

    if DYNAMIC_TICKS:
        fig.update_xaxes(title_text="Time (UTC)", title_font=dict(size=14, color="black"), row=num_rows, col=1)
        return fig

    # Build Ephemeris Multi-line Tick Labels
    eph_funcs = dp.load_ephemeris()
    time_fmt, top_label = ('%m-%d', 'Date') if is_encounter else ('%H:%M', 'Time')
    tick_texts = []
    
    for t in tick_times:
        t_unix = (t - datetime(1970, 1, 1)).total_seconds() 
        if eph_funcs and 'HG_LON' in eph_funcs:
            lbl = f"{t.strftime(time_fmt)}<br>{eph_funcs['HG_LON'](t_unix):.1f}<br>{eph_funcs['HG_LAT'](t_unix):.1f}<br>{eph_funcs['RAD_AU'](t_unix):.3f}"
        else:
            lbl = f"{t.strftime(time_fmt)}<br>--<br>--<br>--"
        tick_texts.append(lbl)

    # Re-bind tickmode, tickvals, and ticktext explicitly to the bottom master axis
    fig.update_xaxes(
        tickmode="array",
        tickvals=tick_vals_std,
        ticktext=tick_texts, 
        tickfont=dict(family="monospace", size=10), 
        row=num_rows, col=1
    )
    
    fig.add_annotation(
        text=f"<b>{top_label}</b><br><b>CARR_LON</b><br><b>CARR_LAT</b><br><b>SC_R_AU</b>",
        xref="paper", yref="paper", x=0, xanchor="right", xshift=-40, y=0, yanchor="top", yshift=-10,  
        showarrow=False, align="right", font=dict(family="monospace", size=10, color="black")
    )
    return fig

# =============================================================================
# PLOT TYPE 1: Standard Wave Analysis (6-Panel)
# =============================================================================
def plot_waveAnalysis(t_b_raw, t_fft_raw, data, freqs, title_str, start_dt, end_dt, t_br=None, val_br=None):
    is_encounter = (end_dt - start_dt).days > 1
    start_unix = start_dt.replace(tzinfo=timezone.utc).timestamp()
    end_unix = end_dt.replace(tzinfo=timezone.utc).timestamp()

    if t_b_raw is not None and len(t_b_raw) > 0:
        valid_b = np.isfinite(t_b_raw)  
        if np.any(valid_b):
            t_b_raw = t_b_raw[valid_b]
            b_mask = (t_b_raw >= start_unix) & (t_b_raw <= end_unix)
            t_b_raw = t_b_raw[b_mask]
            for k in ['Bn', 'Bp', 'Bq']:
                if k in data and data[k] is not None and len(data[k]) > 0:
                    data[k] = data[k][valid_b][b_mask]

    if t_fft_raw is not None and len(t_fft_raw) > 0:
        valid_fft = np.isfinite(t_fft_raw)
        if np.any(valid_fft):
            t_fft_raw = t_fft_raw[valid_fft]
            fft_mask = (t_fft_raw >= start_unix) & (t_fft_raw <= end_unix)
            t_fft_raw = t_fft_raw[fft_mask]
            for k in ['B_power_perp', 'S_Theta', 'ellipticity_b', 'coherency_b', 'wave_normal_b']:
                if k in data and data[k] is not None and len(data[k]) > 0:
                    data[k] = data[k][valid_fft]
                    data[k] = data[k][fft_mask, :] if data[k].ndim == 2 else data[k][fft_mask]

    if t_br is not None and val_br is not None and len(t_br) > 0:
        t_br_unix = cdflib.cdfepoch.unixtime(t_br)
        br_mask = (t_br_unix >= start_unix) & (t_br_unix <= end_unix)
        t_br_unix_cropped = t_br_unix[br_mask]
        val_br = val_br[br_mask]
        t_br_unix_cropped, [val_br] = dp.inject_data_gaps(t_br_unix_cropped, [val_br], is_2d=False)
        t_br_web = [datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None) for ts in t_br_unix_cropped]
    else:
        t_br_web = None

    if t_b_raw is None or len(t_b_raw) == 0 or t_fft_raw is None or len(t_fft_raw) == 0:
        return None

    target_1d = 8000 if is_encounter else dp.MAX_1D_POINTS
    target_2d = 6000 if is_encounter else dp.MAX_2D_POINTS

    t_b_dec, bn_web = dp.downsample_1d(t_b_raw, data['Bn'], target_1d)
    _, bp_web = dp.downsample_1d(t_b_raw, data['Bp'], target_1d)
    _, bq_web = dp.downsample_1d(t_b_raw, data['Bq'], target_1d)
    
    t_fft_dec, b_power_web = dp.downsample_2d(t_fft_raw, data['B_power_perp'], target_2d)
    _, stheta_web = dp.downsample_2d(t_fft_raw, data['S_Theta'], target_2d)
    _, ellipticity_web = dp.downsample_2d(t_fft_raw, data['ellipticity_b'], target_2d)
    _, coherency_web = dp.downsample_2d(t_fft_raw, data['coherency_b'], target_2d)
    _, wavenormal_web = dp.downsample_2d(t_fft_raw, data['wave_normal_b'], target_2d)
    stheta_web[stheta_web < 0] = np.nan

    t_b_dec, [bn_web, bp_web, bq_web] = dp.inject_data_gaps(t_b_dec, [bn_web, bp_web, bq_web], is_2d=False)
    t_fft_dec, [b_power_web, stheta_web, ellipticity_web, coherency_web, wavenormal_web] = dp.inject_data_gaps(
        t_fft_dec, [b_power_web, stheta_web, ellipticity_web, coherency_web, wavenormal_web], is_2d=True
    )

    t_b_web = [datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None) for ts in t_b_dec]
    t_fft_web = [datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None) for ts in t_fft_dec]

    t_b_plotly = dp.to_plotly_time(t_b_web)
    t_fft_plotly = dp.to_plotly_time(t_fft_web)
    t_br_plotly = dp.to_plotly_time(t_br_web)
    
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.008)

    if t_b_plotly is not None:
        eph_b = dp.get_ephemeris_hover_data(t_b_web)
        fig.add_trace(go.Scatter(x=t_b_plotly, y=bn_web, name='Bn', line=dict(color='black', width=1), customdata=eph_b, hovertemplate='B||: %{y:.1f} nT<extra></extra>'), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_b_plotly, y=bp_web, name='Bp', line=dict(color='red', width=1), hovertemplate='B&#8869;1: %{y:.1f} nT<extra></extra>'), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_b_plotly, y=bq_web, name='Bq', line=dict(color='blue', width=1), hovertemplate='B&#8869;2: %{y:.1f} nT<extra></extra>'), row=1, col=1)

    if t_fft_plotly is not None:
        eph_fft = [[eph] for eph in dp.get_ephemeris_hover_data(t_fft_web)]
        fig.add_trace(go.Scatter(x=t_fft_plotly, y=np.full(len(t_fft_plotly), -9999), mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False, customdata=eph_fft, hovertemplate='<b>%{customdata[0]}</b><extra></extra>'), row=1, col=1)

    if t_br_plotly is not None and len(t_br_plotly) > 0:
        fig.add_trace(go.Scatter(x=t_br_plotly, y=val_br, name='Br', opacity=0.6, line=dict(color='#cc5500', width=2), hovertemplate='Br: %{y:.1f} nT<extra></extra>'), row=1, col=1)
        
    b_hc = np.concatenate([bn_web, bp_web, bq_web])
    b_hc_valid = b_hc[np.isfinite(b_hc)]
    ymin, ymax = (np.nanpercentile(b_hc_valid, [1, 99]) if len(b_hc_valid) > 0 else (-10, 10))
    pad = (ymax - ymin) * 0.1 if ymax > ymin else 10
    fig.update_yaxes(range=[ymin-pad, ymax+pad], title_text="B (nT)", ticks="outside", ticklen=5, tickwidth=1, tickcolor="black", row=1, col=1)

    cb_len = 0.16
    psd_zmax = 3 if is_encounter else 2
    psd_tickvals = list(range(-2, psd_zmax + 1))
    psd_ticktext = [f"10<sup>{val}</sup>" for val in psd_tickvals]

    if t_fft_plotly is not None:
        fig.add_trace(go.Heatmap(x=t_fft_plotly, y=freqs, z=np.log10(b_power_web.T), customdata=b_power_web.T, colorscale='turbo', zmin=-2, zmax=psd_zmax, colorbar=dict(len=cb_len, y=0.752, tickvals=psd_tickvals, ticktext=psd_ticktext), hoverongaps=False, hovertemplate='Freq: %{y:.2f} Hz<br>Wave Power: %{customdata:.2e} nT^2/Hz<extra></extra>'), row=2, col=1)
        fig.add_trace(go.Heatmap(x=t_fft_plotly, y=freqs, z=stheta_web.T,      colorscale='RdBu_r',  zmin=0, zmax=180, colorbar=dict(len=cb_len, y=0.584), hoverongaps=False, hovertemplate='Freq: %{y:.2f} Hz<br>Poynting Angle: %{z:.2f}°<extra></extra>'), row=3, col=1)
        fig.add_trace(go.Heatmap(x=t_fft_plotly, y=freqs, z=ellipticity_web.T, colorscale='RdBu_r',  zmin=-1, zmax=1,  colorbar=dict(len=cb_len, y=0.416), hoverongaps=False, hovertemplate='Freq: %{y:.2f} Hz<br>Ellipticity: %{z:.2f}<extra></extra>'), row=4, col=1)
        fig.add_trace(go.Heatmap(x=t_fft_plotly, y=freqs, z=coherency_web.T,   colorscale='Greys',   zmin=0, zmax=1,   colorbar=dict(len=cb_len, y=0.248), hoverongaps=False, hovertemplate='Freq: %{y:.2f} Hz<br>Coherency: %{z:.2f}<extra></extra>'), row=5, col=1)
        fig.add_trace(go.Heatmap(x=t_fft_plotly, y=freqs, z=wavenormal_web.T,  colorscale='Greys_r', zmin=0, zmax=90,  colorbar=dict(len=cb_len, y=0.080), hoverongaps=False, hovertemplate='Freq: %{y:.2f} Hz<br>Wave Normal: %{z:.2f}°<extra></extra>'), row=6, col=1)

    if freqs is not None and len(freqs) > 0:
        f_max = np.nanmax(freqs)
        f_min = np.nanmin(freqs[freqs > 0]) if np.any(freqs > 0) else 0.1
        fig.update_yaxes(type="log", range=[np.log10(f_min), np.log10(f_max)], title_text="Freq<br>(Hz)", ticks="outside", row=2, col=1)
        for i in range(3, 7): fig.update_yaxes(type="log", matches="y2", title_text="Freq<br>(Hz)", ticks="outside", row=i, col=1)

    tick_times, minor_tick_times = [], []
    curr = start_dt.replace(hour=0, minute=0, second=0) if is_encounter else start_dt.replace(minute=0, second=0)
    while curr <= end_dt:
        tick_times.append(curr); curr += timedelta(days=1) if is_encounter else timedelta(hours=2)
    curr_minor = start_dt.replace(hour=0, minute=0, second=0) if is_encounter else start_dt.replace(minute=0, second=0)
    while curr_minor <= end_dt:
        minor_tick_times.append(curr_minor); curr_minor += timedelta(hours=6) if is_encounter else timedelta(minutes=15)

    fig = format_and_apply_xaxes(fig, start_dt, end_dt, is_encounter, 6, tick_times, minor_tick_times)
    fig.update_layout(title=title_str, hovermode="x unified", template="plotly_white", autosize=True, margin=dict(l=120, r=180, t=50, b=90), height=1050)
    
    cb_labels = [(0.752, "Wave Power"), (0.584, "Poynting Angle"), (0.416, "Ellipticity"), (0.248, "Coherency"), (0.080, "Wave Normal<br>(degrees)")]
    for y_pos, text in cb_labels:
        fig.add_annotation(text=text, xref="paper", yref="paper", x=1.08, y=y_pos, textangle=90, showarrow=False, xanchor="left", yanchor="middle", font=dict(size=14, color="black"))
    return fig

# =============================================================================
# PLOT TYPE 2: Plasma Moments (5-Panel)
# =============================================================================
def plot_plasma(start_dt, end_dt, title_str, span_energy_idx=10, **kwargs):
    d = dp.extract_plasma_data(start_dt, end_dt, span_energy_idx=span_energy_idx)
    if d is None or not any(v is not None for v in d.values()): return None

    # Dynamically extract energy value from metadata/data dict
    e_val = d.get("span_energy_ev")
    e_str = f"({e_val:.1f} eV)" if e_val is not None else "(476.0 eV)"

    t_mag_plotly = dp.to_plotly_time(d["t_mag_dt"])
    t_spc_plotly = dp.to_plotly_time(d["t_spc_dt"])
    t_spi_plotly = dp.to_plotly_time(d["t_spi_dt"])
    t_valf_plotly = dp.to_plotly_time(d["t_valf_dt"])
    t_lfr_plotly = dp.to_plotly_time(d["t_lfr_dt"])
    t_span_e_plotly = dp.to_plotly_time(d["t_span_e_dt"])

    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.005)

    def get_range(arrays, is_log=False):
        valid_arrays = [np.atleast_1d(a) for a in arrays if a is not None and len(np.atleast_1d(a)) > 0]
        if not valid_arrays: return [0, 1]
        valid = np.concatenate(valid_arrays)
        valid = valid[np.isfinite(valid)]
        if is_log: valid = valid[valid > 0]
        if len(valid) == 0: return [0, 1]
        vmin, vmax = np.nanpercentile(valid, [0.5, 99.5])
        return [np.log10(vmin) - 0.1, np.log10(vmax) + 0.1] if is_log else [vmin - (vmax-vmin)*0.1, vmax + (vmax-vmin)*0.1]

    eph_mag = dp.get_ephemeris_hover_data(d["t_mag_dt"]) if d["t_mag_dt"] else None
    eph_spc = dp.get_ephemeris_hover_data(d["t_spc_dt"]) if d["t_spc_dt"] else None
    eph_spi = dp.get_ephemeris_hover_data(d["t_spi_dt"]) if d["t_spi_dt"] else None
    eph_valf = dp.get_ephemeris_hover_data(d["t_valf_dt"]) if d["t_valf_dt"] else None
    eph_lfr = dp.get_ephemeris_hover_data(d["t_lfr_dt"]) if d["t_lfr_dt"] else None

    # Row 1: B-Field
    if t_mag_plotly:
        fig.add_trace(go.Scatter(x=t_mag_plotly, y=d["b_tot"], line=dict(color='black', width=1), name="|B|", customdata=eph_mag, hovertemplate='|B|: %{y:.1f} nT<extra></extra>'), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_mag_plotly, y=d["b_r"], line=dict(color='#d62728', width=0.5), name="B_R", customdata=eph_mag, hovertemplate='Br: %{y:.1f} nT<extra></extra>'), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_mag_plotly, y=d["b_t"], line=dict(color='#2ca02c', width=0.5), name="B_T", customdata=eph_mag, hovertemplate='Bt: %{y:.1f} nT<extra></extra>'), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_mag_plotly, y=d["b_n"], line=dict(color='#1f77b4', width=0.5), name="B_N", customdata=eph_mag, hovertemplate='Bn: %{y:.1f} nT<br><b>%{customdata}</b><extra></extra>'), row=1, col=1)
    fig.update_yaxes(title_text="B_RTN<br>(nT)", range=get_range([d["b_tot"], d["b_r"], d["b_t"], d["b_n"]]), row=1, col=1)

    # Row 2: Velocity
    if t_spc_plotly:
        if d.get("spc_vr_full") is not None: fig.add_trace(go.Scatter(x=t_spc_plotly, y=d["spc_vr_full"], line=dict(color='black', width=1), name="SPC Full Vr", customdata=eph_spc, hovertemplate='SPC Full Vr: %{y:.0f} km/s<extra></extra>'), row=2, col=1)
        if d.get("spc_vr_peak") is not None: fig.add_trace(go.Scatter(x=t_spc_plotly, y=d["spc_vr_peak"], line=dict(color='#888888', width=0.8), opacity=0.5, name="SPC Peak Vr", customdata=eph_spc, hovertemplate='SPC Peak Vr: %{y:.0f} km/s<extra></extra>'), row=2, col=1)
    if t_spi_plotly: fig.add_trace(go.Scatter(x=t_spi_plotly, y=d["spi_vr"], line=dict(color='#cc5500', width=1), name="SPANi v_R", customdata=eph_spi, hovertemplate='SPANi Vr: %{y:.0f} km/s<br><b>%{customdata}</b><extra></extra>'), row=2, col=1)
    if t_valf_plotly: fig.add_trace(go.Scatter(x=t_valf_plotly, y=d["valfven"], line=dict(color='blue', width=0.5), name="V_Alfven", customdata=eph_valf, hovertemplate='V_A: %{y:.0f} km/s<extra></extra>'), row=2, col=1)
    fig.update_yaxes(title_text="Velocity_R<br>(km/s)", range=get_range([d.get("spc_vr_full"), d.get("spc_vr_peak"), d["spi_vr"], d["valfven"]]), row=2, col=1)

    # Row 3: Density
    if t_spc_plotly:
        if d.get("spc_np_full") is not None: fig.add_trace(go.Scatter(x=t_spc_plotly, y=d["spc_np_full"], line=dict(color='black', width=1.2), name="SPC Full Np", customdata=eph_spc, hovertemplate='SPC Full Np: %{y:.1f} cm^-3<extra></extra>'), row=3, col=1)
        if d.get("spc_np_peak") is not None: fig.add_trace(go.Scatter(x=t_spc_plotly, y=d["spc_np_peak"], line=dict(color='#888888', width=0.8), opacity=0.4, name="SPC Peak Np", customdata=eph_spc, hovertemplate='SPC Peak Np: %{y:.1f} cm^-3<extra></extra>'), row=3, col=1)
    if t_spi_plotly: fig.add_trace(go.Scatter(x=t_spi_plotly, y=d["spi_np"], line=dict(color='#cc5500', width=1), name="SPANi", customdata=eph_spi, hovertemplate='SPANi Np: %{y:.1f} cm^-3<br><b>%{customdata}</b><extra></extra>'), row=3, col=1)
    if t_lfr_plotly: fig.add_trace(go.Scatter(x=t_lfr_plotly, y=d["np_lfr"], mode='markers', marker=dict(color='#2ca02c', size=3), opacity=0.8, name="LFR", customdata=eph_lfr, hovertemplate='LFR Ne: %{y:.1f} cm^-3<extra></extra>'), row=3, col=1)
    fig.update_yaxes(title_text="Density<br>(cm^-3)", type="log", range=get_range([d["np_lfr"], d["spi_np"], d.get("spc_np_full"), d.get("spc_np_peak")], is_log=True), row=3, col=1)

    # Row 4: Thermal Speed
    if t_spc_plotly:
        if d.get("spc_wp_full") is not None: fig.add_trace(go.Scatter(x=t_spc_plotly, y=d["spc_wp_full"], line=dict(color='black', width=1), name="SPC Full Wp", customdata=eph_spc, hovertemplate='SPC Full Wp: %{y:.1f} km/s<extra></extra>'), row=4, col=1)
        if d.get("spc_wp_peak") is not None: fig.add_trace(go.Scatter(x=t_spc_plotly, y=d["spc_wp_peak"], line=dict(color='#888888', width=0.8), opacity=0.4, name="SPC Peak Wp", customdata=eph_spc, hovertemplate='SPC Peak Wp: %{y:.1f} km/s<extra></extra>'), row=4, col=1)
    if t_spi_plotly: fig.add_trace(go.Scatter(x=t_spi_plotly, y=d["spi_wp"], line=dict(color='#cc5500', width=1), name="SPANi w_p", customdata=eph_spi, hovertemplate='SPANi Wp: %{y:.1f} km/s<br><b>%{customdata}</b><extra></extra>'), row=4, col=1)
    fig.update_yaxes(title_text="Thermal<br>Speed<br>(km/s)", type="log", range=get_range([d.get("spc_wp_full"), d.get("spc_wp_peak"), d["spi_wp"]], is_log=True), row=4, col=1)

    # Row 5: SPANe PAD
    pad_dec = d["pad_dec"]
    if t_span_e_plotly and pad_dec is not None:
        pa_bins = np.linspace(0, 180, pad_dec.shape[1])
        pad_mat = pad_dec.T
        pad_log = np.where(pad_mat > 0, np.log10(np.where(pad_mat > 0, pad_mat, 1)), np.nan)

        domain = fig.layout.yaxis5.domain
        cb_height = domain[1] - domain[0]
        cb_center = (domain[0] + domain[1]) / 2.0

        fig.add_trace(go.Heatmap(
            x=t_span_e_plotly, y=pa_bins, z=pad_log, customdata=pad_mat, 
            colorscale='turbo', zmin=6, zmax=11, hoverongaps=False, 
            hovertemplate='PA: %{y:.0f}°<br>Eflux: %{customdata:.2e}<extra></extra>',
            colorbar=dict(
                len=cb_height, 
                y=cb_center, 
                yanchor='middle',
                x=1.005,
                xanchor='left',
                thickness=12, 
                tickvals=[6, 7, 8, 9, 10, 11], 
                ticktext=['10⁶', '10⁷', '10⁸', '10⁹', '10¹⁰', '10¹¹']
            )
        ), row=5, col=1)
        fig.update_yaxes(
            range=[0, 180], 
            tickvals=[0, 45, 90, 135, 180], 
            title_text=f"SPANe PAD<br>{e_str}", 
            row=5, col=1
        )
    else:
        fig.update_yaxes(range=[0, 180], tickvals=[0, 45, 90, 135, 180], title_text="SPANe PAD", row=5, col=1)

    is_encounter = (end_dt - start_dt).days > 1
    tick_times, minor_tick_times = [], []
    curr = start_dt.replace(hour=0, minute=0, second=0) if is_encounter else start_dt.replace(minute=0, second=0)
    while curr <= end_dt:
        tick_times.append(curr); curr += timedelta(days=1) if is_encounter else timedelta(hours=2)
    curr_minor = start_dt.replace(hour=0, minute=0, second=0) if is_encounter else start_dt.replace(minute=0, second=0)
    while curr_minor <= end_dt:
        minor_tick_times.append(curr_minor); curr_minor += timedelta(hours=6) if is_encounter else timedelta(minutes=15)
        
    fig = format_and_apply_xaxes(fig, start_dt, end_dt, is_encounter, 5, tick_times, minor_tick_times)
    fig.update_layout(title=title_str, hovermode="x unified", template="plotly_white", autosize=True, margin=dict(l=120, r=180, t=50, b=90), height=750, showlegend=False)

    centers = [sum(fig.layout[f'yaxis{i}' if i > 1 else 'yaxis'].domain) / 2.0 for i in range(1, 6)]
    lf = dict(size=10, family="Arial")

    fig.add_annotation(text="<span style='color:black'>|B|</span><br><span style='color:#d62728'>B_R</span><br><span style='color:#2ca02c'>B_T</span><br><span style='color:#1f77b4'>B_N</span>", xref="paper", yref="paper", x=1.015, y=centers[0], showarrow=False, align="left", xanchor="left", yanchor="middle", font=lf)
    fig.add_annotation(text="<span style='color:black'>SPC Full</span><br><span style='color:#888888'>SPC Peak</span><br><span style='color:#cc5500'>SPANi</span><br><span style='color:blue'>V_A</span>", xref="paper", yref="paper", x=1.015, y=centers[1], showarrow=False, align="left", xanchor="left", yanchor="middle", font=lf)
    fig.add_annotation(text="<span style='color:black'>SPC Full</span><br><span style='color:#888888'>SPC Peak</span><br><span style='color:#cc5500'>SPANi</span><br><span style='color:#2ca02c'>LFR</span>", xref="paper", yref="paper", x=1.015, y=centers[2], showarrow=False, align="left", xanchor="left", yanchor="middle", font=lf)
    fig.add_annotation(text="<span style='color:black'>SPC Full</span><br><span style='color:#888888'>SPC Peak</span><br><span style='color:#cc5500'>SPANi</span>", xref="paper", yref="paper", x=1.015, y=centers[3], showarrow=False, align="left", xanchor="left", yanchor="middle", font=lf)
    # Dynamic Right-Side Colorbar Annotation
    fig.add_annotation(
        text=f"Eflux vs PA-E<br>{e_str}", 
        xref="paper", yref="paper", 
        x=1.09, y=centers[4], 
        textangle=90, showarrow=False, 
        xanchor="left", yanchor="middle", 
        font=dict(size=11, color="black")
    )

    return fig

# =============================================================================
# PLOT TYPE 3: SPC L2 Spectra (6-Panel)
# =============================================================================
def plot_spc(start_dt, end_dt, title_str, **kwargs):
    d = dp.extract_spc_l2_data(start_dt, end_dt)
    if d is None: return None

    t_1d_plotly = dp.to_plotly_time(d["t_spc_dt"])
    t_2d_plotly = t_1d_plotly
    master_vz = d["master_vz"]

    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.015)
    centers = [sum(fig.layout[f'yaxis{i}' if i > 1 else 'yaxis'].domain) / 2.0 for i in range(1, 7)]
    cb_len = 0.14 
    cb_tickvals = [0, 1, 2]
    cb_ticktext = ['10⁰', '10¹', '10²']

    if d["flux"] is not None:
        fig.add_trace(go.Heatmap(x=t_2d_plotly, y=master_vz, z=np.log10(np.where(d["flux"].T > 0, d["flux"].T, np.nan)), customdata=d["flux"].T, colorscale='turbo', colorbar=dict(len=cb_len, y=centers[0], tickvals=cb_tickvals, ticktext=cb_ticktext), hoverongaps=False, hovertemplate='Vz: %{y:.0f} km/s<br>Flux Dens: %{customdata:.2f}<extra></extra>'), row=1, col=1)
    
    if t_1d_plotly:
        eph_1d = dp.get_ephemeris_hover_data(d["t_spc_dt"])
        fig.add_trace(go.Scatter(x=t_1d_plotly, y=d["azimuth"], line=dict(color='#cc5500', width=1), name="Azimuth", customdata=eph_1d, hovertemplate='Az: %{y:.1f}°<extra></extra>'), row=2, col=1)
        fig.add_trace(go.Scatter(x=t_1d_plotly, y=d["elevation"], line=dict(color='#00a86b', width=1), name="Elevation", hovertemplate='El: %{y:.1f}°<extra></extra>'), row=2, col=1)

    for idx, (cur_arr, c_name) in enumerate([(d["a_current"], 'A'), (d["b_current"], 'B'), (d["c_current"], 'C'), (d["d_current"], 'D')], start=3):
        if cur_arr is not None:
            fig.add_trace(go.Heatmap(x=t_2d_plotly, y=master_vz, z=np.log10(np.where(cur_arr.T > 0, cur_arr.T, np.nan)), customdata=cur_arr.T, colorscale='turbo', zmin=0, zmax=2, colorbar=dict(len=cb_len, y=centers[idx-1], tickvals=cb_tickvals, ticktext=cb_ticktext), hoverongaps=False, hovertemplate=f'Vz: %{{y:.0f}} km/s<br>{c_name}: %{{customdata:.1f}} pA<extra></extra>'), row=idx, col=1)

    fig.update_yaxes(range=[150, 900], title_text="V_z<br>(km/s)", row=1, col=1)
    fig.update_yaxes(title_text="flow_angle<br>(degrees)", range=[-30, 30], row=2, col=1)
    for i in range(3, 7): fig.update_yaxes(range=[150, 900], title_text="V_z<br>(km/s)", row=i, col=1)

    is_encounter = (end_dt - start_dt).days > 1
    tick_times, minor_tick_times = [], []
    curr = start_dt.replace(hour=0, minute=0, second=0) if is_encounter else start_dt.replace(minute=0, second=0)
    while curr <= end_dt:
        tick_times.append(curr); curr += timedelta(days=1) if is_encounter else timedelta(hours=2)
    curr_minor = start_dt.replace(hour=0, minute=0, second=0) if is_encounter else start_dt.replace(minute=0, second=0)
    while curr_minor <= end_dt:
        minor_tick_times.append(curr_minor); curr_minor += timedelta(hours=6) if is_encounter else timedelta(minutes=15)
        
    fig = format_and_apply_xaxes(fig, start_dt, end_dt, is_encounter, 6, tick_times, minor_tick_times)
    fig.update_layout(title=title_str, hovermode="x unified", template="plotly_white", autosize=True, margin=dict(l=120, r=180, t=50, b=90), height=700, showlegend=True)
    return fig

# =============================================================================
# PLOT TYPE 4: Hammerhead Occurrence (5-Panel)
# =============================================================================
def plot_hammerhead(start_dt, end_dt, title_str, ham_bin=5, span_energy_idx=10, **kwargs):
    f = dp.extract_fields_data(start_dt, end_dt)
    w = dp.extract_wave_data(start_dt, end_dt)
    h = dp.extract_hammerhead_data(start_dt, end_dt, ham_bin=ham_bin)
    s = dp.extract_spane_data(start_dt, end_dt, span_energy_idx=span_energy_idx)
    
    e_val = s.get("span_energy_ev")
    e_str = f"<br>({e_val:.1f} eV)" if e_val is not None else ""

    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.015)
    cb_len = 0.185

    if f["t_mag_dt"]:
        br_ratio = np.where(np.array(f["b_tot"]) > 0, np.array(f["b_r"]) / np.array(f["b_tot"]), np.nan)
        fig.add_trace(go.Scatter(x=dp.to_plotly_time(f["t_mag_dt"]), y=br_ratio, line=dict(color='black', width=1), name="Br/|B|", hovertemplate='Ratio: %{y:.2f}<extra></extra>'), row=1, col=1)
        fig.update_yaxes(range=[-1.1, 1.1], title_text="Br/|B|", row=1, col=1)

    if w["t_fft_raw"] is not None:
        t_wave_plotly = dp.to_plotly_time([datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None) for ts in w["t_fft_raw"]])
        wave_p = w["data"].get('B_power_perp')
        ellip = w["data"].get('ellipticity_b')
        if wave_p is not None:
            fig.add_trace(go.Heatmap(x=t_wave_plotly, y=w["freqs"], z=np.log10(np.where(wave_p.T > 0, wave_p.T, np.nan)), colorscale='turbo', zmin=-2, zmax=3, colorbar=dict(len=cb_len, y=0.703), hoverongaps=False, hovertemplate='Freq: %{y:.1f} Hz<extra></extra>'), row=2, col=1)
        if ellip is not None:
            fig.add_trace(go.Heatmap(x=t_wave_plotly, y=w["freqs"], z=ellip.T, colorscale='RdBu_r', zmin=-1, zmax=1, colorbar=dict(len=cb_len, y=0.500), hoverongaps=False, hovertemplate='Freq: %{y:.1f} Hz<extra></extra>'), row=3, col=1)

    if h["t_ham_dt"]:
        fig.add_trace(go.Scatter(x=dp.to_plotly_time(h["t_ham_dt"]), y=h["ham_counts"], mode='lines+markers', line=dict(color='black', width=1), marker=dict(size=3), name="Counts", hovertemplate='Counts: %{y}<extra></extra>'), row=4, col=1)
        fig.update_yaxes(title_text=f"Hammerhead<br>Occurrence<br>({ham_bin}m Bins)", row=4, col=1)

    if s["t_span_e_dt"] and s["pad_dec"] is not None:
        pa_bins = np.linspace(0, 180, s["pad_dec"].shape[1])
        pad_mat = s["pad_dec"].T
        fig.add_trace(go.Heatmap(x=dp.to_plotly_time(s["t_span_e_dt"]), y=pa_bins, z=np.log10(np.where(pad_mat > 0, pad_mat, np.nan)), colorscale='turbo', zmin=6, zmax=11, colorbar=dict(len=cb_len, y=0.094), hoverongaps=False, hovertemplate='PA: %{y:.0f}°<extra></extra>'), row=5, col=1)
        fig.update_yaxes(
            range=[0, 180], 
            tickvals=[0, 45, 90, 135, 180], 
            title_text=f"SPANe PAD{e_str}", 
            row=5, col=1
        )

    is_encounter = (end_dt - start_dt).days > 1
    tick_times, minor_tick_times = [], []
    curr = start_dt.replace(hour=0, minute=0, second=0) if is_encounter else start_dt.replace(minute=0, second=0)
    while curr <= end_dt:
        tick_times.append(curr); curr += timedelta(days=1) if is_encounter else timedelta(hours=2)
    curr_minor = start_dt.replace(hour=0, minute=0, second=0) if is_encounter else start_dt.replace(minute=0, second=0)
    while curr_minor <= end_dt:
        minor_tick_times.append(curr_minor); curr_minor += timedelta(hours=6) if is_encounter else timedelta(minutes=15)

    fig = format_and_apply_xaxes(fig, start_dt, end_dt, is_encounter, 5, tick_times, minor_tick_times)
    fig.update_layout(title=title_str, hovermode="x unified", template="plotly_white", autosize=True, margin=dict(l=120, r=150, t=50, b=90), height=750, showlegend=False)
    return fig

# =============================================================================
# PLOT TYPE 5: Merged SWEAP Dashboard (5-Panel Spatial)
# =============================================================================
def plot_mergedSWEAP(start_dt, end_dt, title_str, csv_path='your_data.csv', **kwargs):
    if not os.path.exists(csv_path): return None
    df = dp.load_smart_csv(csv_path)
    if 'Times' not in df.columns: return go.Figure()
    df = df.dropna(subset=['Times']).sort_values('Times')
    
    mask = (df['Times'] >= start_dt) & (df['Times'] <= end_dt)
    df_window = df.loc[mask].copy()
    if df_window.empty: return None

    density_safe = np.where(df_window['Np-Parker'] > 0, df_window['Np-Parker'], np.nan)
    v_alfven = 21.8 * df_window['B-Parker'] / np.sqrt(1.1 * density_safe)

    t_raw = (df_window['Times'] - pd.Timestamp("1970-01-01")).dt.total_seconds().values
    is_encounter = (end_dt - start_dt).days > 1
    target_1d = 8000 if is_encounter else dp.MAX_1D_POINTS
    
    t_dec, b_mag_dec = dp.downsample_1d(t_raw, df_window['B-Parker'].values, target_1d)
    _, b_r_dec = dp.downsample_1d(t_raw, df_window['Br-Parker'].values, target_1d)
    _, b_t_dec = dp.downsample_1d(t_raw, df_window['Bt-Parker'].values, target_1d)
    _, b_n_dec = dp.downsample_1d(t_raw, df_window['Bn-Parker'].values, target_1d)
    _, vr_dec = dp.downsample_1d(t_raw, df_window['Vpr-Parker'].values, target_1d)
    _, va_dec = dp.downsample_1d(t_raw, v_alfven, target_1d)
    _, np_dec = dp.downsample_1d(t_raw, df_window['Np-Parker'].values, target_1d)
    _, tp_dec = dp.downsample_1d(t_raw, df_window['Tp-Parker'].values, target_1d)

    t_gap, data_gapped = dp.inject_data_gaps(t_dec, [b_mag_dec, b_r_dec, b_t_dec, b_n_dec, vr_dec, va_dec, np_dec, tp_dec], gap_threshold=3600, is_2d=False)
    b_mag_dec, b_r_dec, b_t_dec, b_n_dec, vr_dec, va_dec, np_dec, tp_dec = data_gapped

    t_dt = [datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None) for ts in t_gap]
    t_plotly = dp.to_plotly_time(t_dt)

    lfr = dp.extract_lfr_data(start_dt, end_dt)
    t_lfr_plotly = dp.to_plotly_time(lfr["t_lfr_dt"]) if lfr["t_lfr_dt"] else None

    eph_funcs = dp.load_ephemeris()
    x_orb, y_orb = np.full(len(t_gap), np.nan), np.full(len(t_gap), np.nan)
    if eph_funcs and 'HG_LON' in eph_funcs and 'RAD_AU' in eph_funcs:
        for i, ts in enumerate(t_gap):
            if np.isfinite(ts):
                r_au = eph_funcs['RAD_AU'](ts)
                lon_deg = eph_funcs['HG_LON'](ts)
                x_orb[i] = r_au * 215.0 * np.cos(np.radians(lon_deg))
                y_orb[i] = r_au * 215.0 * np.sin(np.radians(lon_deg))

    fig = make_subplots(rows=5, cols=2, shared_xaxes=False, vertical_spacing=0.03, specs=[[{"colspan": 2}, None], [{"colspan": 2}, None], [{"colspan": 2}, None], [{"colspan": 2}, None], [{"type": "xy"}, {"type": "xy"}]], row_heights=[0.14, 0.14, 0.14, 0.14, 0.44])

    t_style = dict(mode='lines', showlegend=False)
    fig.add_trace(go.Scatter(x=t_plotly, y=b_mag_dec, line=dict(color='black', width=1.5), hovertemplate='|B|: %{y:.1f} nT<extra></extra>', **t_style), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_plotly, y=b_r_dec, line=dict(color='#d62728', width=1), **t_style), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_plotly, y=b_t_dec, line=dict(color='#2ca02c', width=1), **t_style), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_plotly, y=b_n_dec, line=dict(color='#1f77b4', width=1), hovertemplate='Bn: %{y:.1f} nT<extra></extra>', **t_style), row=1, col=1)

    fig.add_trace(go.Scatter(x=t_plotly, y=vr_dec, line=dict(color='black', width=1.5), hovertemplate='Vr: %{y:.0f} km/s<extra></extra>', **t_style), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_plotly, y=va_dec, line=dict(color='#1f77b4', width=1.2), hovertemplate='Va: %{y:.0f} km/s<extra></extra>', **t_style), row=2, col=1)

    if t_lfr_plotly is not None:
        fig.add_trace(go.Scatter(x=t_lfr_plotly, y=lfr["np_lfr"], mode='markers', marker=dict(color='#2ca02c', size=3), opacity=0.8, name="LFR", hovertemplate='LFR Ne: %{y:.1f} cm^-3<extra></extra>', showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=t_plotly, y=np_dec, line=dict(color='black', width=1.5), hovertemplate='Np: %{y:.1f} cm^-3<extra></extra>', **t_style), row=3, col=1)
    fig.add_trace(go.Scatter(x=t_plotly, y=tp_dec, line=dict(color='black', width=1.5), **t_style), row=4, col=1)

    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(color='gold', size=15, symbol='star'), name='Sun', showlegend=False, hoverinfo='skip'), row=5, col=1)
    fig.add_trace(go.Scatter(x=x_orb, y=y_orb, mode='lines', line=dict(color='gray', width=2), name='OrbitTrack', showlegend=False, hoverinfo='skip'), row=5, col=1)
    valid_idx = np.where(np.isfinite(x_orb))[0]
    sx, sy = (x_orb[valid_idx[0]], y_orb[valid_idx[0]]) if len(valid_idx) > 0 else (0, 0)
    fig.add_trace(go.Scatter(x=[sx], y=[sy], mode='markers', marker=dict(color='black', size=12, symbol='triangle-up'), name='SpacecraftCursor', showlegend=False, hoverinfo='skip'), row=5, col=1)

    fig.update_xaxes(title_text="X-Carr (R<sub>☉</sub>)", range=[-50, 50], row=5, col=1)
    fig.update_yaxes(title_text="Y-Carr (R<sub>☉</sub>)", range=[-50, 50], scaleanchor="x5", scaleratio=1, row=5, col=1)

    for r in range(1, 5): fig.update_xaxes(matches='x', row=r, col=1)
    for r in range(1, 4): fig.update_xaxes(showticklabels=False, row=r, col=1)

    tick_times, minor_tick_times = [], []
    curr = start_dt.replace(hour=0, minute=0, second=0) if is_encounter else start_dt.replace(minute=0, second=0)
    while curr <= end_dt:
        tick_times.append(curr); curr += timedelta(days=1) if is_encounter else timedelta(hours=2)
    curr_minor = start_dt.replace(hour=0, minute=0, second=0) if is_encounter else start_dt.replace(minute=0, second=0)
    while curr_minor <= end_dt:
        minor_tick_times.append(curr_minor); curr_minor += timedelta(hours=6) if is_encounter else timedelta(minutes=15)
        
    fig = format_and_apply_xaxes(fig, start_dt, end_dt, is_encounter, 4, tick_times, minor_tick_times)
    fig.update_layout(title=title_str, hovermode="x unified", template="plotly_white", autosize=True, margin=dict(l=100, r=100, t=50, b=80), height=1100)
    
    custom_js = """
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            var graph = document.getElementById('plotly-graph');
            var orbitIdx = graph.data.findIndex(t => t.name === 'OrbitTrack');
            var cursorIdx = graph.data.findIndex(t => t.name === 'SpacecraftCursor');
            if(orbitIdx !== -1 && cursorIdx !== -1) {
                graph.on('plotly_hover', function(data) {
                    var pt = data.points[0];
                    if (pt && pt.xaxis && pt.xaxis._id !== 'x5' && pt.xaxis._id !== 'x6') {
                        var idx = pt.pointIndex;
                        var new_x = graph.data[orbitIdx].x[idx];
                        var new_y = graph.data[orbitIdx].y[idx];
                        if (!isNaN(new_x) && new_x !== null) {
                            Plotly.restyle(graph, {'x': [[new_x]], 'y': [[new_y]]}, [cursorIdx]);
                        }
                    }
                });
            }
        });
    </script>
    """
    return fig, custom_js

# =============================================================================
# ROUTER & CLI BUILDERS
# =============================================================================
PLOT_ROUTER = {
    'waveAnalysis_daily':     plot_waveAnalysis,
    'waveAnalysis_encounter': plot_waveAnalysis,
    'hammerhead_daily':       plot_hammerhead,
    'hammerhead_encounter':   plot_hammerhead,
    'spc_daily':              plot_spc,            
    'spc_encounter':          plot_spc,
    'plasma_daily':           plot_plasma,
    'plasma_encounter':       plot_plasma,
    'mergedSWEAP_daily':      plot_mergedSWEAP,
    'mergedSWEAP_encounter':  plot_mergedSWEAP,
    'mergedBeta_daily':       plot_mergedSWEAP,
    'mergedBeta_encounter':   plot_mergedSWEAP
}

def generate_interactive_plot(start_dt, end_dt, plot_type, out_path, title_str, **kwargs):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] STARTING PLOT: {title_str}")
    plot_func = PLOT_ROUTER.get(plot_type)

    if 'waveAnalysis' in plot_type:
        w_data = dp.extract_wave_data(start_dt, end_dt)
        if w_data["t_b_raw"] is None: 
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ABORT: No wave data found.")
            return
        result = plot_func(w_data["t_b_raw"], w_data["t_fft_raw"], w_data["data"], w_data["freqs"], title_str, start_dt, end_dt)
    else:
        result = plot_func(start_dt, end_dt, title_str, **kwargs)

    custom_js = None
    if isinstance(result, tuple): fig, custom_js = result
    else: fig = result

    if fig is None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ABORT: Plot generation returned None.")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if custom_js:
        raw_html = fig.to_html(include_plotlyjs="cdn", full_html=True, div_id="plotly-graph")
        final_html = raw_html.replace('</body>\n</html>', custom_js + '\n</body>\n</html>')
        with open(out_path, 'w', encoding='utf-8') as f: f.write(final_html)
    else:
        fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
        
    print(f"[{datetime.now().strftime('%H:%M:%S')}] SUCCESS: Saved plot to {out_path}")
    
def export_mission_ephemeris_json(out_json_path):
    """Legacy wrapper: Exports mission ephemeris JSON for conductor compatibility."""
    eph_funcs = dp.load_ephemeris()
    if not eph_funcs or 'RAD_AU' not in eph_funcs: return

    start_dt = datetime(2018, 10, 1)
    end_dt = datetime(2026, 12, 31)
    t_dt, r_au, carr_lon, carr_lat = [], [], [], []
    curr = start_dt
    while curr <= end_dt:
        ts = curr.replace(tzinfo=timezone.utc).timestamp()
        t_dt.append(curr)
        r_au.append(round(float(eph_funcs['RAD_AU'](ts)), 4))
        carr_lon.append(round(float(eph_funcs['HG_LON'](ts)), 2) if 'HG_LON' in eph_funcs else None)
        carr_lat.append(round(float(eph_funcs['HG_LAT'](ts)), 2) if 'HG_LAT' in eph_funcs else None)
        curr += timedelta(hours=1)

    payload = {
        "encounter_dates": dp.ENCOUNTER_DATES,
        "times": dp.to_plotly_time(t_dt),
        "data": {"r_au": r_au, "carr_lon": carr_lon, "carr_lat": carr_lat}
    }
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, 'w') as f:
        json.dump(payload, f)

def build_daily(start_str, plot_type='waveAnalysis_daily', **kwargs):
    dt = datetime.strptime(start_str, '%Y-%m-%d')
    if dp.GLOBAL_MAX_DATE and dt > dp.GLOBAL_MAX_DATE: return
        
    base_type = plot_type.split('_')[0] 
    root_dir = dp.PLOT_ROOTS.get(base_type, f'{dp.DRIVE_ROOT}/Research/PSP/Misc/Plots/')
    out_dir = os.path.join(root_dir, base_type, 'Daily', dt.strftime('%Y'), dt.strftime('%m'))
    fname = f"psp_{base_type}_daily_{dt.strftime('%Y%m%d')}.html"
    out_path = os.path.join(out_dir, fname)

    if not OVERWRITE and os.path.exists(out_path):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] SKIPPED GENERATION: {fname} already exists.")
    else:
        generate_interactive_plot(dt, dt + timedelta(days=1), plot_type, out_path, f"Daily {base_type}: {dt.strftime('%Y-%m-%d')}", **kwargs)

    if MIRROR_PUB and ACCESS == 'team' and os.path.exists(out_path):
        pub_dir = out_dir.replace('/team/', '/pub/')
        os.makedirs(pub_dir, exist_ok=True)
        pub_path = os.path.join(pub_dir, fname)
        if OVERWRITE or not os.path.exists(pub_path):
            shutil.copy2(out_path, pub_path)

def build_encounter(enc_num, plot_type='waveAnalysis_encounter', **kwargs):
    if enc_num not in dp.ENCOUNTER_DATES: return
    start_str, end_str = dp.ENCOUNTER_DATES[enc_num]
    start_dt = datetime.strptime(start_str, '%Y-%m-%d')
    end_dt = datetime.strptime(end_str, '%Y-%m-%d') + timedelta(days=1)

    base_type = plot_type.split('_')[0] 
    root_dir = dp.PLOT_ROOTS.get(base_type, f'{dp.DRIVE_ROOT}/Research/PSP/Misc/Plots/')

    if not NO_PLOT:
        out_dir = os.path.join(root_dir, base_type, 'Encounter', f'E{enc_num}')
        fname = f"psp_{base_type}_enc_{enc_num}.html"
        out_path = os.path.join(out_dir, fname)

        if not OVERWRITE and os.path.exists(out_path):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] SKIPPED GENERATION: {fname} team plot already exists.")
        else:
            generate_interactive_plot(start_dt, end_dt, plot_type, out_path, f"Encounter {enc_num}: {start_str} to {end_str}", **kwargs)

        if MIRROR_PUB and ACCESS == 'team' and os.path.exists(out_path):
            pub_dir = out_dir.replace('/team/', '/pub/')
            os.makedirs(pub_dir, exist_ok=True)
            pub_path = os.path.join(pub_dir, fname)
            if OVERWRITE or not os.path.exists(pub_path):
                shutil.copy2(out_path, pub_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate interactive PSP WaveAnalysis and SWEAP plots.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="YYYY-MM-DD for daily plots")
    group.add_argument("--enc", type=int, help="Encounter number")

    parser.add_argument("--access", default='team', choices=['team', 'pub'])
    parser.add_argument("--type", default='waveAnalysis', choices=['waveAnalysis', 'hammerhead', 'spc', 'plasma', 'mergedSWEAP', 'mergedBeta'])
    parser.add_argument("--ham-bin", type=int, default=5)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--dynamic-ticks", action="store_true")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mirror-pub", action="store_true")

    # Legacy cron compatibility flags
    parser.add_argument("--json-out", action="store_true", help="Legacy flag: ignored")
    parser.add_argument("--no-plot", action="store_true", help="Legacy flag: ignored")
    return parser.parse_args()

def main():
    args = parse_args()
    global VERBOSE, DYNAMIC_TICKS, ACCESS, OVERWRITE, MIRROR_PUB
    VERBOSE, DYNAMIC_TICKS, ACCESS, OVERWRITE, MIRROR_PUB = args.verbose, args.dynamic_ticks, args.access, args.overwrite, args.mirror_pub
    
    dp.resolve_runtime_paths(args.access, verbose_flag=VERBOSE)
    
    if args.enc:
        build_encounter(args.enc, plot_type=f"{args.type}_encounter", ham_bin=args.ham_bin, csv_override=args.csv)
    else:
        build_daily(args.date, plot_type=f"{args.type}_daily", ham_bin=args.ham_bin, csv_override=args.csv)

if __name__ == "__main__":
    main()
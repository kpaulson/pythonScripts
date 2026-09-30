# Save as test_quadmesh.py and run: python test_quadmesh.py
import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

# Load sliced npz file
npz_path = "/mnt/sweaparc/sa-home/kpaulson/MyDrive/Research/PSP/SPAN/SPANi/SPANi_slicedVDF/2026/03/psp_swp_spi_slicedVDF_20260311.npz"
data = np.load(npz_path)

print("NPZ Keys:", list(data.keys()))
vx = data['p_vx_xz_native'][100].astype(np.float64)
vz = data['p_vz_xz_native'][100].astype(np.float64)
vdf = data['p_xz_native'][100].astype(np.float64)

print(f"Vx shape: {vx.shape}, Range: [{np.nanmin(vx):.1f}, {np.nanmax(vx):.1f}]")
print(f"Vz shape: {vz.shape}, Range: [{np.nanmin(vz):.1f}, {np.nanmax(vz):.1f}]")
print(f"VDF shape: {vdf.shape}, Non-zero count: {np.sum(vdf > 0)}")

def centers_to_nodes(C):
    top = 2 * C[0, :] - C[1, :]
    bot = 2 * C[-1, :] - C[-2, :]
    C_ext0 = np.vstack([top[np.newaxis, :], C, bot[np.newaxis, :]])
    left = 2 * C_ext0[:, 0] - C_ext0[:, 1]
    right = 2 * C_ext0[:, -1] - C_ext0[:, -2]
    C_ext = np.column_stack([left[:, np.newaxis], C_ext0, right[:, np.newaxis]])
    return 0.25 * (C_ext[:-1, :-1] + C_ext[:-1, 1:] + C_ext[1:, :-1] + C_ext[1:, 1:])

vx_nodes = centers_to_nodes(vx)
vz_nodes = centers_to_nodes(vz)
clean_vdf = np.nan_to_num(np.floor(np.maximum(vdf, 0.0)), nan=0.0)

print(f"Node shape: {vx_nodes.shape}, Clean VDF shape: {clean_vdf.shape}")

app = pg.mkQApp("QuadMesh Test")
win = pg.PlotWidget(title="QuadMesh Direct Test")
pmesh = pg.PColorMeshItem(edgecolors=pg.mkPen('#ff0000', width=0.8))
win.addItem(pmesh)

try:
    pmesh.setData(x=vx_nodes, y=vz_nodes, z=clean_vdf)
    print("[SUCCESS] PColorMeshItem.setData executed without exception!")
except Exception as e:
    print(f"[ERROR] PColorMeshItem failed: {e}")

win.show()
app.exec_()
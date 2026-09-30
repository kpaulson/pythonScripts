"""
test_quadmesh_fundamental.py
=============================
Minimal step-by-step diagnostic test script for QuadMesh rendering.

Test 1: Synthetic 2x2 grid (verifies PColorMeshItem backend support)
Test 2: Real SPAN-I data with PColorMeshItem
Test 3: Real SPAN-I data with Native Qt Polygons (guaranteed fallback)
"""

import os
import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# --- MASTER CONFIG IMPORT ---
SCRIPT_DIR = "/home/kpaulson/MyDrive/Software/Python/githubScripts_python" if os.name != 'nt' else "G:/My Drive/Software/Python/githubScripts_python"
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

try:
    import config
    DRIVE_ROOT = config.get_drive_path()
except ImportError:
    DRIVE_ROOT = ""

app = pg.mkQApp("Fundamental QuadMesh Diagnostic")

# =============================================================
# TEST 1: Synthetic 2x2 QuadMesh
# =============================================================
print("--- [TEST 1] Testing Synthetic 2x2 PColorMeshItem ---")
win1 = pg.PlotWidget(title="TEST 1: Synthetic 2x2 PColorMeshItem")
win1.resize(500, 400)
win1.setBackground('#2b2b2b')

# 3x3 nodes for 2x2 cells
x_syn = np.array([[0, 10, 20], [0, 10, 20], [0, 10, 20]], dtype=float)
y_syn = np.array([[0, 0, 0], [10, 10, 10], [20, 20, 20]], dtype=float)
z_syn = np.array([[1.0, 5.0], [3.0, 8.0]], dtype=float)

try:
    pmesh_syn = pg.PColorMeshItem(x=x_syn, y=y_syn, z=z_syn, edgecolors=pg.mkPen('w', width=2))
    pmesh_syn.setColorMap(pg.colormap.get('turbo'))
    win1.addItem(pmesh_syn)
    win1.setXRange(-5, 25)
    win1.setYRange(-5, 25)
    win1.show()
    print("  -> Synthetic 2x2 mesh executed without exception.")
except Exception as e:
    print(f"  [!] Synthetic 2x2 mesh failed: {e}")

# =============================================================
# TEST 2 & 3: Real SPAN-I Data
# =============================================================
date_str = "2026-03-11"
npz_path = os.path.join(DRIVE_ROOT, "Research", "PSP", "SPAN", "SPANi", "SPANi_slicedVDF", "2026", "03", f"psp_swp_spi_slicedVDF_{date_str.replace('-', '')}.npz")

if os.path.exists(npz_path):
    print(f"\n--- [TEST 2 & 3] Loading Real SPAN-I Data ({date_str}) ---")
    data = np.load(npz_path)
    idx = 100

    vx_center = np.nan_to_num(data['p_vx_xz_native'][idx], nan=0.0)
    vz_center = np.nan_to_num(data['p_vz_xz_native'][idx], nan=0.0)
    vdf_raw = np.nan_to_num(data['p_xz_native'][idx], nan=0.0)

    def centers_to_nodes(C):
        top = 2 * C[0, :] - C[1, :]
        bot = 2 * C[-1, :] - C[-2, :]
        C_ext0 = np.vstack([top[np.newaxis, :], C, bot[np.newaxis, :]])
        left = 2 * C_ext0[:, 0] - C_ext0[:, 1]
        right = 2 * C_ext0[:, -1] - C_ext0[:, -2]
        C_ext = np.column_stack([left[:, np.newaxis], C_ext0, right[:, np.newaxis]])
        return 0.25 * (C_ext[:-1, :-1] + C_ext[:-1, 1:] + C_ext[1:, :-1] + C_ext[1:, 1:])

    vx_nodes = centers_to_nodes(vx_center)
    vz_nodes = centers_to_nodes(vz_center)
    z_val = np.floor(np.maximum(vdf_raw, 0.0))

    cmap = pg.colormap.get('turbo')
    pos = np.linspace(0, 1, 9)
    colors = cmap.map(pos, mode='byte')

    # -------------------------------------------------------------
    # TEST 2: PColorMeshItem with Real Data
    # -------------------------------------------------------------
    win2 = pg.PlotWidget(title="TEST 2: Real Data PColorMeshItem")
    win2.resize(500, 400)
    win2.setBackground('#2b2b2b')

    try:
        quant_cmap = pg.ColorMap(pos, colors)
        pmesh_real = pg.PColorMeshItem(x=vx_nodes, y=vz_nodes, z=z_val, edgecolors=pg.mkPen('w', width=0.5))
        pmesh_real.setColorMap(quant_cmap)
        pmesh_real.setLevels((0.0, 8.0))
        win2.addItem(pmesh_real)
        win2.setXRange(np.min(vx_nodes), np.max(vx_nodes))
        win2.setYRange(np.min(vz_nodes), np.max(vz_nodes))
        win2.show()
        print("  -> Real data PColorMeshItem executed without exception.")
    except Exception as e:
        print(f"  [!] Real data PColorMeshItem failed: {e}")

    # -------------------------------------------------------------
    # TEST 3: Bulletproof Native Qt Vector Polygons (Fallback)
    # -------------------------------------------------------------
    win3 = pg.PlotWidget(title="TEST 3: Native Qt Vector Polygons (Guaranteed Fallback)")
    win3.resize(500, 400)
    win3.setBackground('#2b2b2b')

    n_theta, n_energy = vx_center.shape
    poly_count = 0
    for th in range(n_theta):
        for en in range(n_energy):
            v_val = z_val[th, en]
            if v_val >= 1.0:  # Only draw detected non-zero channels
                c_x = [vx_nodes[th, en], vx_nodes[th+1, en], vx_nodes[th+1, en+1], vx_nodes[th, en+1]]
                c_y = [vz_nodes[th, en], vz_nodes[th+1, en], vz_nodes[th+1, en+1], vz_nodes[th, en+1]]

                polygon = QtGui.QPolygonF([QtCore.QPointF(c_x[k], c_y[k]) for k in range(4)])
                poly_item = QtWidgets.QGraphicsPolygonItem(polygon)

                c_idx = int(np.clip(v_val, 0, 8))
                r, g, b, _ = colors[c_idx]
                poly_item.setBrush(QtGui.QBrush(QtGui.QColor(r, g, b)))
                poly_item.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 120), 0.5))
                win3.addItem(poly_item)
                poly_count += 1

    win3.setXRange(np.min(vx_nodes), np.max(vx_nodes))
    win3.setYRange(np.min(vz_nodes), np.max(vz_nodes))
    win3.show()
    print(f"  -> TEST 3 drew {poly_count} vector polygon trapezoids successfully!")

else:
    print(f"[!] Slice dataset not found at: {npz_path}")

sys.exit(app.exec_())
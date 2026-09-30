import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path

# =======================================================================
# 1. SETUP PATHS
# =======================================================================
input_folder = Path("G:/Shared drives/HelioSwarm/KP_Working/SIMION/fcp_prototype/potentialArrays/2026-03-25/")
output_folder = input_folder
output_folder.mkdir(parents=True, exist_ok=True)

# =======================================================================
# 2. IDENTIFY UNIQUE MODELS
# =======================================================================
all_csvs = list(input_folder.glob("*.csv"))
base_names = set(f.stem.split('_Z')[0] for f in all_csvs)

print(f"Found {len(base_names)} unique models to process.")

# =======================================================================
# 3. LOOP OVER EACH MODEL SEPARATELY
# =======================================================================
for base_name in base_names:
    print(f"\n--- Processing Model: {base_name} ---")
    
    slice_files = list(input_folder.glob(f"{base_name}_Z*.csv"))
    print(f"  Found {len(slice_files)} Z-slices.")
    
    z_indices, z_mm = [], []
    potential_slices, electrode_slices = [], []
    dx, dy, dz = 1.0, 1.0, 1.0 # Default fallbacks
    
    for file in slice_files:
        # 3A. READ HEADER METADATA (Z-pos and Scales)
        with open(file, 'r') as f:
            for line in f:
                if "Z_Index:" in line: 
                    z_indices.append(int(line.split(":")[1]))
                elif "Absolute_Z_Position_mm:" in line: 
                    z_mm.append(float(line.split(":")[1]))
                elif "# dx:" in line: 
                    dx = float(line.split(":")[1])
                elif "# dy:" in line: 
                    dy = float(line.split(":")[1])
                elif "# dz:" in line: 
                    dz = float(line.split(":")[1])
                elif not line.startswith("#"): 
                    break 
        
        # 3B. READ DATA & PIVOT
        df = pd.read_csv(file, comment='#')
        # Pivot turns the 1D CSV rows into a 2D Y-X matrix
        v_2d = df.pivot(index='y_idx', columns='x_idx', values='potential_v').values
        e_2d = df.pivot(index='y_idx', columns='x_idx', values='is_electrode').values
        
        potential_slices.append(v_2d)
        electrode_slices.append(e_2d)
        
        x_indices = np.sort(df['x_idx'].unique())
        y_indices = np.sort(df['y_idx'].unique())

    # 3C. SORT AND STACK INTO 3D ARRAY (Shape: Z, Y, X)
    sort_idx = np.argsort(z_indices)
    v_cube = np.array(potential_slices)[sort_idx]
    e_cube = np.array(electrode_slices)[sort_idx]
    z_coords = np.array(z_indices)[sort_idx]
    z_coords_mm = np.array(z_mm)[sort_idx]

    # =======================================================================
    # 4. CALCULATE ELECTRIC FIELD (V/mm)
    # =======================================================================
    # np.gradient returns [grad_Z, grad_Y, grad_X] based on v_cube shape (Z,Y,X)
    print(f"  Calculating E-Field with spacing: dx={dx}, dy={dy}, dz={dz}")
    grad_z, grad_y, grad_x = np.gradient(v_cube, dz, dy, dx)
    
    # E = -grad V
    ex_cube, ey_cube, ez_cube = -grad_x, -grad_y, -grad_z

    # =======================================================================
    # 5. BUILD XARRAY DATASET
    # =======================================================================
    ds = xr.Dataset(
        data_vars=dict(
            potential=(["z", "y", "x"], v_cube),
            is_electrode=(["z", "y", "x"], e_cube),
            Ex=(["z", "y", "x"], ex_cube),
            Ey=(["z", "y", "x"], ey_cube),
            Ez=(["z", "y", "x"], ez_cube),
        ),
        coords=dict(
            x=(["x"], x_indices),
            y=(["y"], y_indices),
            z=(["z"], z_coords),
            x_mm=(["x"], x_indices * dx),
            y_mm=(["y"], y_indices * dy),
            z_mm=(["z"], z_coords_mm),
        ),
        attrs=dict(
            description=f"SIMION Field Map: {base_name}",
            dx_mm=dx, dy_mm=dy, dz_mm=dz,
            units="Potential (V), E-Field (V/mm), Spacing (mm)"
        ),
    )

    # =======================================================================
    # 6. REORDER DIMENSIONS TO (X, Y, Z) AND SAVE
    # =======================================================================
    ds = ds.transpose("x", "y", "z")

    output_file = output_folder / f"{base_name}.nc"
    ds.to_netcdf(output_file)
    print(f"  -> Successfully saved: {output_file}")
    print(f"     Final Array Dimensions: {ds.potential.dims}")

print("\n--- All models successfully converted and stitched! ---")
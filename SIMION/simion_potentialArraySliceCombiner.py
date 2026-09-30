import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path

# =======================================================================
# 1. SETUP PATHS
# =======================================================================
input_folder = Path("C:/Users/KPaulson/OneDrive/OneDrive - Smithsonian Institution/SAO - Central Engineering - Project-HelioSwarm/01-HSFC-SCIENCE/SIMION/fcp_prototype/potentialArrays/OLD/")
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
    
    z_coords = []
    z_coords_mm = []
    potential_grids = []
    electrode_grids = []
    
    for file in slice_files:
        # 3A. READ THE CUSTOM HEADER FOR Z-METADATA
        z_idx, z_mm = None, None
        with open(file, 'r') as f:
            for line in f:
                if line.startswith("# Z_Index:"):
                    z_idx = int(line.split(":")[1].strip())
                elif line.startswith("# Absolute_Z_Position_mm:"):
                    z_mm = float(line.split(":")[1].strip())
                elif not line.startswith("#"):
                    break # Stop reading once we hit the data block
                    
        # 3B. READ AND PIVOT THE CSV DATA
        df = pd.read_csv(file, comment='#')
        V_2d = df.pivot(index='y_idx', columns='x_idx', values='potential_v').values
        E_2d = df.pivot(index='y_idx', columns='x_idx', values='is_electrode').values
        
        x_coords = np.sort(df['x_idx'].unique())
        y_coords = np.sort(df['y_idx'].unique())
        
        z_coords.append(z_idx)
        z_coords_mm.append(z_mm)
        potential_grids.append(V_2d)
        electrode_grids.append(E_2d)

    # 3C. SORT STACK BY Z-INDEX
    sort_idx = np.argsort(z_coords)
    z_coords = np.array(z_coords)[sort_idx]
    z_coords_mm = np.array(z_coords_mm)[sort_idx]
    potential_grids = np.array(potential_grids)[sort_idx]
    electrode_grids = np.array(electrode_grids)[sort_idx]

    # 3D. CALCULATE PHYSICAL SCALE (mm/gu)
    # Find any non-zero Z slice to reverse-engineer the scale factor
    scale = 1.0 
    for idx, pos in zip(z_coords, z_coords_mm):
        if idx != 0:
            scale = pos / idx
            break
            
    # Generate the physical X and Y arrays using the calculated scale
    x_coords_mm = x_coords * scale
    y_coords_mm = y_coords * scale

    # =======================================================================
    # 4. PACKAGE AND EXPORT TO NETCDF (DUAL COORDINATES)
    # =======================================================================
    ds = xr.Dataset(
        data_vars=dict(
            potential=(["z", "y", "x"], potential_grids),
            is_electrode=(["z", "y", "x"], electrode_grids),
        ),
        # Here we assign both the index coordinates AND the physical coordinates
        coords=dict(
            z=(["z"], z_coords),
            y=(["y"], y_coords),
            x=(["x"], x_coords),
            z_mm=(["z"], z_coords_mm),
            y_mm=(["y"], y_coords_mm),
            x_mm=(["x"], x_coords_mm),
        ),
        attrs=dict(
            description=f"SIMION Potential Array: {base_name}",
            scale_mm_per_gu=scale
        ),
    )

    output_file = output_folder / f"{base_name}.nc"
    ds.to_netcdf(output_file)
    print(f"  -> Successfully saved: {output_file}")

print("\n--- All models successfully converted! ---")
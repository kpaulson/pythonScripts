import cdflib
import numpy as np
import os

def write_wave_analysis_cdf(output_path, data_dict, global_attrs):
    """
    Writes the WaveAnalysis data to a CDF file.
    data_dict: Dictionary where key=var_name, value=NumPy array
    """
    
    # 1. Create the CDF object
    if os.path.exists(output_path):
        os.remove(output_path)
    
    cdf = cdflib.cdfwrite.CDF(output_path)

    # 2. Define Global Attributes (Version, Source, etc.)
    cdf.write_global_attrs(global_attrs)

    # 3. Define Variable Specifications and Attributes
    for var_name, data in data_dict.items():
        
        # Handle TT2000 Time specifically (int64)
        if 'time' in var_name.lower():
            var_type = cdflib.constants.CDF_TIME_TT2000
        else:
            var_type = cdflib.constants.CDF_DOUBLE

        # Define Variable Attributes (analogous to QDataSet properties)
        var_attrs = {
            'CATDESC': var_name,
            'UNITS': 'nT^2/Hz' if 'power' in var_name.lower() else 'degrees',
            'FILLVAL': -1e31,
            'DEPEND_0': 'Bfield_time' if 'time' not in var_name else '',
            'VAR_TYPE': 'data'
        }

        # Handle NaNs and Infs (This prevents the 'trash data' in the file)
        # We replace them with the standard CDF FillValue
        clean_data = np.copy(data)
        if clean_data.dtype.kind == 'f':  # Only for floats
            clean_data[~np.isfinite(clean_data)] = -1e31

        # Write to CDF
        cdf.write_var(
            {
                'Variable': var_name,
                'Data_Type': var_type,
                'Num_Elements': 1,
                'Rec_Vary': True,
                'Dim_Sizes': list(clean_data.shape[1:]) if clean_data.ndim > 1 else []
            },
            var_attrs=var_attrs,
            data=clean_data
        )

    cdf.close()
    print(f"File successfully written: {output_path}")
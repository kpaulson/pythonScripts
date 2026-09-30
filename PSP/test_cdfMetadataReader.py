import cdflib
import sys

def print_global_attrs(cdf_path):
    try:
        # Open the CDF file
        cdf = cdflib.CDF(cdf_path)
        
        # Fetch the global attributes dictionary
        global_attrs = cdf.globalattsget()
        
        print(f"--- Global Attributes for {cdf_path} ---")
        for attr_name, attr_value in global_attrs.items():
            # In cdflib, the value is usually stored at index 0 of the dictionary/array
            val = attr_value[0] if isinstance(attr_value, dict) or isinstance(attr_value, list) else attr_value
            print(f"{attr_name:<25}: {val}")
            
        cdf.close()
        
    except Exception as e:
        print(f"Error reading CDF: {e}")

if __name__ == "__main__":
    # You can pass the file path as an argument or hardcode it here
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "/home/kpaulson/MyDrive/Research/PSP/WaveAnalysis/WaveAnalysis_Files/v1.3/2026/03/PSP_WaveAnalysis_2026-03-11_0600_v1.3.cdf"
        
    print_global_attrs(file_path)
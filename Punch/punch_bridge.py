# punch_bridge.py
import sys
from punchbowl.data import read_punch
from autoplot import java2py, py2java

# Autoplot passes the URL via sys.argv or environment
url = sys.argv[1]

# Use PUNCH's official loader to handle the compression/VLA
# This returns a SunPy/NDCube object or similar
data_obj = read_punch(url)

# Extract the raw numpy array
# For L3 CTM files, this handles the HDU1 decompression automatically
image_data = data_obj.data

# Send it back to Autoplot
# This converts the numpy array to an Autoplot QDataSet
print(py2java(image_data))
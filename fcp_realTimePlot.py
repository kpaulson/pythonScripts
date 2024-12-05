import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import time
import pandas as pd
import shutil
import tempfile
import os
import pathlib
import os.path

# Change this to the correct data directory for real-time file, or set to '' and work from the directory
dataDir = 'C:/Users/solarcup/Desktop/data/'
dataDir = 'C:/Users/solarcup/Box/Helioswarm/05-HSFC-ENGINEERING/10-LAB NOTES TESTING/DEC-2024 lab test data/data/'
#print([f for f in pathlib.Path(dataDir).glob('data_*.csv')])
filename = max([f for f in pathlib.Path(dataDir).glob('data_*.csv')], 
               key=os.path.getctime)

print(filename)

def update_plot(freq):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
        shutil.copyfileobj(open(filename, 'rb'), temp_file)
    
    # Get the user-defined variables
    dataLength     = dataLength_var.get()
    config         = config_var.get()
    fix_yaxisRange = yaxisRange_var.get()

    # Read the last few lines of the temp file into a dataframe
    df = pd.read_csv(temp_file.name).tail(dataLength)
    
    # Define plot orientations
    plotOrder_topDownInTank = [1, 0, 2, 3]
    plotOrder_gainOrder = [0, 1, 2, 3]

    # Select the appropriate columns based on the configuration
    if config == 'Gain0 (all collectors)':
        cols = [0, 4, 8, 12]
        labels = ['A gain0', 'B gain0', 'C gain0', 'D gain0']
        plotOrder = plotOrder_topDownInTank
    elif config == 'Gain1 (all collectors)':
        cols = [1, 5, 9, 13]
        labels = ['A gain1', 'B gain1', 'C gain1', 'D gain1']
        plotOrder = plotOrder_topDownInTank
    elif config == 'Gain2 (all collectors)':
        cols = [2, 6, 10, 14]
        labels = ['A gain2', 'B gain2', 'C gain2', 'D gain2']
        plotOrder = plotOrder_topDownInTank
    elif config == 'Gain3 (all collectors)':
        cols = [3, 7, 11, 15]
        labels = ['A gain3', 'B gain3', 'C gain3', 'D gain3']
        plotOrder = plotOrder_topDownInTank
    elif config == 'AllGain (A)':
        cols = [0, 1, 2, 3]
        labels = ['A gain0', 'A gain1', 'A gain2', 'A gain3']
        plotOrder = plotOrder_gainOrder
    elif config == 'AllGain (B)':
        cols = [4, 5, 6, 7]
        labels = ['B gain0', 'B gain1', 'B gain2', 'B gain3']
        plotOrder = plotOrder_gainOrder
    elif config == 'AllGain (C)':
        cols = [8, 9, 10, 11]
        labels = ['C gain0', 'C gain1', 'C gain2', 'C gain3']
        plotOrder = plotOrder_gainOrder
    elif config == 'AllGain (D)':
        cols = [12, 13, 14, 15]
        labels = ['D gain0', 'D gain1', 'D gain2', 'D gain3']
        plotOrder = plotOrder_gainOrder

    # Isolate relevant columns and ensure numeric data types
    df_subset = df.iloc[:, cols]#.values.reshape(-1, 4)
    df_subset = df_subset.astype(float)

    # Clear the axes and plot the new data
    for ax in axes.flat:
        ax.clear()

    # Plot the data in a 2x2 grid
    for i in range(4):
        axes[i//2, i%2].plot(df_subset.iloc[:, plotOrder[i]])
        axes[i//2, i%2].set_xlabel('record #')
        axes[i//2, i%2].set_ylabel('%s (DN)'%labels[plotOrder[i]])
        if fix_yaxisRange:
            axes[i//2, i%2].set_ylim(0, 4096)

    # Draw the plot
    canvas.draw()
	
    # Delete the temporary file
    os.remove(temp_file.name)

    # Schedule the next update
    root.after(int(freq * 1000), update_plot, float(freq_var.get()))

# Create the main window
root = tk.Tk()
root.title("FCP Real-time Plot - ")
root.resizable(True, True)
root.rowconfigure(1, weight=1)
root.columnconfigure(0, weight=1)
root.columnconfigure(2, weight=1)
root.columnconfigure(4, weight=1)
root.columnconfigure(6, weight=1)
#root.columnconfigure(1, weight=1)
root.wm_state('zoomed')

# Create a Figure and Axes
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

# Embed the Matplotlib Figure in Tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
#canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
canvas.get_tk_widget().grid(row=1, column=0, columnspan=8, sticky="nsew")

# Create a dropdown menu for plot configuration
config_label = tk.Label(root, text="Plot Configuration:")
config_var   = tk.StringVar(value='Gain0 (all collectors)')
config_menu  = tk.OptionMenu(root, config_var, 'Gain0 (all collectors)', 'Gain1 (all collectors)', 'Gain2 (all collectors)', 'Gain3 (all collectors)', 'AllGain (A)', 'AllGain (B)', 'AllGain (C)', 'AllGain (D)')

# Create a dropdown menu for update frequency
freq_label = tk.Label(root, text="Update Frequency (s):")
freq_var   = tk.StringVar(value="1")
freq_menu  = tk.OptionMenu(root, freq_var, "0.25", "0.5", "1", "2")

# Create a dropdown menu for length of data
dataLength_label = tk.Label(root, text="Length of Dataset:")
dataLength_var   = tk.IntVar(value=200)
dataLength_menu  = tk.OptionMenu(root, dataLength_var, 50, 100, 200, 500)

# Create a dropdown menu to set axes
yaxisRange_label  = tk.Label(root, text="Set y-axes to full width:")
yaxisRange_var    = tk.BooleanVar()
yaxisRange_button = tk.Checkbutton(root, variable=yaxisRange_var)

# Grid layout
config_label.grid(row=0, column=0, sticky="ew")
config_menu.grid(row=0, column=1, sticky="ew")

freq_label.grid(row=0, column=2, sticky="ew")
freq_menu.grid(row=0, column=3, sticky="ew")

dataLength_label.grid(row=0, column=4, sticky="ew")
dataLength_menu.grid(row=0, column=5, sticky="ew")

yaxisRange_label.grid(row=0, column=6, sticky="ew")
yaxisRange_button.grid(row=0, column=7, sticky="ew")

# Start the update process
update_plot(1)

root.mainloop()
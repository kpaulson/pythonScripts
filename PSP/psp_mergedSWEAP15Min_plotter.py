import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Define Calculation Function ---
def calc_alfven_speed(b_mag_nt, density_cm3, mu=1.1):
    density_safe = np.where(density_cm3 > 0, density_cm3, np.nan)
    return 21.8 * b_mag_nt / np.sqrt(mu * density_safe)

# --- 2. Load Data ---
# Replace 'your_data.csv' with your actual file path
df = pd.read_csv('your_data.csv')

# Ensure 'Times' is a datetime object for proper plotting
df['Times'] = pd.to_datetime(df['Times'])

# Calculate Alfven Speed using Parker data
df['V_Alfven'] = calc_alfven_speed(df['B-Parker'], df['Np-Parker'])

# --- 3. Setup Plotly Subplots ---
# 4 rows, 1 column, with shared X-axis
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
    subplot_titles=(
        "Magnetic Field (Parker)", 
        "Radial Velocity vs Alfvén Speed", 
        "Proton Density", 
        "Proton Temperature"
    )
)

# --- 4. Add Traces ---

# Panel 1: Magnetic Field (B, Br, Bt, Bn)
fig.add_trace(go.Scatter(x=df['Times'], y=df['B-Parker'], mode='lines', name='|B|', line=dict(color='black')), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Times'], y=df['Br-Parker'], mode='lines', name='B_R', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Times'], y=df['Bt-Parker'], mode='lines', name='B_T', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Times'], y=df['Bn-Parker'], mode='lines', name='B_N', line=dict(color='blue')), row=1, col=1)

# Panel 2: Velocity & Alfvén Speed
fig.add_trace(go.Scatter(x=df['Times'], y=df['Vpr-Parker'], mode='lines', name='V_R (Proton)', line=dict(color='orange')), row=2, col=1)
fig.add_trace(go.Scatter(x=df['Times'], y=df['V_Alfven'], mode='lines', name='V_Alfven', line=dict(color='purple')), row=2, col=1)

# Panel 3: Density (Log Scale often preferred for space plasma density)
fig.add_trace(go.Scatter(x=df['Times'], y=df['Np-Parker'], mode='lines', name='N_p', line=dict(color='brown')), row=3, col=1)

# Panel 4: Temperature
fig.add_trace(go.Scatter(x=df['Times'], y=df['Tp-Parker'], mode='lines', name='T_p', line=dict(color='darkred')), row=4, col=1)


# --- 5. Formatting & Layout ---
fig.update_layout(
    title_text="Parker Solar Probe Encounter Data",
    height=900,  # Adjust overall height
    hovermode="x unified", # Shows all data for a specific time on hover
    template="plotly_white"
)

# Update Y-axes labels and types
fig.update_yaxes(title_text="B (nT)", row=1, col=1)
fig.update_yaxes(title_text="Velocity (km/s)", row=2, col=1)
fig.update_yaxes(title_text="Density (cm^-3)", type="log", row=3, col=1) # Using log scale for density
fig.update_yaxes(title_text="Temperature (eV/K)", row=4, col=1) # Update units based on your CSV's standard

fig.update_xaxes(title_text="Time (UTC)", row=4, col=1)

# --- 6. Export to HTML ---
output_filename = "PSP_Encounter_Plot.html"
fig.write_html(output_filename, include_plotlyjs="cdn")
print(f"Plot successfully saved to {output_filename}")
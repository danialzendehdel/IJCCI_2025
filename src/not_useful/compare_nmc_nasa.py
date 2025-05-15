import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from src.utils.config_reader import load_config

# Add the project root to the path to fix imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the data reader directly
from src.utils.NMC_data_reader.data_reader import load_matlab_data

# Paths
config_path = "/home/danial/Documents/Codes_new/N-IJCCI-BMS/src/configuration/config.yml"
config = load_config(config_path)
nasa_lookup_path = config.data.Nasa_data + "/lookup_tables"
nmc_data_path = config.data.matlab_data
nasa_npz_path = config.data.Nasa_data + "/battery_lookup_tables.npz"

# Create output directory for plots
output_dir = "/home/danial/Documents/Codes_new/N-IJCCI-BMS/battery_results/comparison"
os.makedirs(output_dir, exist_ok=True)

# 1. Load original NASA data from NPZ file
nasa_data = np.load(nasa_npz_path)
nasa_soc = nasa_data['soc_lp']
nasa_ocv = nasa_data['ocv_lp']

# 2. Load transformed NASA data from first cell
cell_paths = [f for f in os.listdir(nasa_lookup_path) if f.endswith('_lookup.csv') and f.startswith('cell_')]
cell_paths.sort(key=lambda x: int(x.split('_')[1]))
nasa_transformed_data = pd.read_csv(os.path.join(nasa_lookup_path, cell_paths[0]))
nasa_transformed_soc = nasa_transformed_data['SOC'].values
nasa_transformed_ocv = nasa_transformed_data['OCV'].values

# 3. Load NMC data if available
try:
    nmc_data = load_matlab_data(nmc_data_path)
    has_nmc_data = True
    print(f"Loaded NMC data from {nmc_data_path}")
except Exception as e:
    print(f"Could not load NMC data: {e}")
    has_nmc_data = False

# Create the comparison plot
plt.figure(figsize=(12, 8))

# Plot NASA original OCV curve
plt.plot(nasa_soc, nasa_ocv, 'b-', linewidth=2, label='NASA Original (LiCoO2)')

# Plot NASA transformed OCV curve (for NMC)
plt.plot(nasa_transformed_soc[::100], nasa_transformed_ocv[::100], 'r-', linewidth=2, 
         label='NASA Transformed for NMC')

# Plot NMC original data if available
if has_nmc_data:
    soc_values = nmc_data.get('SOC', [])
    ocv_values = nmc_data.get('OCV', [])
    
    if len(soc_values) > 0 and len(ocv_values) > 0:
        # Ensure soc_values is sorted for proper plotting
        sort_idx = np.argsort(soc_values)
        plt.plot(soc_values[sort_idx], ocv_values[sort_idx], 'g-', linewidth=2, 
                label='NMC Reference Data')

# Add a grid, labels, and title
plt.grid(True, alpha=0.3)
plt.xlabel('State of Charge (%)', fontsize=14)
plt.ylabel('Open Circuit Voltage (V)', fontsize=14)
plt.title('Comparison of NASA and NMC Battery SOC-OCV Curves', fontsize=16)
plt.legend(fontsize=12)

# Show voltage range differences with annotations
min_nasa = nasa_ocv.min()
max_nasa = nasa_ocv.max()
min_transformed = nasa_transformed_ocv.min()
max_transformed = nasa_transformed_ocv.max()

# Add text annotations highlighting the differences
plt.annotate(f'NASA LiCoO2 Voltage Range: {min_nasa:.2f}V - {max_nasa:.2f}V', 
             xy=(0.5, 0.05), xycoords='axes fraction', 
             bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.8))

plt.annotate(f'Transformed NMC Voltage Range: {min_transformed:.2f}V - {max_transformed:.2f}V', 
             xy=(0.5, 0.11), xycoords='axes fraction', 
             bbox=dict(boxstyle="round,pad=0.3", fc="lightpink", alpha=0.8))

# Add explanation of key differences
plt.figtext(0.5, 0.01, 
           "Key Differences in NMC vs LiCoO2:\n"
           "1. NMC has lower average voltage (~3.7V vs ~3.9V)\n"
           "2. NMC has more linear voltage profile in mid-SOC region\n"
           "3. NMC has less steep voltage drop at low SOC\n"
           "4. NMC exhibits characteristic plateau at high SOC",
           ha='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

# Save the plot
plt.tight_layout(rect=[0, 0.07, 1, 0.95])  # Adjust to make room for explanation text
plt.savefig(os.path.join(output_dir, 'nasa_nmc_ocv_comparison.png'), dpi=300, bbox_inches='tight')
print(f"Saved comparison plot to: {os.path.join(output_dir, 'nasa_nmc_ocv_comparison.png')}")

# Show and save a second plot with multiple transformed cells to demonstrate variations
plt.figure(figsize=(12, 8))

# Plot some sample cells to show variations
for i, cell_file in enumerate(cell_paths[:10]):  # Plot first 10 cells
    cell_data = pd.read_csv(os.path.join(nasa_lookup_path, cell_file))
    cell_soc = cell_data['SOC'].values
    cell_ocv = cell_data['OCV'].values
    cell_id = cell_file.split('_lookup')[0]
    plt.plot(cell_soc[::100], cell_ocv[::100], '-', linewidth=1, alpha=0.7, label=cell_id)

plt.grid(True, alpha=0.3)
plt.xlabel('State of Charge (%)', fontsize=14)
plt.ylabel('Open Circuit Voltage (V)', fontsize=14)
plt.title('Cell-to-Cell Variations in Transformed NMC Battery Models', fontsize=16)
plt.legend(fontsize=10, loc='lower right')

# Save the second plot
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'nmc_cell_variations.png'), dpi=300, bbox_inches='tight')
print(f"Saved cell variations plot to: {os.path.join(output_dir, 'nmc_cell_variations.png')}")

if __name__ == "__main__":
    print("Generated comparison plots for NASA and NMC battery characteristics") 
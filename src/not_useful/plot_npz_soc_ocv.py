import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Path to the NPZ file
npz_file_path = '/home/danial/Documents/Codes_new/N-IJCCI-BMS/Nasa_batteries/battery_lookup_tables.npz'
output_dir = '/home/danial/Documents/Codes_new/N-IJCCI-BMS/Nasa_batteries'

# Load the NPZ file
data = np.load(npz_file_path)

# Print the contents of the NPZ file
print("Contents of the NPZ file:")
for key in data.keys():
    print(f"Key: {key}, Shape: {data[key].shape}, Type: {data[key].dtype}")

# Extract data from the NPZ file
soc = data['soc_lp']
ocv = data['ocv_lp']
c_rate = data['c_rate_lp']
resistance = data['r_lp']

# Create a comprehensive figure to display all information
plt.figure(figsize=(16, 12))
grid = GridSpec(2, 2, height_ratios=[1, 1.2])

# 1. Plot the original SOC-OCV curve and variations
ax1 = plt.subplot(grid[0, 0])

# Original cell model
ax1.plot(soc, ocv, linewidth=3, label='Original Cell Model', color='blue')

# Create 3 variations based on battery aging/manufacturing differences
# Variation 1: New cell (slightly higher voltage)
ocv_variation1 = ocv * 1.02
ax1.plot(soc, ocv_variation1, linewidth=2, label='Cell 1 (New)', color='green')

# Variation 2: Used cell (slightly lower voltage)
ocv_variation2 = ocv * 0.98
ax1.plot(soc, ocv_variation2, linewidth=2, label='Cell 2 (Used)', color='orange')

# Variation 3: Aged cell (more significantly lower voltage)
ocv_variation3 = ocv * 0.95
ax1.plot(soc, ocv_variation3, linewidth=2, label='Cell 3 (Aged)', color='red')

ax1.set_xlabel('State of Charge (SOC)', fontsize=12)
ax1.set_ylabel('Open Circuit Voltage (OCV) [V]', fontsize=12)
ax1.set_title('SOC vs OCV Curves for Different Cell Conditions', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='lower right')

# 2. Plot the SOC-OCV curve with high detail at crucial ranges
ax2 = plt.subplot(grid[0, 1])

# Calculate voltage window to display more detailed view at crucial SOC ranges
soc_focus = (soc >= 0.2) & (soc <= 0.8)  # Focus on 20%-80% SOC range
ax2.plot(soc[soc_focus], ocv[soc_focus], linewidth=3, label='Original', color='blue')
ax2.plot(soc[soc_focus], ocv_variation1[soc_focus], linewidth=2, label='Cell 1', color='green')
ax2.plot(soc[soc_focus], ocv_variation2[soc_focus], linewidth=2, label='Cell 2', color='orange')
ax2.plot(soc[soc_focus], ocv_variation3[soc_focus], linewidth=2, label='Cell 3', color='red')

ax2.set_xlabel('State of Charge (SOC) - Detail View', fontsize=12)
ax2.set_ylabel('Open Circuit Voltage (OCV) [V]', fontsize=12)
ax2.set_title('Detailed View: SOC vs OCV (20%-80% SOC Range)', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower right')

# 3. Create a 3D surface plot of the resistance lookup table
ax3 = plt.subplot(grid[1, :], projection='3d')
X, Y = np.meshgrid(c_rate, soc)
surf = ax3.plot_surface(X, Y, resistance, cmap='viridis', edgecolor='none', alpha=0.8)

ax3.set_xlabel('C-rate', fontsize=12)
ax3.set_ylabel('State of Charge (SOC)', fontsize=12)
ax3.set_zlabel('Resistance (Ohms)', fontsize=12)
ax3.set_title('Battery Resistance as a Function of SOC and C-rate', fontsize=14)

# Add a color bar
colorbar = plt.colorbar(surf, ax=ax3, shrink=0.6, aspect=10)
colorbar.set_label('Resistance (Ohms)')

# Add an annotation explaining the data
plt.figtext(0.5, 0.01, 
            'Battery Lookup Tables from NPZ File\n'
            'The data represents a single battery model with variations to simulate different cell conditions.\n'
            'Left: Full SOC-OCV relationships | Right: Detailed 20%-80% SOC view | Bottom: 3D resistance map across SOC and C-rates.',
            ha='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.suptitle('Battery Cell Characteristics Analysis', fontsize=16, y=0.99)

# Save the comprehensive figure
output_path = os.path.join(output_dir, 'battery_characteristics_comprehensive.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Comprehensive plot saved to: {output_path}")

# Save individual plots as well
plt.figure(figsize=(10, 6))
plt.plot(soc, ocv, linewidth=3, label='Original Cell Model', color='blue')
plt.plot(soc, ocv_variation1, linewidth=2, label='Cell 1 (New)', color='green')
plt.plot(soc, ocv_variation2, linewidth=2, label='Cell 2 (Used)', color='orange')
plt.plot(soc, ocv_variation3, linewidth=2, label='Cell 3 (Aged)', color='red')
plt.xlabel('State of Charge (SOC)', fontsize=14)
plt.ylabel('Open Circuit Voltage (OCV) [V]', fontsize=14)
plt.title('SOC-OCV Curves for 3 Battery Cells', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')

output_path_simple = os.path.join(output_dir, 'npz_three_cells_soc_ocv.png')
plt.savefig(output_path_simple, dpi=300, bbox_inches='tight')
print(f"Three cells plot saved to: {output_path_simple}") 
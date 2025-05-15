import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Path to NASA batteries lookup tables
lookup_tables_dir = '/home/danial/Documents/Codes_new/N-IJCCI-BMS/Nasa_batteries/lookup_tables'
output_dir = '/home/danial/Documents/Codes_new/N-IJCCI-BMS/Nasa_batteries'

# Get all cell lookup CSV files in the directory (ensure we only get cell lookup files)
csv_files = []
for f in os.listdir(lookup_tables_dir):
    if f.endswith('_lookup.csv') and f.startswith('cell_'):
        try:
            # Verify we can extract a cell number
            cell_num = int(f.split('_')[1])
            csv_files.append(f)
        except (ValueError, IndexError):
            print(f"Skipping file with invalid format: {f}")

print(f"Found {len(csv_files)} valid cell files")

# Sort files by cell number for better visualization
csv_files.sort(key=lambda x: int(x.split('_')[1]))

# Create a figure with more space for the plot
plt.figure(figsize=(14, 10))

# Color map for different cells
colors = plt.cm.viridis(np.linspace(0, 1, len(csv_files)))

# Process each cell file
for i, file in enumerate(csv_files):
    file_path = os.path.join(lookup_tables_dir, file)
    cell_num = file.split('_')[1]  # Extract cell number from filename
    
    try:
        # Load the data - the CSV has headers
        data = pd.read_csv(file_path)
        
        # Print the first few column names to debug
        if i == 0:
            print(f"Columns in {file}: {data.columns.tolist()}")
        
        # Extract SoC and OCV using column names
        soc = data['SOC']
        ocv = data['OCV']
        
        # Plot with a slightly transparent line to see overlaps better
        # Only label every 50th cell to avoid legend clutter
        plt.plot(soc, ocv, color=colors[i], alpha=0.6, linewidth=0.8, 
                 label=f"Cell {cell_num}" if i % 50 == 0 else "")
    
    except Exception as e:
        print(f"Error processing file {file}: {e}")

# Add labels and title
plt.xlabel('State of Charge (SoC)', fontsize=16)
plt.ylabel('Open Circuit Voltage (OCV)', fontsize=16)
plt.title('SoC vs OCV Curves for NASA Battery Cells', fontsize=18, pad=20)
plt.grid(True, alpha=0.3)

# Adjust figure layout with more space for the legend
plt.subplots_adjust(bottom=0.15, right=0.85)

# Move legend outside the plot to avoid overlapping with the curves
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

# Add a clear border
plt.box(True)

# Save the figure with high resolution
output_path = os.path.join(output_dir, 'nasa_cells_soc_ocv.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Also save as SVG for vector graphics
plt.savefig(os.path.join(output_dir, 'nasa_cells_soc_ocv.svg'), format='svg', bbox_inches='tight')
print(f"SVG also saved to: {os.path.join(output_dir, 'nasa_cells_soc_ocv.svg')}")

# Save a version with no legend for a cleaner look
plt.legend().set_visible(False)
plt.savefig(os.path.join(output_dir, 'nasa_cells_soc_ocv_no_legend.png'), dpi=300, bbox_inches='tight')
print(f"Version without legend saved to: {os.path.join(output_dir, 'nasa_cells_soc_ocv_no_legend.png')}") 
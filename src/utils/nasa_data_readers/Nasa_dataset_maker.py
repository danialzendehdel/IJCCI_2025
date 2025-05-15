import os
import re
import numpy as np
from pathlib import Path
import scipy.io as sio
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt

class NasaDataReader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.cells = []
        self.cell_data = {}

    def _find_cells(self):
        root = Path(self.data_path)
        cells = list(root.rglob("*.mat"))
        print(f"Found {len(cells)} cells in {self.data_path}")
        return cells
    
    def _load_data(self):
        cells = self._find_cells()
        for cell in cells:
            cell_id = cell.stem  # e.g., B0005
            print(f"Processing cell: {cell_id}")
            
            self.cell_data[cell_id] = {
                'charge_data': [],
                'discharge_data': [],
                'impedance_data': []
            }
            
            raw_data = loadmat(cell, squeeze_me=True, struct_as_record=False)
            battery_keys = [k for k in raw_data.keys() if not k.startswith('__') and k.startswith('B')]
            print(f"  Using key: {battery_keys[0]}")

            cycle_data = raw_data[battery_keys[0]].cycle
            for cycle_number, cycle in enumerate(cycle_data):
                cycle_type = cycle.type
                ambient_temp = cycle.ambient_temperature
                time = cycle.time
                time_str = f"{int(time[0])}-{int(time[1]):02d}-{int(time[2]):02d}T{int(time[3]):02d}:{int(time[4]):02d}:{time[5]:.2f}"

                if cycle_type == "discharge":
                    cycle_data = {
                        'cycle_number': cycle_number + 1,
                        'type': cycle_type,
                        'ambient_temperature': ambient_temp,
                        'time': time_str,
                        'voltage': cycle.data.Voltage_measured,
                        'current': cycle.data.Current_measured,
                        'temperature': cycle.data.Temperature_measured,
                        'capacity': getattr(cycle.data, 'Capacity', None),
                        'time_vec': cycle.data.Time,
                    }
                    if hasattr(cycle.data, 'Current_charge'):
                        cycle_data['current_charge'] = cycle.data.Current_charge
                    if hasattr(cycle.data, 'Voltage_charge'):
                        cycle_data['voltage_charge'] = cycle.data.Voltage_charge
                    self.cell_data[cell_id]['discharge_data'].append(cycle_data)

                elif cycle_type == "charge":
                    cycle_data = {
                        'cycle_number': cycle_number + 1,
                        'type': cycle_type,
                        'ambient_temperature': ambient_temp,
                        'time': time_str,
                        'voltage': cycle.data.Voltage_measured,
                        'current': cycle.data.Current_measured,
                        'temperature': cycle.data.Temperature_measured,
                        'time_vec': cycle.data.Time,
                    }
                    if hasattr(cycle.data, 'Current_charge'):
                        cycle_data['current_charge'] = cycle.data.Current_charge
                    if hasattr(cycle.data, 'Voltage_charge'):
                        cycle_data['voltage_charge'] = cycle.data.Voltage_charge
                    self.cell_data[cell_id]['charge_data'].append(cycle_data)

                elif cycle_type == "impedance":
                    cycle_data = {
                        'cycle_number': cycle_number + 1,
                        'type': cycle_type,
                        'ambient_temperature': ambient_temp,
                        'time': time_str,
                    }
                    for field in ['Sense_current', 'Battery_current', 'Current_ratio',
                                  'Battery_impedance', 'Rectified_impedance', 'Re', 'Rct']:
                        if hasattr(cycle.data, field):
                            cycle_data[field.lower()] = getattr(cycle.data, field)
                    self.cell_data[cell_id]['impedance_data'].append(cycle_data)

    def _extract_features(self):
        # Base output directory
        base_output_dir = os.path.join(self.data_path, 'lookup_tables')
        os.makedirs(base_output_dir, exist_ok=True)

        for cell_id in self.cell_data.keys():
            print(f"\nExtracting features for cell {cell_id}")
            
            # Create a subfolder for this cell
            output_dir = os.path.join(base_output_dir, cell_id)
            os.makedirs(output_dir, exist_ok=True)

            discharge_data = self.cell_data[cell_id]['discharge_data']
            charge_data = self.cell_data[cell_id]['charge_data']
            impedance_data = self.cell_data[cell_id]['impedance_data']

            # Step 1: Extract SOC-OCV and SOC_lookup, OCV_lookup
            soc_values = []
            ocv_values = []

            # Get nominal capacity
            nominal_capacity = None
            for cycle in discharge_data:
                capacity = cycle.get('capacity')
                if isinstance(capacity, (int, float)) and capacity > 0:
                    nominal_capacity = capacity
                    print(f"  Using nominal capacity: {nominal_capacity} Ah from cycle {cycle['cycle_number']}")
                    break
            
            if nominal_capacity is None:
                print(f"  Warning: No valid nominal capacity found for cell {cell_id}. Using fallback value.")
                nominal_capacity = 1.1

            # Estimate average R for OCV calculation
            r_avg = 0
            r_count = 0
            for imp_cycle in impedance_data:
                if 're' in imp_cycle and 'rct' in imp_cycle:
                    r = imp_cycle['re'] + imp_cycle['rct']
                    if isinstance(r, (int, float)) and 0 < r < 1.0:
                        r_avg += r
                        r_count += 1
            if r_count > 0:
                r_avg /= r_count
                print(f"  Average R from impedance: {r_avg:.4f} Ω")
            else:
                r_avg = 0.1
                print(f"  No valid impedance data for R. Using fallback R: {r_avg:.4f} Ω")

            # Compute SOC and OCV
            for cycle in discharge_data:
                voltage = cycle['voltage']
                current = cycle['current']
                time_vec = cycle['time_vec']
                capacity = cycle['capacity']

                if not isinstance(capacity, (int, float)) or capacity <= 0:
                    print(f"  Skipping cycle {cycle['cycle_number']} - Invalid capacity: {capacity}")
                    continue

                voltage = np.array(voltage, dtype=float)
                current = np.array(current, dtype=float)
                time_vec = np.array(time_vec, dtype=float)

                min_len = min(len(voltage), len(current), len(time_vec))
                if min_len == 0:
                    print(f"  Skipping cycle {cycle['cycle_number']} - Empty data arrays")
                    continue

                voltage = voltage[:min_len]
                current = current[:min_len]
                time_vec = time_vec[:min_len]

                if np.any(np.isnan(voltage)) or np.any(np.isnan(current)) or np.any(np.isnan(time_vec)):
                    print(f"  Skipping cycle {cycle['cycle_number']} - NaN values detected")
                    continue

                # Compute discharged capacity
                q_discharged = np.zeros(len(time_vec))
                for i in range(1, len(time_vec)):
                    dt = time_vec[i] - time_vec[i-1]
                    avg_current = (current[i] + current[i-1]) / 2
                    if not np.isfinite(avg_current) or not np.isfinite(dt):
                        q_discharged[i] = q_discharged[i-1]
                        continue
                    q_discharged[i] = q_discharged[i-1] + avg_current * dt / 3600

                q_discharged = np.abs(q_discharged)
                soc = (1 - q_discharged / capacity) * 100
                ocv = voltage + current * r_avg

                for s, o in zip(soc, ocv):
                    if not (np.isfinite(s) and np.isfinite(o)):
                        continue
                    if 0 <= s <= 100 and 2.5 <= o <= 4.2:
                        soc_values.append(s)
                        ocv_values.append(o)

            # Convert to numpy arrays for SOC-OCV LUT
            SOC_lookup = np.array(soc_values, dtype=float)
            OCV_lookup = np.array(ocv_values, dtype=float)

            # Step 2: Extract R(SOC) and R_lookup
            soc_r_values = []
            r_values = []

            total_discharged_capacity = 0
            for cycle in discharge_data + charge_data + impedance_data:
                cycle_type = cycle['type']
                cycle_number = cycle['cycle_number']

                if cycle_type in ['discharge', 'charge']:
                    current = np.array(cycle['current'], dtype=float)
                    time_vec = np.array(cycle['time_vec'], dtype=float)

                    min_len = min(len(current), len(time_vec))
                    if min_len == 0:
                        continue

                    current = current[:min_len]
                    time_vec = time_vec[:min_len]

                    if np.any(np.isnan(current)) or np.any(np.isnan(time_vec)):
                        continue

                    q_cycle = 0
                    for i in range(1, len(time_vec)):
                        dt = time_vec[i] - time_vec[i-1]
                        avg_current = (current[i] + current[i-1]) / 2
                        if not np.isfinite(avg_current) or not np.isfinite(dt):
                            continue
                        q_cycle += avg_current * dt / 3600

                    if cycle_type == 'discharge':
                        total_discharged_capacity += np.abs(q_cycle)
                    else:
                        total_discharged_capacity -= np.abs(q_cycle)

                    total_discharged_capacity = max(0, min(total_discharged_capacity, nominal_capacity))

                elif cycle_type == 'impedance':
                    soc_imp = (1 - total_discharged_capacity / nominal_capacity) * 100
                    soc_imp = max(0, min(100, soc_imp))

                    if 're' in cycle and 'rct' in cycle:
                        r = cycle['re'] + cycle['rct']
                        if isinstance(r, (int, float)) and 0.01 < r < 1.0:
                            soc_r_values.append(soc_imp)
                            r_values.append(r)

            R_lookup = np.array(r_values, dtype=float)
            SOC_R_lookup = np.array(soc_r_values, dtype=float)

            # Step 3: Extract C-rate_lookup
            c_rate_values = set()

            for cycle in discharge_data:
                try:
                    avg_current = np.abs(np.mean(cycle['current']))
                    capacity = cycle['capacity']
                    if not isinstance(capacity, (int, float)) or capacity <= 0:
                        continue
                    c_rate = avg_current / capacity
                    if 0.5 < c_rate < 2.0:
                        c_rate_values.add(round(c_rate, 3))
                except:
                    continue

            for cycle in charge_data:
                try:
                    voltage = np.array(cycle['voltage'], dtype=float)
                    current = np.array(cycle['current'], dtype=float)
                    min_len = min(len(voltage), len(current))
                    if min_len == 0:
                        continue
                    voltage = voltage[:min_len]
                    current = current[:min_len]

                    cc_mask = (voltage < 4.2) & (current > 1.0)
                    if not np.any(cc_mask):
                        continue
                    avg_current = np.mean(current[cc_mask])
                    c_rate = avg_current / nominal_capacity
                    if 0.5 < c_rate < 2.0:
                        c_rate_values.add(round(c_rate, 3))
                except:
                    continue

            C_rate_lookup = np.array(sorted(list(c_rate_values)), dtype=float)

            # Step 4: Outlier Detection
            skip_cell = False
            
            # Filter for empty SOC-OCV data
            if len(SOC_lookup) == 0:
                print(f"  Warning: Cell {cell_id} has no valid SOC-OCV data")
                skip_cell = True
            
            # Filter for SOC-OCV coverage
            if len(SOC_lookup) > 0:
                soc_range = max(SOC_lookup) - min(SOC_lookup)
                if soc_range < 30:  # At least 30% SOC coverage (relaxed from 50%)
                    print(f"  Cell {cell_id} is an outlier: Insufficient SOC range ({soc_range:.1f}%)")
                    skip_cell = True
            
            # Filter for C-rate range
            if len(C_rate_lookup) < 5:  # Relaxed from 10
                print(f"  Cell {cell_id} is an outlier: Not enough C-rate data points ({len(C_rate_lookup)})")
                skip_cell = True
            
            # Filter for resistance values consistency
            if len(R_lookup) < 5:  # Relaxed from 10
                print(f"  Cell {cell_id} is an outlier: Not enough resistance data points ({len(R_lookup)})")
                skip_cell = True
            
            if skip_cell:
                print(f"  Skipping cell {cell_id} due to outlier detection")
                
                # Delete the directory if it exists (to clean up previous run)
                cell_dir = os.path.join(output_dir, cell_id)
                if os.path.exists(cell_dir):
                    import shutil
                    shutil.rmtree(cell_dir)
                    print(f"  Deleted directory for filtered cell: {cell_dir}")
                
                continue
            
            # Step 5: Save LUTs as .mat file
            
            # Calculate additional battery health metrics
            initial_capacity = nominal_capacity if nominal_capacity is not None else 0
            final_capacity = 0
            max_current = 0
            cycle_count = 0
            
            # Get the final capacity from the last discharge cycle
            if len(discharge_data) > 0:
                # Sort discharge cycles by cycle number
                sorted_discharge = sorted(discharge_data, key=lambda x: x['cycle_number'])
                if len(sorted_discharge) > 0:
                    last_cycle = sorted_discharge[-1]
                    if 'capacity' in last_cycle and last_cycle['capacity'] is not None:
                        final_capacity = last_cycle['capacity']
                    cycle_count = last_cycle['cycle_number']
            
            # Get the maximum current
            for cycle in discharge_data + charge_data:
                if 'current' in cycle and len(cycle['current']) > 0:
                    max_cycle_current = np.max(np.abs(cycle['current']))
                    max_current = max(max_current, max_cycle_current)
            
            # Calculate capacity loss percentage
            capacity_loss_percent = 0
            if initial_capacity > 0 and final_capacity > 0:
                capacity_loss_percent = ((initial_capacity - final_capacity) / initial_capacity) * 100
            
            # Add health metrics to the lookup data
            lut_data = {
                'R_lookup': R_lookup,
                'SOC_lookup': SOC_lookup,
                'C_rate_lookup': C_rate_lookup,
                'OCV_lookup': OCV_lookup,
                'SOC_R_lookup': SOC_R_lookup,
                'initial_capacity': initial_capacity,
                'final_capacity': final_capacity,
                'capacity_loss_percent': capacity_loss_percent,
                'max_current': max_current,
                'cycle_count': cycle_count
            }
            
            # Print capacity information
            print(f"  Cell Health Metrics:")
            print(f"    Initial Capacity: {initial_capacity:.3f} Ah")
            print(f"    Final Capacity: {final_capacity:.3f} Ah")
            print(f"    Capacity Loss: {capacity_loss_percent:.1f}%")
            print(f"    Maximum Current: {max_current:.3f} A")
            print(f"    Cycle Count: {cycle_count}")
            
            lut_path = os.path.join(output_dir, f'{cell_id}_lookup.mat')
            sio.savemat(lut_path, lut_data)
            print(f"  Saved LUT to {lut_path}")
            print(f"  SOC-OCV LUT: {len(SOC_lookup)} points")
            if len(SOC_lookup) > 0:
                print(f"  SOC Stats: min={min(SOC_lookup):.4f}, max={max(SOC_lookup):.4f}, mean={np.mean(SOC_lookup):.4f}")
                print(f"  OCV Stats: min={min(OCV_lookup):.4f}, max={max(OCV_lookup):.4f}, mean={np.mean(OCV_lookup):.4f}")
            else:
                print(f"  Warning: No SOC-OCV data available for stats")
            print(f"  R LUT: {len(R_lookup)} points")
            if len(R_lookup) > 0:
                print(f"  R Stats: min={min(R_lookup):.4f}, max={max(R_lookup):.4f}, mean={np.mean(R_lookup):.4f}")
            else:
                print(f"  Warning: No R data available for stats")
            print(f"  C-rate LUT: {len(C_rate_lookup)} points")
            if len(C_rate_lookup) > 0:
                print(f"  C-rate Stats: min={min(C_rate_lookup):.4f}, max={max(C_rate_lookup):.4f}, mean={np.mean(C_rate_lookup):.4f}")
            else:
                print(f"  Warning: No C-rate data available for stats")

            # Save CSVs in the cell-specific folder
            if len(soc_values) > 0:
                soc_ocv_df = pd.DataFrame({
                    'SOC': soc_values,
                    'OCV': ocv_values
                })
            else:
                soc_ocv_df = pd.DataFrame(columns=['SOC', 'OCV'])
            soc_ocv_path = os.path.join(output_dir, f'{cell_id}_soc_ocv.csv')
            soc_ocv_df.to_csv(soc_ocv_path, index=False)
            print(f"  Saved SOC-OCV CSV to {soc_ocv_path}")

            if len(soc_r_values) > 0:
                r_df = pd.DataFrame({
                    'SOC': soc_r_values,
                    'R': r_values
                })
            else:
                r_df = pd.DataFrame(columns=['SOC', 'R'])
            r_path = os.path.join(output_dir, f'{cell_id}_r.csv')
            r_df.to_csv(r_path, index=False)
            print(f"  Saved R(SOC) CSV to {r_path}")

            if len(c_rate_values) > 0:
                c_rate_df = pd.DataFrame({
                    'C_rate': list(c_rate_values)
                })
            else:
                c_rate_df = pd.DataFrame(columns=['C_rate'])
            c_rate_path = os.path.join(output_dir, f'{cell_id}_c_rate.csv')
            c_rate_df.to_csv(c_rate_path, index=False)
            print(f"  Saved C-rate CSV to {c_rate_path}")

            # Step 6: Generate Plots in the cell-specific folder
            if len(SOC_lookup) > 0:
                try:
                    plt.figure(figsize=(10, 6))
                    df = pd.DataFrame({'SOC': SOC_lookup, 'OCV': OCV_lookup})
                    plt.scatter(SOC_lookup, OCV_lookup, s=1, alpha=0.1, label='Raw Data')
                    
                    # Only create binned plot if we have enough data points
                    if len(SOC_lookup) >= 100:
                        df_binned = df.groupby(pd.cut(df['SOC'], bins=100)).mean().dropna()
                        plt.plot(df_binned['SOC'], df_binned['OCV'], 'r-', label='Smoothed Curve')
                    
                    plt.xlabel('SOC (%)')
                    plt.ylabel('OCV (V)')
                    plt.title(f'SOC-OCV Curve for {cell_id}')
                    plt.legend()
                    plt.grid()
                    plot_path = os.path.join(output_dir, f'{cell_id}_soc_ocv.png')
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"  SOC-OCV plot saved to {plot_path}")
                except Exception as e:
                    print(f"  Warning: Could not generate SOC-OCV plot: {e}")
                    plt.close()
            else:
                print(f"  Skipping SOC-OCV plot for {cell_id} - No data available")

            if len(R_lookup) > 0:
                try:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(SOC_R_lookup, R_lookup, s=5, label='R vs SOC')
                    plt.xlabel('SOC (%)')
                    plt.ylabel('Resistance (Ω)')
                    plt.title(f'R(SOC) for {cell_id}')
                    plt.legend()
                    plt.grid()
                    plot_path = os.path.join(output_dir, f'{cell_id}_r_soc.png')
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"  R(SOC) plot saved to {plot_path}")
                except Exception as e:
                    print(f"  Warning: Could not generate R(SOC) plot: {e}")
                    plt.close()
            else:
                print(f"  Skipping R(SOC) plot for {cell_id} - No data available")

            if len(C_rate_lookup) > 1:  # Need at least 2 points for histogram
                try:
                    plt.figure(figsize=(10, 6))
                    plt.hist(C_rate_lookup, bins=min(20, len(C_rate_lookup)), edgecolor='black')
                    plt.xlabel('C-rate')
                    plt.ylabel('Frequency')
                    plt.title(f'C-rate Distribution for {cell_id}')
                    plt.grid()
                    plot_path = os.path.join(output_dir, f'{cell_id}_c_rate_hist.png')
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"  C-rate histogram saved to {plot_path}")
                except Exception as e:
                    print(f"  Warning: Could not generate C-rate histogram: {e}")
                    plt.close()
            else:
                print(f"  Skipping C-rate histogram for {cell_id} - Not enough data")

if __name__ == "__main__":
    data_path = "/home/danial/Documents/Codes_new/N-IJCCI-BMS/datasets/Nasa_batteries"
    
    # First, delete all existing lookup tables to start fresh
    lookup_tables_dir = os.path.join(data_path, 'lookup_tables')
    if os.path.exists(lookup_tables_dir):
        import shutil
        print(f"Deleting existing lookup tables directory: {lookup_tables_dir}")
        shutil.rmtree(lookup_tables_dir)
        os.makedirs(lookup_tables_dir)
    else:
        os.makedirs(lookup_tables_dir)
    
    # Define the list of cells to keep based on the report
    cells_to_keep = [
        "B0018", "B0005", "B0007", "B0029", "B0044", 
        "B0025", "B0043", "B0027", "B0028", "B0042", 
        "B0026", "B0032", "B0030", "B0048", "B0046", "B0047"
    ]
    
    print("NASA Battery Cell Filtering Report")
    print(f"Dataset: {data_path}")
    print("Keeping only the following cells based on criteria:")
    print("1. Drop cells with negative or >70% capacity loss")
    print("2. Keep cells with initial capacity in 1.6-2.0 Ah band")
    print("3. Keep cells with current ratings ~2A")
    print("\nCells to keep:")
    for cell in cells_to_keep:
        print(f"  - {cell}")
    
    reader = NasaDataReader(data_path)
    reader._load_data()
    
    try:
        # Process only cells in the keep list
        cell_ids = list(reader.cell_data.keys())
        for cell_id in cell_ids:
            if cell_id not in cells_to_keep:
                print(f"Skipping cell {cell_id} - not in the list of cells to keep")
                del reader.cell_data[cell_id]
        
        # Store cell health metrics for summary report
        cell_metrics = []
        
        reader._extract_features()
        
        # After processing all cells, create a summary report
        print("\nGenerating summary report...")
        
        # Get all cell health metrics from the saved MAT files
        for cell_id in cells_to_keep:
            cell_dir = os.path.join(lookup_tables_dir, cell_id)
            if os.path.exists(cell_dir):
                lut_path = os.path.join(cell_dir, f'{cell_id}_lookup.mat')
                try:
                    data = sio.loadmat(lut_path)
                    metrics = {
                        'cell_id': cell_id,
                        'initial_capacity': float(data.get('initial_capacity', 0)),
                        'final_capacity': float(data.get('final_capacity', 0)),
                        'capacity_loss_percent': float(data.get('capacity_loss_percent', 0)),
                        'max_current': float(data.get('max_current', 0)),
                        'cycle_count': int(data.get('cycle_count', 0))
                    }
                    cell_metrics.append(metrics)
                except Exception as e:
                    print(f"Warning: Could not load metrics for cell {cell_id}: {e}")
        
        # Create a CSV report
        if cell_metrics:
            import csv
            report_path = os.path.join(lookup_tables_dir, 'cell_metrics_report.csv')
            with open(report_path, 'w', newline='') as csvfile:
                fieldnames = ['cell_id', 'cycle_count', 'initial_capacity', 'final_capacity', 
                              'capacity_loss_percent', 'max_current']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for metrics in cell_metrics:
                    writer.writerow(metrics)
            print(f"Saved metrics report to {report_path}")
            
            # Create a text report similar to the format provided by the user
            text_report_path = os.path.join(lookup_tables_dir, 'cell_metrics_report.txt')
            with open(text_report_path, 'w') as f:
                import datetime
                f.write(f"NASA Battery Cell Filtering Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dataset: {data_path}\n")
                f.write(f"Total cells processed: {len(cell_metrics)}\n\n")
                
                f.write("Filtering Rules Applied:\n")
                f.write("1. Drop cells with negative or > 70% capacity-loss (likely measurement glitches)\n")
                f.write("2. Keep cells with initial capacity in the 1.6-2.0 Ah band (for comparable starting points)\n")
                f.write("3. Keep cells with current ratings ≈ 2 A\n\n")
                
                f.write("Kept Cells:\n")
                f.write("|--------------------------------------------------------------------------------------|\n")
                f.write("| Cell Name | Cycles | Initial Cap. | Final Cap. | Cap. Loss % | Max Current | Comment |\n")
                f.write("|--------------------------------------------------------------------------------------|\n")
                
                for metrics in sorted(cell_metrics, key=lambda x: x['cell_id']):
                    cell_id = metrics['cell_id']
                    cycles = metrics['cycle_count']
                    initial_cap = metrics['initial_capacity']
                    final_cap = metrics['final_capacity']
                    cap_loss = metrics['capacity_loss_percent']
                    max_current = metrics['max_current']
                    
                    f.write(f"| {cell_id:<9} | {cycles:<6} | {initial_cap:<12.3f} | {final_cap:<10.3f} | {cap_loss:<11.1f} | {max_current:<11.3f} | KEPT |\n")
                
                f.write("|--------------------------------------------------------------------------------------|\n")
            
            print(f"Saved detailed report to {text_report_path}")
        
        print(f"\nSuccessfully processed {len(cells_to_keep)} cells")
    except Exception as e:
        print(f"Error processing cells: {e}")
        import traceback
        traceback.print_exc()
        print("Processing failed")
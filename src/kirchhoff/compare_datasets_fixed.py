import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from src.utils.config_reader import load_config
from kir_1_fixed import OptimizeSoC
import pandas as pd

# Create output directory
os.makedirs("comparison_results_fixed", exist_ok=True)

def load_nmc_data(config):
    """Load NMC battery data"""
    print("\nLoading NMC Data...")
    from src.utils.NMC_data_reader.load_data import matlab_load_data
    
    R_lookup, SOC_lookup, C_rate_lookup, OCV_lookup = matlab_load_data(config.data.matlab_data)
    
    print(f"NMC Data Loaded:")
    print(f"  R_lookup shape: {R_lookup.shape}")
    print(f"  SOC_lookup shape: {SOC_lookup.shape}")
    print(f"  C_rate_lookup shape: {C_rate_lookup.shape}")
    print(f"  OCV_lookup shape: {OCV_lookup.shape}")
    
    return R_lookup, SOC_lookup, C_rate_lookup, OCV_lookup

def load_nasa_data(config, cell_id="B0005"):
    """Load NASA battery data matched to NMC format"""
    print(f"\nLoading NASA Data (Cell {cell_id})...")
    from src.utils.nasa_data_readers.nasa_data_compatibility import get_nasa_cell_as_nmc
    
    nasa_data = get_nasa_cell_as_nmc(cell_id, config.data.matlab_data)
    
    R_lookup = nasa_data['R']
    SOC_lookup = nasa_data['SOC']
    C_rate_lookup = nasa_data['C_rate']
    OCV_lookup = np.flip(nasa_data['OCV'])
    
    print(f"NASA Data Loaded:")
    print(f"  R_lookup shape: {R_lookup.shape}")
    print(f"  SOC_lookup shape: {SOC_lookup.shape}")
    print(f"  C_rate_lookup shape: {C_rate_lookup.shape}")
    print(f"  OCV_lookup shape: {OCV_lookup.shape}")
    
    return R_lookup, SOC_lookup, C_rate_lookup, OCV_lookup

def run_optimization(dataset_name, R_lookup, SOC_lookup, C_rate_lookup, OCV_lookup, config):
    """Run optimization using the given dataset"""
    from src.utils.Energy_data.energy_data_reader import PNetDataLoader
    
    # Load power data
    P_net = PNetDataLoader(config.data.P_net)
    
    # Parameters from config
    soc_init = config.simulation.soc_init
    epsilon = config.simulation.epsilon
    nominal_capacity = config.battery.nominal_capacity
    nominal_current = config.battery.nominal_current
    time_interval = config.simulation.time_interval/60  # minutes
    nominal_Ah = config.battery.nominal_Ah
    
    # Results storage
    results = {
        'power': [],
        'soc': [],
        'resistance': [],
        'current': [],
        'mode': []  # Add mode (charging/discharging)
    }
    
    # Run optimization for a number of power steps
    num_steps = 20  # Same as in main.py
    
    # Starting SOC for recording delta
    starting_soc = soc_init
    
    print(f"\nRunning optimization with {dataset_name} dataset...")
    for ind in range(num_steps):
        print(f"Power step {ind+1}/{num_steps}")
        power = P_net[ind]
        
        # Print power value to help diagnose
        if ind < 5:  # Just show first few to keep output clean
            print(f"  Power at step {ind+1}: {power} kW")
        
        # Get OCV via interpolation
        ocv = np.interp(soc_init, SOC_lookup, OCV_lookup)
        
        # Create optimizer
        optimizer = OptimizeSoC(
            soc_init,
            SOC_lookup,
            C_rate_lookup,
            R_lookup,
            ocv,
            power,
            epsilon,
            nominal_current,
            nominal_capacity,
            GA=False,
            verbose=False
        )
        
        # Optimize
        new_R, new_I, info = optimizer._optimize(soc_init, ocv)
        
        # Determine mode based on current
        mode = "discharging" if new_I < 0 else "charging"
        
        # Update SOC with clear explanation
        old_soc = soc_init
        soc_init += new_I * time_interval / nominal_Ah * 100
        
        # Log details for debugging
        if ind < 5:  # Just show first few to keep output clean
            print(f"  Current: {new_I:.4f} A, Mode: {mode}")
            print(f"  SOC change: {old_soc:.4f} → {soc_init:.4f} ({soc_init-old_soc:+.4f})")
        
        # Store results
        results['power'].append(power)
        results['soc'].append(soc_init)
        results['resistance'].append(new_R)
        results['current'].append(new_I)
        results['mode'].append(mode)
    
    # Summarize charge/discharge distribution
    charging_steps = results['mode'].count("charging")
    discharging_steps = results['mode'].count("discharging")
    print(f"\n{dataset_name} Summary:")
    print(f"  Starting SOC: {starting_soc:.2f}%")
    print(f"  Ending SOC: {soc_init:.2f}%")
    print(f"  Net SOC change: {soc_init - starting_soc:+.2f}%")
    print(f"  Charging steps: {charging_steps} ({charging_steps/num_steps*100:.1f}%)")
    print(f"  Discharging steps: {discharging_steps} ({discharging_steps/num_steps*100:.1f}%)")
    
    return results

def compare_and_save_results(nmc_results, nasa_results):
    """Compare and save results from both datasets"""
    # Convert results to DataFrames
    nmc_df = pd.DataFrame(nmc_results)
    nasa_df = pd.DataFrame(nasa_results)
    
    # Calculate percentage differences
    comparison_df = pd.DataFrame()
    comparison_df['step'] = range(len(nmc_df))
    comparison_df['power'] = nmc_df['power']
    comparison_df['soc_nmc'] = nmc_df['soc']
    comparison_df['soc_nasa'] = nasa_df['soc']
    comparison_df['resistance_nmc'] = nmc_df['resistance']
    comparison_df['resistance_nasa'] = nasa_df['resistance']
    comparison_df['current_nmc'] = nmc_df['current']
    comparison_df['current_nasa'] = nasa_df['current']
    comparison_df['resistance_diff_pct'] = (nasa_df['resistance'] - nmc_df['resistance']) / nmc_df['resistance'] * 100
    comparison_df['current_diff_pct'] = (nasa_df['current'] - nmc_df['current']) / nmc_df['current'] * 100
    comparison_df['soc_diff_pct'] = (nasa_df['soc'] - nmc_df['soc']) / nmc_df['soc'] * 100
    
    # Save to CSV
    nmc_df.to_csv("comparison_results_fixed/nmc_results.csv", index=False)
    nasa_df.to_csv("comparison_results_fixed/nasa_results.csv", index=False)
    comparison_df.to_csv("comparison_results_fixed/comparison_analysis.csv", index=False)
    
    # Print summary statistics
    print("\nComparison Summary:")
    print(f"  Average Resistance (NMC): {nmc_df['resistance'].mean():.6f} Ohm")
    print(f"  Average Resistance (NASA): {nasa_df['resistance'].mean():.6f} Ohm")
    print(f"  Resistance Difference: {comparison_df['resistance_diff_pct'].mean():.2f}%")
    print(f"  Current Magnitude Difference: {comparison_df['current_diff_pct'].mean():.2f}%")
    print(f"  SOC Change (NMC): {nmc_df['soc'].iloc[-1] - nmc_df['soc'].iloc[0]:.4f}%")
    print(f"  SOC Change (NASA): {nasa_df['soc'].iloc[-1] - nasa_df['soc'].iloc[0]:.4f}%")
    
    # Create comparison plots
    create_comparison_plots(nmc_df, nasa_df, comparison_df)
    
    return nmc_df, nasa_df, comparison_df

def create_comparison_plots(nmc_df, nasa_df, comparison_df):
    """Create enhanced comparison plots between NMC and NASA results"""
    # Plot 1: SOC comparison
    plt.figure(figsize=(12, 8))
    plt.plot(nmc_df.index, nmc_df['soc'], 'b-', linewidth=2, label='NMC')
    plt.plot(nasa_df.index, nasa_df['soc'], 'r--', linewidth=2, label='NASA')
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('State of Charge (%)', fontsize=14)
    plt.title('SOC Comparison between NMC and NASA Data (FIXED)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig("comparison_results_fixed/soc_comparison.png", dpi=300, bbox_inches='tight')
    
    # Plot 2: Resistance comparison
    plt.figure(figsize=(12, 8))
    plt.plot(nmc_df.index, nmc_df['resistance'], 'b-', linewidth=2, label='NMC')
    plt.plot(nasa_df.index, nasa_df['resistance'], 'r--', linewidth=2, label='NASA')
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Resistance (Ohm)', fontsize=14)
    plt.title('Resistance Comparison between NMC and NASA Data (FIXED)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig("comparison_results_fixed/resistance_comparison.png", dpi=300, bbox_inches='tight')
    
    # Plot 3: Current comparison
    plt.figure(figsize=(12, 8))
    plt.plot(nmc_df.index, nmc_df['current'], 'b-', linewidth=2, label='NMC')
    plt.plot(nasa_df.index, nasa_df['current'], 'r--', linewidth=2, label='NASA')
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Current (A)', fontsize=14)
    plt.title('Current Comparison between NMC and NASA Data (FIXED)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig("comparison_results_fixed/current_comparison.png", dpi=300, bbox_inches='tight')
    
    # Plot 4: Power profile with charge/discharge indication
    plt.figure(figsize=(12, 8))
    plt.bar(nmc_df.index, nmc_df['power'], color=['r' if i > 0 else 'g' for i in nmc_df['power']])
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Power (kW)', fontsize=14)
    plt.title('Input Power Profile (Red=Discharge, Green=Charge) - FIXED', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.savefig("comparison_results_fixed/power_profile_colored.png", dpi=300, bbox_inches='tight')
    
    # Plot 5: Percentage differences
    plt.figure(figsize=(12, 8))
    plt.plot(comparison_df['step'], comparison_df['resistance_diff_pct'], 'g-', linewidth=2, label='Resistance Diff %')
    plt.plot(comparison_df['step'], comparison_df['current_diff_pct'], 'b-', linewidth=2, label='Current Diff %')
    plt.plot(comparison_df['step'], comparison_df['soc_diff_pct'], 'r-', linewidth=2, label='SOC Diff %')
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Difference (%)', fontsize=14)
    plt.title('Percentage Differences (NASA vs NMC) - FIXED', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig("comparison_results_fixed/percentage_differences.png", dpi=300, bbox_inches='tight')
    
    # Plot 6: SOC delta from start
    plt.figure(figsize=(12, 8))
    plt.plot(nmc_df.index, nmc_df['soc'] - nmc_df['soc'].iloc[0], 'b-', linewidth=2, label='NMC')
    plt.plot(nasa_df.index, nasa_df['soc'] - nasa_df['soc'].iloc[0], 'r--', linewidth=2, label='NASA')
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('SOC Change from Start (%)', fontsize=14)
    plt.title('SOC Change from Initial Value - FIXED', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig("comparison_results_fixed/soc_delta.png", dpi=300, bbox_inches='tight')

def main():
    # Load configuration
    config_path = "/home/danial/Documents/Codes_new/N-IJCCI-BMS/src/configuration/config.yml"
    config = load_config(config_path)
    
    # Load NMC data
    try:
        R_nmc, SOC_nmc, C_rate_nmc, OCV_nmc = load_nmc_data(config)
    except Exception as e:
        print(f"Error loading NMC data: {e}")
        print("Falling back to NASA data for NMC comparison...")
        R_nmc, SOC_nmc, C_rate_nmc, OCV_nmc = load_nasa_data(config, "B0007")
    
    # Load NASA data
    R_nasa, SOC_nasa, C_rate_nasa, OCV_nasa = load_nasa_data(config, "B0005")
    
    # Analyze raw data differences
    print("\nRaw Data Comparison:")
    print(f"  R_lookup mean (NMC): {np.mean(R_nmc):.6f}")
    print(f"  R_lookup mean (NASA): {np.mean(R_nasa):.6f}")
    print(f"  R_lookup difference: {(np.mean(R_nasa) - np.mean(R_nmc))/np.mean(R_nmc)*100:.2f}%")
    print(f"  OCV range (NMC): {np.min(OCV_nmc):.4f} to {np.max(OCV_nmc):.4f} V")
    print(f"  OCV range (NASA): {np.min(OCV_nasa):.4f} to {np.max(OCV_nasa):.4f} V")
    
    # Run optimization with NMC data
    nmc_results = run_optimization("NMC", R_nmc, SOC_nmc, C_rate_nmc, OCV_nmc, config)
    
    # Run optimization with NASA data
    nasa_results = run_optimization("NASA", R_nasa, SOC_nasa, C_rate_nasa, OCV_nasa, config)
    
    # Compare and save results
    nmc_df, nasa_df, comparison_df = compare_and_save_results(nmc_results, nasa_results)
    
    print("\nComparison completed!")
    print(f"Results saved in: {os.path.abspath('comparison_results_fixed')}")
    print("\nIMPORTANT: The fixed model now correctly handles the power sign convention:")
    print("  - Positive P_net (deficit) results in battery discharge (SOC decreases)")
    print("  - Negative P_net (surplus) results in battery charge (SOC increases)")

if __name__ == "__main__":
    main() 
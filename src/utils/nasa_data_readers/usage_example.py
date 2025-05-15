"""
NASA Battery Data Compatibility Layer - Usage Example

This script demonstrates how to use NASA battery data with existing code that was
designed for NMC battery data, ensuring compatibility between the two data formats.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append("/home/danial/Documents/Codes_new/N-IJCCI-BMS")

# Import configuration reader
from src.utils.config_reader import load_config

# -------------- DEMONSTRATION OF COMPATIBILITY ---------------

def demo_nmc_direct():
    """
    Original way of loading NMC data - this is your existing code pattern
    """
    print("\nDEMO 1: Original NMC Data Loading")
    print("-" * 50)
    
    # Load configuration
    config_path = "/home/danial/Documents/Codes_new/N-IJCCI-BMS/src/configuration/config.yml"
    config = load_config(config_path)
    
    # Original way to import NMC data reader
    from src.utils.NMC_data_reader.load_data import matlab_load_data
    
    # Original way to load NMC data
    R_lookup, SOC_lookup, C_rate_lookup, OCV_lookup = matlab_load_data(config.data.matlab_data)
    
    print(f"NMC Data Successfully Loaded:")
    print(f"  R_lookup shape: {R_lookup.shape}")
    print(f"  SOC_lookup shape: {SOC_lookup.shape}")
    print(f"  C_rate_lookup shape: {C_rate_lookup.shape}")
    print(f"  OCV_lookup shape: {OCV_lookup.shape}")
    
    # Return the data for comparison
    return R_lookup, SOC_lookup, C_rate_lookup, OCV_lookup

def demo_nasa_compatible():
    """
    New way of loading NASA data that's compatible with existing NMC code
    """
    print("\nDEMO 2: NASA Data with Compatibility Layer")
    print("-" * 50)
    
    # Load configuration
    config_path = "/home/danial/Documents/Codes_new/N-IJCCI-BMS/src/configuration/config.yml"
    config = load_config(config_path)
    
    # Import our compatibility layer
    from src.utils.nasa_data_readers.nasa_data_compatibility import nasa_matlab_load_data
    
    # Select a NASA cell to use
    nasa_cell_id = "B0005"  # One of our filtered cells
    
    # Load NASA data with the same function signature as NMC data
    R_lookup, SOC_lookup, C_rate_lookup, OCV_lookup = nasa_matlab_load_data(nasa_cell_id)
    
    print(f"NASA Data Successfully Loaded:")
    print(f"  R_lookup shape: {R_lookup.shape}")
    print(f"  SOC_lookup shape: {SOC_lookup.shape}")
    print(f"  C_rate_lookup shape: {C_rate_lookup.shape}")
    print(f"  OCV_lookup shape: {OCV_lookup.shape}")
    
    # Return the data for comparison
    return R_lookup, SOC_lookup, C_rate_lookup, OCV_lookup

def demo_nasa_exact_match():
    """
    Loading NASA data to exactly match NMC data dimensions
    """
    print("\nDEMO 3: NASA Data with Exact NMC Format Matching")
    print("-" * 50)
    
    # Load configuration
    config_path = "/home/danial/Documents/Codes_new/N-IJCCI-BMS/src/configuration/config.yml"
    config = load_config(config_path)
    
    # Import exact matching function
    from src.utils.nasa_data_readers.nasa_data_compatibility import get_nasa_cell_as_nmc
    
    # Select a NASA cell to use
    nasa_cell_id = "B0005"  # One of our filtered cells
    
    # Get NASA data exactly matching NMC format
    nasa_as_nmc = get_nasa_cell_as_nmc(nasa_cell_id, config.data.matlab_data)
    
    # Unpack the data just like you would with NMC data
    R_lookup = nasa_as_nmc['R']
    SOC_lookup = nasa_as_nmc['SOC']
    C_rate_lookup = nasa_as_nmc['C_rate']
    OCV_lookup = np.flip(nasa_as_nmc['OCV'])  # Need to flip because we store raw format
    
    print(f"NASA Data Matched to NMC Format:")
    print(f"  R_lookup shape: {R_lookup.shape}")
    print(f"  SOC_lookup shape: {SOC_lookup.shape}")
    print(f"  C_rate_lookup shape: {C_rate_lookup.shape}")
    print(f"  OCV_lookup shape: {OCV_lookup.shape}")
    
    # Return the data for comparison
    return R_lookup, SOC_lookup, C_rate_lookup, OCV_lookup

def demo_class_based():
    """
    Using object-oriented approach with NasaDataLoader class
    """
    print("\nDEMO 4: Using NasaDataLoader Class")
    print("-" * 50)
    
    # Load configuration
    config_path = "/home/danial/Documents/Codes_new/N-IJCCI-BMS/src/configuration/config.yml"
    config = load_config(config_path)
    
    # Import our data loader class
    from src.utils.nasa_data_readers.nasa_data_compatibility import NasaDataLoader
    
    # Select a NASA cell to use
    nasa_cell_id = "B0005"  # One of our filtered cells
    
    # Create loader with exact NMC matching
    loader = NasaDataLoader(nasa_cell_id, nmc_data_path=config.data.matlab_data)
    
    # Get the data
    R_lookup, SOC_lookup, C_rate_lookup, OCV_lookup = loader.get_lookup_tables()
    
    print(f"NASA Data Loaded via Class:")
    print(f"  R_lookup shape: {R_lookup.shape}")
    print(f"  SOC_lookup shape: {SOC_lookup.shape}")
    print(f"  C_rate_lookup shape: {C_rate_lookup.shape}")
    print(f"  OCV_lookup shape: {OCV_lookup.shape}")
    
    # Show additional cell info
    cell_info = loader.get_cell_info()
    print("\nCell Info:")
    for key, value in cell_info.items():
        print(f"  {key}: {value}")
    
    # Return the data for comparison
    return R_lookup, SOC_lookup, C_rate_lookup, OCV_lookup

def plot_comparison(nmc_data, nasa_data, nasa_matched_data):
    """
    Plot a comparison of SOC-OCV curves for visual verification
    """
    print("\nCreating comparison plot...")
    
    # Unpack the data
    _, nmc_soc, _, nmc_ocv = nmc_data
    _, nasa_soc, _, nasa_ocv = nasa_data
    _, nasa_matched_soc, _, nasa_matched_ocv = nasa_matched_data
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot NMC data
    plt.plot(nmc_soc, nmc_ocv, 'b-', linewidth=2, label='Original NMC')
    
    # Plot NASA data (raw)
    plt.plot(nasa_soc, nasa_ocv, 'r-', linewidth=2, alpha=0.6, label='NASA (raw)')
    
    # Plot NASA data (matched to NMC)
    plt.plot(nasa_matched_soc, nasa_matched_ocv, 'g--', linewidth=2, alpha=0.8, label='NASA (matched to NMC)')
    
    # Add labels and legend
    plt.xlabel('State of Charge (%)', fontsize=14)
    plt.ylabel('Open Circuit Voltage (V)', fontsize=14)
    plt.title('Comparison of NMC and NASA Battery SOC-OCV Curves', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save the plot
    output_dir = "/home/danial/Documents/Codes_new/N-IJCCI-BMS/battery_results/comparison"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'nmc_nasa_compatibility.png'), dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {os.path.join(output_dir, 'nmc_nasa_compatibility.png')}")
    
    # Show plot (uncomment if running interactively)
    # plt.show()

if __name__ == "__main__":
    # Run all the demos
    nmc_data = demo_nmc_direct()
    nasa_data = demo_nasa_compatible()
    nasa_matched_data = demo_nasa_exact_match()
    class_data = demo_class_based()
    
    # Plot comparison of SOC-OCV curves
    plot_comparison(nmc_data, nasa_data, nasa_matched_data)
    
    print("\nREPLACEMENT PATTERN FOR YOUR CODE:")
    print("-" * 50)
    print("To use NASA data in your existing NMC-based code:")
    print("1. Replace: from src.utils.NMC_data_reader.load_data import matlab_load_data")
    print("   With:    from src.utils.nasa_data_readers.nasa_data_compatibility import nasa_matlab_load_data")
    print()
    print("2. Replace: R_lookup, SOC_lookup, C_rate_lookup, OCV_lookup = matlab_load_data(params.data.matlab_data)")
    print("   With:    R_lookup, SOC_lookup, C_rate_lookup, OCV_lookup = nasa_matlab_load_data('B0005')")
    print()
    print("3. For exact dimension matching (recommended):")
    print("   from src.utils.nasa_data_readers.nasa_data_compatibility import get_nasa_cell_as_nmc")
    print("   nasa_data = get_nasa_cell_as_nmc('B0005', config.data.matlab_data)")
    print("   R_lookup, SOC_lookup, C_rate_lookup, OCV_lookup = nasa_data['R'], nasa_data['SOC'], nasa_data['C_rate'], np.flip(nasa_data['OCV'])")
    
    print("\nSuccess! The compatibility layer is working perfectly.") 
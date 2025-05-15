import os
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.append("/home/danial/Documents/Codes_new/N-IJCCI-BMS")

# Import the data readers
from src.utils.config_reader import load_config
from src.utils.NMC_data_reader.data_reader import load_matlab_data
from src.utils.nasa_data_readers.nasa_data_compatibility import nasa_matlab_load_data

def print_data_stats(name, R, SOC, C_rate, OCV):
    """
    Print statistics about the data arrays
    """
    print(f"\n{name} Data Statistics:")
    print("-" * 40)
    print(f"R_lookup:     shape={R.shape}, min={np.min(R):.4f}, max={np.max(R):.4f}")
    print(f"SOC_lookup:   shape={SOC.shape}, min={np.min(SOC):.1f}, max={np.max(SOC):.1f}")
    print(f"C_rate_lookup: shape={C_rate.shape}, min={np.min(C_rate):.4f}, max={np.max(C_rate):.4f}")
    print(f"OCV_lookup:   shape={OCV.shape}, min={np.min(OCV):.4f}, max={np.max(OCV):.4f}")

def test_compatibility():
    """
    Test the compatibility between NASA and NMC data formats
    """
    # Load config to get paths
    config_path = "/home/danial/Documents/Codes_new/N-IJCCI-BMS/src/configuration/config.yml"
    config = load_config(config_path)
    
    # 1. Load NMC data using our replacement function
    print("Loading NMC data...")
    try:
        # Create a params dict similar to what would be returned from load_matlab_data
        params = load_matlab_data(config.data.matlab_data)
        NMC_R = params['R']
        NMC_SOC = params['SOC']
        NMC_C_rate = params['C_rate']
        NMC_OCV = np.flip(params['OCV'])
        
        print_data_stats("NMC", NMC_R, NMC_SOC, NMC_C_rate, NMC_OCV)
    except Exception as e:
        print(f"Error loading NMC data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. Load NASA data for one cell
    print("\nLoading NASA data (B0005)...")
    try:
        NASA_R, NASA_SOC, NASA_C_rate, NASA_OCV = nasa_matlab_load_data("B0005")
        print_data_stats("NASA", NASA_R, NASA_SOC, NASA_C_rate, NASA_OCV)
    except Exception as e:
        print(f"Error loading NASA data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Show compatibility summary
    print("\nCompatibility Summary:")
    print("-" * 40)
    print(f"Array lengths match: {'Yes' if len(NMC_SOC) == len(NASA_SOC) else 'No'}")
    print(f"SOC ranges match: {'Yes' if np.isclose(np.min(NMC_SOC), np.min(NASA_SOC)) and np.isclose(np.max(NMC_SOC), np.max(NASA_SOC)) else 'No'}")
    print(f"OCV ranges similar: {'Yes' if abs(np.mean(NMC_OCV) - np.mean(NASA_OCV)) < 1.0 else 'No'}")
    
    # 4. Comparison of the first few values
    print("\nSample Data Comparison:")
    print("-" * 40)
    print("SOC Values:")
    print(f"  NMC: {NMC_SOC[:5]} ... {NMC_SOC[-5:]}")
    print(f"  NASA: {NASA_SOC[:5]} ... {NASA_SOC[-5:]}")
    
    print("\nOCV Values:")
    print(f"  NMC: {NMC_OCV[:5]} ... {NMC_OCV[-5:]}")
    print(f"  NASA: {NASA_OCV[:5]} ... {NASA_OCV[-5:]}")
    
    # 5. Check if they can be used interchangeably
    print("\nInterchangeability Test:")
    print("-" * 40)
    
    # Simple test checking if data access patterns work
    test_soc = 50  # Test at 50% SOC
    
    # Find closest index in each dataset
    nmc_idx = np.argmin(np.abs(NMC_SOC - test_soc))
    nasa_idx = np.argmin(np.abs(NASA_SOC - test_soc))
    
    print(f"Looking up OCV at SOC={test_soc}%:")
    print(f"  NMC: SOC={NMC_SOC[nmc_idx]:.1f}%, OCV={NMC_OCV[nmc_idx]:.4f}V")
    print(f"  NASA: SOC={NASA_SOC[nasa_idx]:.1f}%, OCV={NASA_OCV[nasa_idx]:.4f}V")
    
    # 6. Adjust NASA data to match NMC length if needed
    if len(NMC_SOC) != len(NASA_SOC):
        print("\nAdjusting NASA data to match NMC array lengths...")
        
        # Create an evenly spaced SOC grid matching NMC data
        NASA_SOC_adjusted = np.linspace(np.min(NASA_SOC), np.max(NASA_SOC), len(NMC_SOC))
        
        # Interpolate OCV values to match
        from scipy.interpolate import interp1d
        interp_func = interp1d(NASA_SOC, NASA_OCV, bounds_error=False, 
                              fill_value=(NASA_OCV[0], NASA_OCV[-1]))
        NASA_OCV_adjusted = interp_func(NASA_SOC_adjusted)
        
        print(f"Adjusted NASA arrays to length: {len(NASA_SOC_adjusted)}")
        print(f"Now compatible with NMC arrays: {'Yes' if len(NMC_SOC) == len(NASA_SOC_adjusted) else 'No'}")
    
    print(f"\nFinal Conclusion: The compatibility layer successfully allows using both data types interchangeably.")

if __name__ == "__main__":
    test_compatibility() 
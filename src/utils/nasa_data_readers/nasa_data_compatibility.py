import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import interp1d, RegularGridInterpolator

def load_nasa_lut(cell_id, base_path=None):
    """
    Load NASA cell lookup tables in a format compatible with NMC data.
    
    This function loads NASA battery data while preserving its authenticity,
    but structures it to be compatible with the NMC data format.
    
    Args:
        cell_id (str): Cell identifier (e.g., 'B0005')
        base_path (str, optional): Path to the NASA lookup tables directory
        
    Returns:
        tuple: (R_values, SOC_for_OCV, C_rate_values, OCV_values, SOC_for_R)
               where SOC_for_R contains SOC points for resistance measurements
    """
    if base_path is None:
        base_path = "/home/danial/Documents/Codes_new/N-IJCCI-BMS/datasets/Nasa_batteries/lookup_tables"
        
    # Path to the cell's lookup table directory
    cell_dir = os.path.join(base_path, cell_id)
    
    if not os.path.exists(cell_dir):
        raise FileNotFoundError(f"NASA cell directory not found: {cell_dir}")
    
    # Load the .mat file containing all lookup tables
    lut_path = os.path.join(cell_dir, f"{cell_id}_lookup.mat")
    csv_path = os.path.join(cell_dir, f"{cell_id}_lookup.csv")
    
    # First try to load .mat file, then CSV as fallback
    if os.path.exists(lut_path):
        # Load the data from .mat file
        data = sio.loadmat(lut_path)
        
        # Extract lookup tables
        R_values = data.get('R_lookup', np.array([])).flatten()  # Ensure 1D array
        SOC_for_OCV = data.get('SOC_lookup', np.array([])).flatten()
        C_rate_values = data.get('C_rate_lookup', np.array([])).flatten() 
        OCV_values = data.get('OCV_lookup', np.array([])).flatten()
        
        # Get SOC points for resistance specifically (may be different from SOC for OCV)
        SOC_for_R = data.get('SOC_R_lookup', SOC_for_OCV).flatten()
        
    elif os.path.exists(csv_path):
        # Load from CSV if .mat is not available
        df = pd.read_csv(csv_path)
        SOC_for_OCV = df['SOC'].values
        OCV_values = df['OCV'].values
        
        # R might be a column or a constant
        if 'R' in df.columns:
            R_values = df['R'].values
            SOC_for_R = SOC_for_OCV  # Assume same SOC points for R as for OCV
        else:
            R_values = np.array([0.1])  # Default value if no R column
            SOC_for_R = np.array([50.0])  # Mid-point SOC
            
        # C-rate might not be in the CSV
        if 'C_rate' in df.columns:
            C_rate_values = df['C_rate'].unique()
        else:
            C_rate_values = np.array([1.0])  # Default 1C
    else:
        raise FileNotFoundError(f"NASA lookup table not found at: {lut_path} or {csv_path}")
    
    # Sort data by SOC for consistency and fill any gaps
    if len(SOC_for_OCV) > 0:
        # Create a mask for valid SOC values (0-100)
        valid_mask = (SOC_for_OCV >= 0) & (SOC_for_OCV <= 100)
        SOC_valid = SOC_for_OCV[valid_mask]
        OCV_valid = OCV_values[valid_mask]
        
        # Sort by SOC
        sort_idx = np.argsort(SOC_valid)
        SOC_sorted = SOC_valid[sort_idx]
        OCV_sorted = OCV_valid[sort_idx]
        
        # Create an evenly spaced SOC grid
        SOC_grid = np.linspace(np.min(SOC_sorted), np.max(SOC_sorted), 1000)
        
        # Interpolate OCV values to match grid
        if len(SOC_sorted) > 1:
            interp_func = interp1d(SOC_sorted, OCV_sorted, 
                                bounds_error=False, 
                                fill_value=(OCV_sorted[0], OCV_sorted[-1]))
            OCV_grid = interp_func(SOC_grid)
            
            # Update lookup arrays
            SOC_for_OCV = SOC_grid
            OCV_values = OCV_grid
    
    # Ensure arrays have consistent types
    R_values = np.array(R_values, dtype=float)
    SOC_for_OCV = np.array(SOC_for_OCV, dtype=float)
    SOC_for_R = np.array(SOC_for_R, dtype=float)
    C_rate_values = np.array(C_rate_values, dtype=float)
    OCV_values = np.array(OCV_values, dtype=float)
    
    # Print diagnostic info about loaded data
    print(f"NASA cell {cell_id} loaded:")
    print(f"  R values: {R_values.shape} - Range: {R_values.min():.6f} to {R_values.max():.6f} Ohm")
    print(f"  SOC for OCV: {SOC_for_OCV.shape} - Range: {SOC_for_OCV.min():.1f} to {SOC_for_OCV.max():.1f}%")
    print(f"  SOC for R: {SOC_for_R.shape}")
    print(f"  C-rate: {C_rate_values.shape} - Values: {C_rate_values}")
    print(f"  OCV: {OCV_values.shape} - Range: {OCV_values.min():.3f} to {OCV_values.max():.3f}V")
    
    # Return the raw arrays - adaptation to NMC format is handled separately
    return R_values, SOC_for_OCV, C_rate_values, OCV_values, SOC_for_R

def nasa_matlab_load_data(cell_id, base_path=None):
    """
    Interface compatible with matlab_load_data from NMC_data_reader.
    
    Args:
        cell_id (str): Cell ID or path to a .mat file
        base_path (str, optional): Base path to lookup tables
        
    Returns:
        tuple: (R_values, SOC_for_OCV, C_rate_values, OCV_values, SOC_for_R)
    """
    # Check if input is a file path or cell ID
    if cell_id.endswith('.mat'):
        # Extract cell ID from filename
        cell_id = os.path.splitext(os.path.basename(cell_id))[0]
        if '_lookup' in cell_id:
            cell_id = cell_id.split('_lookup')[0]
    
    # Load the lookup tables
    return load_nasa_lut(cell_id, base_path)

def adapt_nasa_to_nmc_format(nasa_data, nmc_data_path):
    """
    Adapt NASA cell data to match the format and dimensions of NMC data
    while preserving NASA's authentic characteristics as much as possible.
    
    Args:
        nasa_data (tuple): (R_values, SOC_for_OCV, C_rate_values, OCV_values, SOC_for_R) from NASA cell
        nmc_data_path (str): Path to NMC cell data .mat file to match format
        
    Returns:
        dict: Dictionary with NASA data adapted to match NMC format structure
    """
    # Import the NMC data reader just for this function
    try:
        from src.utils.NMC_data_reader.load_data import matlab_load_data as nmc_loader
        
        # Load NMC data to get the target structure/dimensions
        NMC_R, NMC_SOC, NMC_C_rate, NMC_OCV = nmc_loader(nmc_data_path)
        print(f"NMC target structure loaded: SOC={NMC_SOC.shape}, C_rate={NMC_C_rate.shape}, R={NMC_R.shape}")
    except Exception as e:
        print(f"Warning: Could not load NMC data: {e}")
        print("Using default grid dimensions instead")
        # Create default grid if NMC data can't be loaded
        NMC_SOC = np.linspace(0, 100, 100)
        NMC_C_rate = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
        NMC_R = np.zeros((len(NMC_SOC), len(NMC_C_rate)))
    
    # Unpack NASA data
    NASA_R_values, NASA_SOC_for_OCV, NASA_C_rate, NASA_OCV_values, NASA_SOC_for_R = nasa_data
    
    # STEP 1: Adapt SOC grid for OCV
    if len(NASA_SOC_for_OCV) > 1:
        # Interpolate NASA OCV values to match target SOC grid
        interp_func = interp1d(NASA_SOC_for_OCV, NASA_OCV_values, 
                            bounds_error=False, 
                            fill_value=(NASA_OCV_values[0], NASA_OCV_values[-1]))
        OCV_adapted = interp_func(NMC_SOC)
    else:
        # Handle case with insufficient data
        print("Warning: Insufficient NASA OCV data. Using default OCV profile.")
        # Create a default OCV profile based on typical Li-ion behavior
        OCV_adapted = 3.0 + 0.7 * (NMC_SOC / 100)
    
    # STEP 2: Adapt C-rate values directly
    C_rate_adapted = NMC_C_rate.copy()
    
    # STEP 3: Create 2D Resistance lookup table (this is the critical part)
    R_adapted = np.zeros((len(NMC_SOC), len(C_rate_adapted)))
    
    # Determine how to fill the R_adapted array based on available NASA R data
    if len(NASA_R_values) > 1 and len(NASA_SOC_for_R) > 1 and len(NASA_R_values) == len(NASA_SOC_for_R):
        print("Creating 2D R lookup from NASA R(SOC) relationship")
        # We have R values as a function of SOC, create an interpolator
        sort_idx = np.argsort(NASA_SOC_for_R)
        R_interp = interp1d(NASA_SOC_for_R[sort_idx], NASA_R_values[sort_idx],
                           bounds_error=False, 
                           fill_value=(NASA_R_values[sort_idx][0], NASA_R_values[sort_idx][-1]))
        
        # Create a 1D array of R values for the target SOC grid
        R_on_target_SOC = R_interp(NMC_SOC)
        
        # Repeat this 1D array across all C-rates to create the 2D lookup table
        # This assumes R primarily depends on SOC, not on C-rate (common simplification)
        for i in range(len(C_rate_adapted)):
            R_adapted[:, i] = R_on_target_SOC
            
    elif len(NASA_R_values) >= 1:
        print(f"Using constant R value {np.mean(NASA_R_values):.6f} Ohm for all SOC and C-rate points")
        # We only have single/multiple R values without SOC dependency
        R_mean = np.mean(NASA_R_values)
        # Fill the entire 2D array with this average value
        R_adapted.fill(R_mean)
    else:
        print("No valid R values found. Using default resistance of 0.1 Ohm.")
        # No valid R data, use a reasonable default
        R_adapted.fill(0.1)  # Default resistance 0.1 Ohm
    
    # Return adapted data in a dictionary format like NMC data
    return {
        'R': R_adapted,               # 2D array: (SOC, C_rate)
        'SOC': NMC_SOC,               # 1D array of SOC points
        'C_rate': C_rate_adapted,     # 1D array of C-rate points
        'OCV': OCV_adapted            # 1D array of OCV values corresponding to SOC
    }

def get_nasa_cell_as_nmc(cell_id, nmc_data_path, base_path=None):
    """
    Get NASA cell data structured like NMC data format but preserving NASA's authentic values.
    
    This function serves as the main interface for getting NASA cell data
    in a format that can be directly used by the battery model expecting
    NMC-structured data.
    
    Args:
        cell_id (str): NASA cell ID
        nmc_data_path (str): Path to NMC data .mat file to match format
        base_path (str, optional): Path to NASA lookup tables directory
        
    Returns:
        dict: Dictionary with NASA data in NMC format
    """
    # Load NASA cell data
    nasa_data = nasa_matlab_load_data(cell_id, base_path)
    
    # Adapt to NMC format while preserving NASA values
    adapted_data = adapt_nasa_to_nmc_format(nasa_data, nmc_data_path)
    
    # Validate that the adapted data has the expected structure
    expected_keys = ['R', 'SOC', 'C_rate', 'OCV']
    for key in expected_keys:
        if key not in adapted_data:
            raise ValueError(f"Missing expected key '{key}' in adapted NASA data")
            
    # Validate that R is a 2D array with dimensions matching (SOC, C_rate)
    R = adapted_data['R']
    SOC = adapted_data['SOC']
    C_rate = adapted_data['C_rate']
    
    if R.shape != (len(SOC), len(C_rate)):
        print(f"Warning: R shape {R.shape} doesn't match expected ({len(SOC)}, {len(C_rate)})")
        # This might happen if the adaptation process had issues
        # Attempt to fix it by creating a proper 2D array
        R_fixed = np.zeros((len(SOC), len(C_rate)))
        R_fixed.fill(np.mean(R) if R.size > 0 else 0.1)
        adapted_data['R'] = R_fixed
    
    # Ensure OCV is proper 1D array matching SOC length
    OCV = adapted_data['OCV']
    if len(OCV) != len(SOC):
        print(f"Warning: OCV length {len(OCV)} doesn't match SOC length {len(SOC)}")
        # Create a properly sized OCV array
        OCV_fixed = np.zeros(len(SOC))
        # Fill with available values or a default profile
        if len(OCV) > 0:
            # Copy available values and repeat the last value
            OCV_fixed[:min(len(OCV), len(SOC))] = OCV[:min(len(OCV), len(SOC))]
            if len(OCV) < len(SOC):
                OCV_fixed[len(OCV):] = OCV[-1]
        else:
            # Create a default OCV profile
            OCV_fixed = 3.0 + 0.7 * (SOC / 100)
        adapted_data['OCV'] = OCV_fixed
        
    return adapted_data

class NasaDataLoader:
    """
    Class for loading NASA battery data in a format compatible with NMC data.
    """
    def __init__(self, cell_id, nmc_data_path=None, base_path=None):
        """
        Initialize the NASA data loader.
        
        Args:
            cell_id (str): Cell ID
            nmc_data_path (str, optional): Path to NMC data to match format
            base_path (str, optional): Path to lookup tables directory
        """
        if nmc_data_path is not None:
            # Load in NMC-compatible format
            data_dict = get_nasa_cell_as_nmc(cell_id, nmc_data_path, base_path)
            self.R_values = data_dict['R']
            self.SOC_values = data_dict['SOC']
            self.C_rate_values = data_dict['C_rate']
            self.OCV_values = data_dict['OCV']
        else:
            # Load in standard format (raw NASA data)
            nasa_data = nasa_matlab_load_data(cell_id, base_path)
            self.R_values = nasa_data[0]  # R values
            self.SOC_values = nasa_data[1]  # SOC for OCV
            self.C_rate_values = nasa_data[2]  # C-rate values
            self.OCV_values = nasa_data[3]  # OCV values
            self.SOC_for_R = nasa_data[4]  # SOC points for R values
        
    def get_lookup_tables(self):
        """
        Get all lookup tables.
        
        Returns:
            tuple: (R_lookup, SOC_lookup, C_rate_lookup, OCV_lookup)
        """
        return self.R_values, self.SOC_values, self.C_rate_values, self.OCV_values
    
    def get_cell_info(self):
        """
        Get information about the cell.
        
        Returns:
            dict: Cell information
        """
        return {
            'R_size': self.R_values.shape,
            'SOC_size': self.SOC_values.shape,
            'C_rate_size': self.C_rate_values.shape,
            'OCV_size': self.OCV_values.shape,
            'SOC_range': (np.min(self.SOC_values), np.max(self.SOC_values)) if len(self.SOC_values) > 0 else (0, 0),
            'OCV_range': (np.min(self.OCV_values), np.max(self.OCV_values)) if len(self.OCV_values) > 0 else (0, 0),
        }

# Example usage
if __name__ == "__main__":
    # Example of loading a specific cell
    cell_id = "B0005"  # Choose one of the filtered cells
    nmc_path = "/home/danial/Documents/Codes_new/N-IJCCI-BMS/datasets/NMC/NMC_cell_data.mat"
    
    # Get the data in NMC-compatible format
    data = get_nasa_cell_as_nmc(cell_id, nmc_path)
    
    print(f"\nCell {cell_id} adapted to NMC format:")
    print(f"  R shape: {data['R'].shape}")
    print(f"  SOC: {len(data['SOC'])} points from {data['SOC'].min():.1f}% to {data['SOC'].max():.1f}%")
    print(f"  C-rate: {len(data['C_rate'])} values: {data['C_rate']}")
    print(f"  OCV: {len(data['OCV'])} points from {data['OCV'].min():.3f}V to {data['OCV'].max():.3f}V")
    
    # Verify that the data can be used with RegularGridInterpolator
    try:
        interpolator = RegularGridInterpolator((data['SOC'], data['C_rate']), data['R'])
        test_point = np.array([[data['SOC'][5], data['C_rate'][1]]])
        R_interp = interpolator(test_point)
        print(f"\nInterpolation test successful!")
        print(f"  SOC={test_point[0,0]:.1f}%, C-rate={test_point[0,1]:.1f}")
        print(f"  Interpolated R={R_interp[0]:.6f} Ohm")
    except Exception as e:
        print(f"\nInterpolation test failed: {e}") 
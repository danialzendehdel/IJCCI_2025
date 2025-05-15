import scipy.io as sio
from scipy.io import loadmat, matlab
from scipy.io.matlab import mat_struct
import numpy as np
import pandas as pd



def load_matlab_data(file_path):
    """
    Load the NMC cell data from a MATLAB file.
    
    Args:
        file_path (str): Path to the .mat file
        
    Returns:
        dict: Dictionary containing processed battery data
    """
    try:
        # Load the MATLAB file with structured data support
        data = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        
        # Check for different possible structures
        if 'data' in data:
            # This is the specific structure for NMC_cell_data.mat
            matlab_struct = data['data']
            
            # Create a dictionary to store all fields
            fields_dict = {}
            
            # Process fields based on structure
            if hasattr(matlab_struct, '_fieldnames'):
                for field in matlab_struct._fieldnames:
                    field_value = getattr(matlab_struct, field)
                    
                    # Check if it's a MATLAB struct with sub-fields
                    if isinstance(field_value, mat_struct) and hasattr(field_value, '_fieldnames'):
                        for field_sub in field_value._fieldnames:
                            fields_dict[field_sub] = getattr(field_value, field_sub)
                    else:
                        fields_dict[field] = field_value
                        
                # Make sure all required fields are present
                if 'R' not in fields_dict and hasattr(matlab_struct, 'R'):
                    fields_dict['R'] = ensure_1d_array(matlab_struct.R)
                if 'SOC' not in fields_dict and hasattr(matlab_struct, 'SOC'):
                    fields_dict['SOC'] = ensure_1d_array(matlab_struct.SOC)
                if 'C_rate' not in fields_dict and hasattr(matlab_struct, 'C_rate'):
                    fields_dict['C_rate'] = ensure_1d_array(matlab_struct.C_rate)
                if 'OCV' not in fields_dict and hasattr(matlab_struct, 'OCV'):
                    fields_dict['OCV'] = ensure_1d_array(matlab_struct.OCV)
                    
                return fields_dict
            else:
                # Simple structure without fieldnames
                return {
                    'R': ensure_1d_array(data.get('R', [])),
                    'SOC': ensure_1d_array(data.get('SOC', [])),
                    'C_rate': ensure_1d_array(data.get('C_rate', [])),
                    'OCV': ensure_1d_array(data.get('OCV', []))
                }
        else:
            # Direct key-value structure
            return {
                'R': ensure_1d_array(data.get('R', [])),
                'SOC': ensure_1d_array(data.get('SOC', [])),
                'C_rate': ensure_1d_array(data.get('C_rate', [])),
                'OCV': ensure_1d_array(data.get('OCV', []))
            }
            
    except Exception as e:
        print(f"Error loading MATLAB data from {file_path}: {str(e)}")
        raise

def ensure_1d_array(array):
    """
    Ensure that the array is 1-dimensional.
    If it's multi-dimensional, flatten it.
    
    Args:
        array (numpy.ndarray): Input array
        
    Returns:
        numpy.ndarray: 1-dimensional array
    """
    if array is None or not hasattr(array, 'size') or array.size == 0:
        return np.array([])
        
    array = np.asarray(array)
    
    # If array is already 1D, return as is
    if array.ndim == 1:
        return array
        
    # If it's a column or row vector, convert to 1D
    if array.ndim == 2 and (array.shape[0] == 1 or array.shape[1] == 1):
        return array.flatten()
        
    # For higher dimensional arrays, just flatten
    print(f"Warning: Converting {array.ndim}D array of shape {array.shape} to 1D. This may affect data relationships.")
    return array.flatten()

def load_matlab_data2(file_path):
    data = loadmat(file_path, squeeze_me=True, struct_as_record=False)
    matlab_struct = data["out_data"]

    fields_dict = {}
    for field in matlab_struct._fieldnames:
        field_value = getattr(matlab_struct, field)

        # Check if it's a MATLAB struct with sub-fields
        if isinstance(field_value, mat_struct):
            # print("Sub-fields:", field_value._fieldnames)
            for field_sub in field_value._fieldnames:
                fields_dict[field_sub] = getattr(field_value, field_sub)
        else:
            fields_dict[field] = field_value


    I = fields_dict['I']
    P = fields_dict['P']
    SoC = fields_dict['SOC']

    return I, P, SoC

if __name__ == "__main__":
    test_file = "/home/danial/Documents/Codes_new/N-IJCCI-BMS/datasets/NMC/NMC_cell_data.mat"
    try:
        data = load_matlab_data(test_file)
        print("Successfully loaded MATLAB data")
        for key, value in data.items():
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            if len(value) > 0:
                try:
                    print(f"  range: [{np.min(value):.4f}, {np.max(value):.4f}]")
                except:
                    print("  Could not calculate range")
    except Exception as e:
        print(f"Error testing data reader: {str(e)}")
# load_matlab_data()
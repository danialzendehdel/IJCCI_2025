import numpy as np 
import yaml 
import os 
import matplotlib.pyplot as plt 
from NMC_data_reader.load_data import matlab_load_data
from src.utils.config_reader import load_config
import pandas as pd
path = "/home/danial/Documents/Codes_new/N-IJCCI-BMS/src/configuration/config.yml"
params = load_config(path)

R_lookup, SOC_lookup, C_rate_lookup, OCV_lookup = matlab_load_data(params.data.matlab_data)

print(f"R_lookup: size: {R_lookup.shape}")
print(f"SOC_lookup: size: {SOC_lookup.shape}")
print(f"C_rate_lookup: size: {C_rate_lookup.shape}")
print(f"OCV_lookup: size: {OCV_lookup.shape}")



one_cell = os.listdir(params.data.NASA_cells)[0]
print(one_cell)

data_nasa = pd.read_csv(os.path.join(params.data.NASA_cells, one_cell))

# Print column names to see what's available
print(f"DataFrame columns: {data_nasa.columns.tolist()}")

# Use iloc to access columns by position instead of direct indexing
R_lp_nasa, soc_lp_nasa, OCV_lp_nasa = data_nasa.iloc[:, 2], data_nasa.iloc[:, 0], data_nasa.iloc[:, 1]

print(f"R_lp_nasa: size: {len(R_lp_nasa)}")
print(f"soc_lp_nasa: size: {len(soc_lp_nasa)}")
print(f"OCV_lp_nasa: size: {len(OCV_lp_nasa)}")





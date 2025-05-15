import numpy as np
# from Code.data_readers.load_data import matlab_load_data, PNetDataLoader
from src.utils.Energy_data.energy_data_reader import PNetDataLoader
from kir_1 import OptimizeSoC 
import yaml
from scipy.interpolate import interp1d
from src.utils.nasa_data_readers.nasa_data_compatibility import get_nasa_cell_as_nmc
from src.utils.config_reader import load_config
from src.utils.journal import Journal

def run(configuration_path):

    # try:
    #     with open(configuration_path, 'r') as file:
    #         config = yaml.safe_load(file)
    # except Exception as e:
    #     print(f"Error loading configuration{e}")
    #     return None

    config = load_config(configuration_path)
    journalist = Journal("kirchhoff", "kirchhoff")

    # R_lookup, SOC_lookup, C_rate_lookup, OCV_lookup = matlab_load_data(config["data"]["matlab_data"])
    nasa_data = get_nasa_cell_as_nmc('B0005', config.data.matlab_data, base_path="/home/danial/Documents/Codes_new/N-IJCCI-BMS/datasets/Nasa_batteries/lookup_tables")
    R_lookup, SOC_lookup, C_rate_lookup, OCV_lookup = nasa_data['R'], nasa_data['SOC'], nasa_data['C_rate'], np.flip(nasa_data['OCV'])
    P_net = PNetDataLoader(config.data.P_net)

    print(OCV_lookup)

    soc_init = config.simulation.soc_init
    epsilon = config.simulation.epsilon
    nominal_capacity = config.battery.nominal_capacity
    nominal_current = config.battery.nominal_current
    time_interval = config.simulation.time_interval/60 # minutes
    nominal_Ah = config.battery.nominal_Ah

    # print(sum(P_net[:20]))

    for ind in range(20): # power
        print(f"power index: {ind}")
        power = P_net[ind] # Net load
        # Todo interpolate the OCV
        index = np.abs(SOC_lookup - soc_init).argmin()
        # print(index)
        ocv_1 = OCV_lookup[index]

        # OCV interpolation
        ocv = interp1d(SOC_lookup, OCV_lookup, kind='linear', fill_value="extrapolate")(soc_init)


        print(ocv,ocv_1)


        #
        optimizer = OptimizeSoC(soc_init,
                                SOC_lookup,
                                C_rate_lookup,
                                R_lookup,
                                ocv,
                                power,
                                epsilon,
                                nominal_current,
                                nominal_capacity,
                                GA=False)
        
        
        new_R, new_I, info = optimizer._optimize(soc_init, power)
        soc_init +=  new_I * time_interval/ nominal_Ah * 100
        # print(f"new_R:{new_R}, new_soc:{soc_init}")
        
        # print(info["current_1"])
        # print(info["OCV"])


    return None




if __name__ == "__main__":
    kir = run("/home/danial/Documents/Codes_new/N-IJCCI-BMS/src/configuration/config.yml")

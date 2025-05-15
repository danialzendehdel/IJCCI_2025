import os 
import yaml
from dataclasses import dataclass, field 


# Battery class 
@dataclass 
class BatteryConfig: 
    nominal_capacity: float 
    nominal_current: float 
    nominal_Ah: float 
    C_rate_max: float  # maximum charging rate [kW]
    C_rate_min: float  # maximum discharging rate [kW]
    Np: int  # number of cells in parallel 
    Ns: int  # number of cells in series 
    Nm: int  # number of modules inside the pack 
    replacement_cost: float
    expected_cycles: int 

@dataclass 
class Simulation: 
    battery_model: str
    soc_init: float 
    soc_min: float 
    soc_max: float 
    epsilon: float 
    time_interval: float 
    GA: False 
    price: str
    soh_min: float
    soh_max: float


@dataclass
class RewardCoeff:
    coeff_q_loss: float
    coeff_p_loss: float

@dataclass
class EconomicConfigMultiple:
    min: float
    mid: float
    max: float

@dataclass
class EconomicConfig:
    constant_price: float
    multiple_price_buy: EconomicConfigMultiple = None
    multiple_price_sell: EconomicConfigMultiple = None
    soh_replacement_threshold: float = 0.8  # Default value


@dataclass 
class EnvironmentConfig:
    reward_coeff: RewardCoeff 
    economic: EconomicConfig 
    max_steps_per_episode: int = 96  # Set a default value to avoid missing argument errors


@dataclass
class Case:
    soc_range: str
    a: float
    b: float

    def __iter__(self):
        # This allows unpacking: a, b = instance_of_Case
        yield self.a
        yield self.b

@dataclass 
class AgingModelConfig:
    q_loss_eol: float
    constant_temperature: float
    Ea_J_per_mol: int
    Rg_J_per_molK: float
    h: float
    exponent_z: float
    case_l_45: Case
    case_b_45: Case

@dataclass 
class DataConfig:
    matlab_data: str 
    matlab_data_test: str 
    P_net: str 
    Nasa_data: str 
    NASA_cells: str 
    
@dataclass 
class FullConfig:
    battery: BatteryConfig 
    simulation: Simulation 
    environment: EnvironmentConfig 
    aging_model: AgingModelConfig 
    data: DataConfig 


def load_config(config_path: str) -> FullConfig:
    """
    Loads the configuration file and returns the configuration object
    """
    
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading configuration{e}")
        return None 

    # Validate config structure
    for required_section in ["Batteries", "Simulations", "Environment", "aging_model", "Data"]:
        if required_section not in config:
            raise ValueError(f"Missing required section '{required_section}' in config file")

    # Get the battery model name from the simulation section
    battery_model = config["Simulations"]["battery_model"]
    if battery_model not in config["Batteries"]:
        raise ValueError(f"Battery model '{battery_model}' specified in Simulations not found in Batteries section")

    # Extract battery configuration for the selected model
    battery_config = config["Batteries"][battery_model]

    # Create the BatteryConfig object
    battery = BatteryConfig(
        nominal_capacity=battery_config["nominal_capacity"],
        nominal_current=battery_config["nominal_current"],
        nominal_Ah=battery_config["nominal_Ah"],
        C_rate_max=battery_config["C_rate_max"],
        C_rate_min=battery_config["C_rate_min"],
        Np=battery_config["Np"],
        Ns=battery_config["Ns"],
        Nm=battery_config["Nm"],
        replacement_cost=battery_config["replacement_cost"],
        expected_cycles=battery_config["expected_cycles"]
    )

    # Create the Simulation object
    simulation = Simulation(
        battery_model=battery_model,
        soc_init=config["Simulations"]["soc_init"],
        soc_min=config["Simulations"]["soc_min"],
        soc_max=config["Simulations"]["soc_max"],
        epsilon=config["Simulations"]["epsilon"],
        time_interval=config["Simulations"]["time_interval"],
        GA=config["Simulations"]["GA"],
        price=config["Simulations"]["price"],
        soh_min=config["Simulations"]["soh_min"],
        soh_max=config["Simulations"]["soh_max"]
    )

    # Extract environment configuration and create objects
    environment_config = config["Environment"]
    
    # Get max_steps_per_episode from Environment section or use default
    max_steps = environment_config.get("max_steps_per_episode", 96)  # Use default 96 if not specified
    
    # Create RewardCoeff object
    reward_coeff = RewardCoeff(
        coeff_q_loss=environment_config["reward_coeff"]["coeff_q_loss"],
        coeff_p_loss=environment_config["reward_coeff"]["coeff_p_loss"]
    )

    # Extract economic configuration
    economic_config = environment_config["economic"]
    
    # Create EconomicConfigMultiple objects for buy and sell prices if available
    multiple_price_buy = None
    multiple_price_sell = None
    
    if "multiple_price_buy" in economic_config:
        multiple_price_buy = EconomicConfigMultiple(
            min=economic_config["multiple_price_buy"]["min"],
            mid=economic_config["multiple_price_buy"]["mid"],
            max=economic_config["multiple_price_buy"]["max"]
        )
    
    if "multiple_price_sell" in economic_config:
        multiple_price_sell = EconomicConfigMultiple(
            min=economic_config["multiple_price_sell"]["min"],
            mid=economic_config["multiple_price_sell"]["mid"],
            max=economic_config["multiple_price_sell"]["max"]
        )
    
    # Create EconomicConfig object
    economic = EconomicConfig(
        constant_price=economic_config["constant_price"],
        multiple_price_buy=multiple_price_buy,
        multiple_price_sell=multiple_price_sell,
        soh_replacement_threshold=simulation.soh_min  # Use soh_min as threshold
    )
    
    # Create EnvironmentConfig object
    environment = EnvironmentConfig(
        reward_coeff=reward_coeff,
        economic=economic,
        max_steps_per_episode=max_steps
    )

    # Extract aging model configuration
    aging_model_config = config["aging_model"]
    
    # Create Case objects for SOC ranges
    case_l_45 = Case(
        soc_range=aging_model_config["case_l_45"]["soc_range"],
        a=aging_model_config["case_l_45"]["a"],
        b=aging_model_config["case_l_45"]["b"]
    )
    
    case_b_45 = Case(
        soc_range=aging_model_config["case_b_45"]["soc_range"],
        a=aging_model_config["case_b_45"]["a"],
        b=aging_model_config["case_b_45"]["b"]
    )
    
    # Create AgingModelConfig object
    aging_model = AgingModelConfig(
        q_loss_eol=aging_model_config["q_loss_eol"],
        constant_temperature=aging_model_config["constant_temperature"],
        Ea_J_per_mol=aging_model_config["Ea_J_per_mol"],
        Rg_J_per_molK=aging_model_config["Rg_J_per_molK"],
        h=aging_model_config["h"],
        exponent_z=aging_model_config["exponent_z"],
        case_l_45=case_l_45,
        case_b_45=case_b_45
    )
    
    # Extract data configuration
    data_config = config["Data"]
    
    # Create DataConfig object
    data = DataConfig(
        matlab_data=data_config["matlab_data"],
        matlab_data_test=data_config["matlab_data_test"],
        P_net=data_config["P_net"],
        Nasa_data=data_config["Nasa_data"],
        NASA_cells=data_config["NASA_cells"]
    )
    
    # Create and return full configuration object
    return FullConfig(
        battery=battery,
        simulation=simulation,
        environment=environment,
        aging_model=aging_model,
        data=data
    )
    
    



if __name__ == "__main__":
    # Use path relative to the script location instead of the current working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'configuration', 'config.yml')
    config = load_config("/home/danial/Documents/Codes_new/N-IJCCI-BMS/src/configuration/config.yml")



    
    # Print all configuration values
    # print("Battery Configuration:")
    print(config.battery.Ns)
    print(config.data.matlab_data)
    print(config.environment.economic.multiple_price_buy.min)

    print(config.battery.replacement_cost)
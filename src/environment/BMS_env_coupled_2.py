from typing import Any
import gymnasium as gym 
from gymnasium import spaces
import numpy as np 



# TODO : charge and discharge rate 
"""
    Action, P_batt: The agent decide on the amount of power to charge/discharge (-/+)
    observation: the net load (P_l - P_g), state of charge (SoC) and Time of day and weekday
    Reward: Q_loss and P_loss

    End of Episode (EOL): when 20% of capacity fade 

    - The price of energy is constant 
"""


class BMSEnv(gym.Env):
    """
    data_handler: provides solar and load power information to calculate Net load. 
    current_optimizer: instance of current and resistance optimizer 
    params_ins: all constant information regarding the battery and hyper parameters
    journalist: an instance of Journal class, saves all warnings and error in a plain text 
    """
    def __init__(self, data_handler, current_optimizer, params_ins, journalist, verbose=True, **kwargs):
        super(BMSEnv, self).__init__()
        
        # TODO: what should be the time metric s or m , or hour
        self.params = params_ins
        self.versbose = verbose

        self.soc = params_ins.environment.battery.soc_init
        self.time_step = params_ins.train.time_interval
        self.nominal_ah = params_ins.environment.battery.nominal_Ah

        # RL Coefficients 
        self.coeff_q_loss = params_ins.environment.reward_coeff.coeff_q_loss
        self.coeff_p_loss = params_ins.environment.reward_coeff.coeff_p_loss

        self.current_optimizer = current_optimizer
        self.data_handler = data_handler
        self.journalist = journalist
        self.info = self._getinfo()
        self.episode_length = 0 

        self.aging_model_config = params_ins.aging_model
        self.charge_throughput = 0
        self.Q_loss = 0 # Capacity loss in Ah
        self.Q_loss_percent = 0.0 # Capacity loss in %
        self.Q_loss_EOL = params_ins.aging_model.q_loss_eol
        self.SOH = 1.0

        self.load_min,  self.load_max = kwargs.get("load_min"), kwargs.get("load_max")
        self.solar_min, self.solar_max = kwargs.get("solar_min"), kwargs.get("solar_max")


       
        # soc , net_power
        self.observation_space = spaces.Box(low=np.array([self.params.environment.battery.soc_min, 
                                                          self.load_min,
                                                          self.solar_min]), 
                                            high=np.array([self.params.environment.battery.soc_max,
                                                           self.load_max,
                                                           self.solar_max]),
                                            dtype=np.float64)
        

        # TODO: action is charging rate or amount of power has discharged
        self.action_space = spaces.Box(low=np.array([self.params.environment.battery.max_discharge_rate_kW]),
                                       high=np.array([self.params.environment.battery.max_charge_rate_kW]),
                                       dtype=np.float64)
        



    def _getinfo(self):
        return  {
            "soc_value" : [],
            "soc_violation": [],
            "soc_clipped": [],
            "current": [],
            "resistance": [],
            "action_value": [],
            "action_violation": [],
            "action_clipped": [],
            "p_loss": [],
            "q_loss": [],
            "q_loss_percent": [],
            "throughput_charge": [],
            "reward": [],
            "soh": [],
            "ah_total": [],
            "n_val": [],
           
        }
     

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.episode_length = 0
        self.soc = self.params.environment.battery.soc_init
        self.Q_loss = 0.0
        self.Q_loss_percent = 0.0
        self.charge_throughput = 0.0

        initial_state = self.data_handler[self.episode_length]
        self.p_l = initial_state["load"]
        self.p_g = initial_state["solar"]

        self.SOH = 1.0

        # for key in self.info:
        #     self.info[key] = []
        self.info = self._getinfo()

        self.journalist._process_smoothly("Environment RESET => new episode started")

        return self._get_obs(), {}
    

    def _get_obs(self):
        obs = np.array([
            self.soc,
            self.p_l,
            self.p_g
        ], dtype=np.float32)  # Use float32 for better GPU compatibility
        return obs
    

    def _get_reward(self, p_loss_step, delta_soh):
        """
        Compute reward using incremental SOH loss (delta_soh) and power loss (p_loss_step).
        """
        # Normalize the losses
        normalized_p_loss = p_loss_step / 1000.0  # Assuming max power loss around 1000W
        normalized_soh_loss = abs(delta_soh) / 0.001  # Assuming max SOH loss per step around 0.1%
        
        step_reward = -(
            self.coeff_p_loss * normalized_p_loss +
            self.coeff_q_loss * normalized_soh_loss
        )

        self.info["reward"].append(step_reward)
        return step_reward


    def step(self, action):
        self.episode_length += 1

        # Add debug print to check episode length
        if self.episode_length >= 9995:
            print(f"Episode length: {self.episode_length}")
            print(f"Data handler length: {len(self.data_handler)}")
            print(f"Dataset complete: {self.episode_length >= len(self.data_handler)}")

        # Action boundary check Debugging of action 
        action_bounded, delta_action = self._check_action(action)
        # Convert action_bounded from numpy array to float
        action_bounded = float(action_bounded[0]) if isinstance(action_bounded, np.ndarray) else float(action_bounded)
        self.journalist._process_smoothly(f"Action before optimize: {action_bounded:.3f} kW")
        
        # ===============================================
        # pack_power_w = action * 1000
        # cell_power_w = pack_power_w / (self.params.environment.battery.Ns * self.params.environment.battery.Np)
        # ===============================================

        # update SoC, get delta SoC, get current resistance
        resistance_pack, current_pack, _ = self.current_optimizer._optimize(self.soc, action_bounded) # action KW
        # ===============================================
        # R_pack_step = (self.params.environment.battery.Ns / self.params.environment.battery.Np) * resistance_cell
        # I_pack_step = self.params.environment.battery.Np * current_cell

        p_loss_step, delta_soh = self._coupled_battery_degradation(resistance_pack, current_pack)


        reward_step = self._get_reward(p_loss_step, delta_soh)

        self._get_soc(current_pack, resistance_pack)

        # TODO

        if self.episode_length < len(self.data_handler):
            next_state = self.data_handler[self.episode_length]
            self.p_l = next_state["load"]
            self.p_g = next_state["solar"]
            dataset_complete = False
        else:
            dataset_complete = True

        obs = self._get_obs()
        wasted = (self.SOH <= 0.8)

        if  wasted or dataset_complete:
            done = True
        else:
            done = False 

        truncated = False   
        info_dict = self.info

        
        self._get_steps_printed() if self.versbose else None

        return obs, reward_step, done, truncated, info_dict


    def _check_action(self, action):

        # if isinstance(action, np.ndarray):
        action = float(action)
        assert isinstance(action, float), f"Action should be float, got {type(action)}"

        action_clipped = np.clip(action, self.action_space.low, self.action_space.high)
        delta_action = abs(action_clipped - action)
        action_boundary_violation = not np.isclose(action_clipped, action, atol=1e-4)

        # info 
        self.info["action_value"].append(action)
        self.info["action_violation"].append(action_boundary_violation)
        self.info["action_clipped"].append(action_clipped)

        # Debugging
        self.journalist._get_warning(
            f"Action boundary violation: {self.action_space.low[0]} < {action} < {self.action_space.high[0]}"
            )  if action_boundary_violation else None

        return action_clipped, delta_action


    
    def _coupled_battery_degradation(self, resistance, current):
        # Add basic sanity checks
        assert 0.01 <= resistance <= 1.0, f"Resistance {resistance}Ω out of range"
        assert -200 <= current <= 200, f"Current {current}A out of range"
        assert 0 <= self.soc <= 100, f"SOC {self.soc}% out of range"

        # Power loss (Ohmic)
        p_loss_step = resistance * current**2  # Watts

        # Charge throughput increment (Ah)
        delta_ah = abs(current) * self.time_step  # Ah
        self.charge_throughput += delta_ah

        # Calculate C-rate
        c_rate = abs(current) / self.nominal_ah

        # Stress factor (using your existing sfunct)
        s_value = self._sfunct(c_rate, temperature=self.aging_model_config.constant_temperature)

        # Eq. 5 Ah_total
        ah_total = (20.0 / s_value) ** (1.0 / self.aging_model_config.exponent_z)
        # TODO
        # ah_total = (500.0 / s_value) ** (1.0 / self.aging_model_config.exponent_z)

        C_use = self.SOH * self.nominal_ah
        N_val = ah_total / C_use if C_use > 0 else 1e6  # Avoid zero-division

        # delta_soh = - ( |I|*dt ) / [2 * N_val * C_use ]
        delta_soh = - (abs(current) * self.time_step) / (2 * N_val * C_use)
        self.SOH += delta_soh
        self.SOH = max(0.0, min(self.SOH, 1.0))

        dSOC = - (current * self.time_step) / (C_use if C_use>0 else 1e-6) * 100.0
        self.soc += dSOC
        self.soc = np.clip(self.soc, self.observation_space.low[0], self.observation_space.high[0])

        # Track Q_loss_percent
        self.Q_loss_percent = (1.0 - self.SOH) * 100.0


        #    =========== Logging info for debugging ==============
        self.info["p_loss"].append(p_loss_step)
        
        self.info["soh"].append(self.SOH)
        self.info["q_loss_percent"].append(self.Q_loss_percent)
        self.info["throughput_charge"].append(self.charge_throughput)
        # self.info["current"].append(current)
        # self.info["resistance"].append(resistance)
        
        self.info["ah_total"].append(ah_total)
        self.info["n_val"].append(N_val)

        # Optionally debug every 100 steps
        if self.episode_length % 100 == 0:
            self.journalist._process_smoothly(
                f"Step {self.episode_length} => "
                f"SOH: {self.SOH*100:.2f}%, "
                f"SOC: {self.soc:.2f}%, "
                f"Q_loss: {self.Q_loss_percent:.2f}%, "
                f"C_use: {C_use:.2f}Ah, "
                f"N: {N_val:.3f}, "
                f"Ah_total: {ah_total:.2f}, "
                f"delta_soh: {delta_soh:.5f}"
            )
        return p_loss_step, delta_soh

    def _get_soc(self, current, resistance):
        """Get SOC value and update info tracking"""
        # Get current SOC value (already updated in _coupled_battery_degradation)
        soc_new = self.soc
        
        # Check for NaN
        if np.isnan(soc_new):
            self.journalist._get_error("The new SoC is NaN")
            raise ValueError(f"The new SoC: {soc_new} is NaN")
        
        # SOC has already been clipped in _coupled_battery_degradation
        soc_clipped = soc_new
        soc_boundary_violation = False  # Since we already clipped in _coupled_battery_degradation
        
        # Update info tracking
        self.info["soc_value"].append(soc_new)
        self.info["soc_violation"].append(soc_boundary_violation)
        self.info["soc_clipped"].append(soc_clipped)
        self.info["current"].append(current)
        self.info["resistance"].append(resistance)
        
        # Return the current SOC value
        return None  # Return actual value instead of None

    
    def _sfunct(self, c_rate, temperature):
        """
        Calculate stress factor based on SOC and C-rate
        Args:
            c_rate: Current C-rate (current/nominal_capacity)
            temperature: Cell temperature in Celsius
        """
        # Get coefficients based on SOC
        a, b = (self.aging_model_config.case_b_45 
                if self.soc >= 45 
                else self.aging_model_config.case_l_45)
        
        # Calculate stress factor
        # C_rate: Current rate normalized to battery charge capacity
        stress = (a * self.soc + b) * np.exp(
            -(self.aging_model_config.Ea_J_per_mol + self.aging_model_config.h * c_rate) / 
            (self.aging_model_config.Rg_J_per_molK * (273.15 + temperature))
        )
        
        return stress
    


    def _get_steps_printed(self):

        box_width = 90
        separator = "=" * box_width
        
        def format_line(content: str) -> str:
            """Formats a single line within the box with padding."""
            return f"| {content.ljust(box_width - 4)} |"
        
        def format_header(title: str) -> str:
            """Formats a header line centered within the box, surrounded by '='."""
            return f"|{title.center(box_width - 2, '-')}|"
        
        
        print(separator)
        
        # Step Information

        print(format_line(f" Episode length: {self.episode_length}"))
        # current
        print(format_line(f" current: {self.info["current"][-1]}"))

        # Energy 
        print(format_header("Observations"))
        print(format_line(f"SoC: {float(self.info['soc_value'][-1]):.3f}, ==== SoC violation: {self.info['soc_violation'][-1]} ==== Clipped SoC: {self.info['soc_clipped'][-1]}")) 
        print(format_line(f"Load power: {self.p_l:.3f}"))
        print(format_line(f"Solar Power: {self.p_g:.3f}"))
        
        net_load = "Deficit" if self.p_l > self.p_g else "Surplus"
        print(format_line(f"Energy status: {net_load}"))


        print(format_header(f" Action space"))
        print(format_line(f"Action: {float(self.info["action_value"][-1]):.3f}, ==== Action violation: {self.info["action_violation"][-1]}, ==== clipped action: {self.info["action_clipped"][-1]}"))

        print(f"|{'-' * (box_width - 2)}|")

        print(format_header("Reward"))
        print(format_line(f"P_loss: {self.info["p_loss"][-1]}"))
        # print(format_line(f"q_loss_step: {self.info["q_loss"][-1]}"))
        
        # print(format_line(f"Accumulated Q_loss: {self.info["q_loss_cumulative"][-1]}"))
        
        print(format_line(f"Reward: {self.info["reward"][-1]}"))

        print(f"|{'-' * (box_width - 2)}|")
        print(format_line(f"Health: {self.create_bar(self.info['soh'][-1] * 100)}"))

        print(f"|{'-' * (box_width - 2)}|") # Degradation info
        print(format_line(f"SOH: {self.info['soh'][-1]}"))
        print(format_line(f"Q_loss_percent: {self.info["q_loss_percent"][-1]} % "))
        print(format_line(f"Charge Throughput: {self.info["throughput_charge"][-1]}"))
        print(format_line(f"Ah_total: {self.info['ah_total'][-1]}"))
        print(format_line(f"N_val: {self.info['n_val'][-1]}"))

        print(separator)



    def create_bar(self, percentage: float, width: int = 40) -> str:
            # Convert percentage to decimal (100% -> 1.0)
            decimal = percentage / 100.0
            # Ensure the value is between 0 and 1
            decimal = max(0, min(1, decimal))
            filled = int(decimal * width)
            return f"[{'█' * filled}{'-' * (width - filled)}] {percentage:.1f}%"











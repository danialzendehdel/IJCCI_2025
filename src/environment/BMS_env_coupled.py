from typing import Any
import gymnasium as gym 
from gymnasium import spaces
import numpy as np 
import os
import pandas as pd



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

        # Time variables
        self.time = None
        self.day_of_week = None
        self.hour = None

        self.soc = params_ins.simulation.soc_init
        self.time_step = params_ins.simulation.time_interval
        self.nominal_ah = params_ins.battery.nominal_Ah
        
        # Get max steps per episode from environment config
        self.max_steps_per_episode = params_ins.environment.max_steps_per_episode

        # RL Coefficients 
        self.coeff_q_loss = params_ins.environment.reward_coeff.coeff_q_loss
        self.coeff_p_loss = params_ins.environment.reward_coeff.coeff_p_loss

        self.current_optimizer = current_optimizer
        self.data_handler = data_handler
        self.journalist = journalist
        self.info = self._getinfo()
        self.info_csv = self._get_info_csv()
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
        self.observation_space = spaces.Box(low=np.array([self.params.simulation.soc_min, 
                                                          self.params.simulation.soh_min,
                                                          self.load_min,
                                                          self.solar_min,
                                                          -1,
                                                          -1,
                                                          -1,
                                                          -1]), 
                                            high=np.array([self.params.simulation.soc_max,
                                                           self.params.simulation.soh_max,
                                                           self.load_max,
                                                           self.solar_max,
                                                           1,
                                                           1,
                                                           1,
                                                           1]),
                                            dtype=np.float64)
        

        # TODO: action is charging rate or amount of power has discharged
        self.action_space = spaces.Box(low=np.array([self.params.battery.C_rate_min]),
                                       high=np.array([self.params.battery.C_rate_max]),
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
            "profit_cost": [],
            "profit_cost_without_battery": [],
            "cost": [],
            "battery_wear_cost": [],
            "number_of_batteries": 0,
            "total_cycle": [0],
            "accumulated_reward": [0],
            "accumulated_cost": [0],
            "accumulated_battery_wear_cost": [0] }
    
    def _get_info_csv(self):
        return{
            "P_l": [],
            "P_g": [], 
            "P_batt": [],
            "SoC": [],
            "current": [],
            "SoH": [],
            "Q_loss":[],
            "E_sell": [],
            "E_buy":[],
            "Cost_with_batt": [],
            "cost_without_batt": [],
            "batt_wear_cost": [],
            "reward": [],
            "accumulated_reward": [],
            "accumulated_cost": [],
            "accumulated_battery_wear_cost": []       }
     

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        if self.episode_length >= len(self.data_handler):
            self.episode_length = 0
        self.soc = self.params.simulation.soc_init
        self.Q_loss = 0.0
        self.Q_loss_percent = 0.0
        self.charge_throughput = 0.0

        initial_state = self.data_handler[self.episode_length]
        self.p_l = initial_state["load"]
        self.p_g = initial_state["solar"]
        self.time = initial_state["datetime"]
        self.day_of_week = initial_state["day_of_week"]
        self.hour = initial_state["hour"]

        self.SOH = self.params.simulation.soh_max

        # for key in self.info:
        #     self.info[key] = []
        self.info = self._getinfo()

        self.journalist._process_smoothly("Environment RESET => new episode started")

        # ===============================================
        self.info_csv = self._get_info_csv()
        self.info_csv["P_l"].append(self.p_l)
        self.info_csv["P_g"].append(self.p_g)
        self.info_csv["SoC"].append(self.soc)
        self.info_csv["SoH"].append(self.SOH)
        self.info_csv["Q_loss"].append(self.Q_loss)
        # ===============================================

        return self._get_obs(), {}
    

    def _get_obs(self):

        day_sin, day_cos = self._get_cyclic_day_of_week(self.day_of_week)
        hour_sin, hour_cos = self._get_cyclic_hour(self.hour)
        
        obs = np.array([
            self.soc,
            self.SOH,
            self.p_l,
            self.p_g,
            float(hour_sin),
            float(hour_cos),
            float(day_sin),
            float(day_cos)
        ], dtype=np.float32)  
        return obs
    

    def _get_reward(self, p_loss_step, delta_soh, action):
        """
        Compute reward using incremental SOH loss (delta_soh) and power loss (p_loss_step).
        """
        # Normalize the losses
        normalized_p_loss = p_loss_step / 1000.0  #  R * I^2 (degradation)
        normalized_soh_loss = abs(delta_soh) / 0.001  # Assuming max SOH loss per step around 0.1%
        
        revenue, revenue_without_battery = self._get_energy_cost(action) 

        cost = revenue - revenue_without_battery # Euro

        total_cycle = self.charge_throughput / (self.nominal_ah * 2) # cell 
        battery_wear_cost = self.params.battery.replacement_cost/(self.params.simulation.soh_max - self.params.simulation.soh_min) * np.abs(delta_soh)



        step_reward = cost - battery_wear_cost
        
        
        # step_reward = -(
        #     self.coeff_p_loss * normalized_p_loss +
        #     self.coeff_q_loss * normalized_soh_loss
        # )

        self.info["reward"].append(step_reward)
        self.info["cost"].append(cost)
        self.info["battery_wear_cost"].append(battery_wear_cost)
        self.info["total_cycle"].append(total_cycle + self.info["total_cycle"][-1])
        self.info["accumulated_reward"].append(sum(self.info["reward"]))
        self.info["accumulated_cost"].append(sum(self.info["cost"]))
        self.info["accumulated_battery_wear_cost"].append(sum(self.info["battery_wear_cost"]))


        # =================== CSV =========================
        self.info_csv["batt_wear_cost"].append(battery_wear_cost)
        self.info_csv["reward"].append(step_reward)
        self.info_csv["accumulated_reward"].append(sum(self.info["reward"]))
        self.info_csv["accumulated_cost"].append(sum(self.info["cost"]))
        self.info_csv["accumulated_battery_wear_cost"].append(sum(self.info["battery_wear_cost"]))
        # =================================================
        return step_reward


    def step(self, action):
        self.episode_length += 1

        # Add debug print to check episode length
        # if self.episode_length >= 9995:
        #     print(f"Episode length: {self.episode_length}")
        #     print(f"Data handler length: {len(self.data_handler)}")
        #     print(f"Dataset complete: {self.episode_length >= len(self.data_handler)}")

        # Action boundary check Debugging of action 
        action_bounded, delta_action = self._check_action(action)
        # Convert action_bounded from numpy array to float
        action_bounded = float(action_bounded[0]) if isinstance(action_bounded, np.ndarray) else float(action_bounded)
        self.journalist._process_smoothly(f"Action before optimize: {action_bounded:.3f} kW") if self.versbose else None
        
        # ===============================================
        # pack_power_w = action * 1000
        # cell_power_w = pack_power_w / (self.params.environment.battery.Ns * self.params.environment.battery.Np)
        # ===============================================

        # update SoC, get delta SoC, get current resistance
        resistance_pack, current_pack, _ = self.current_optimizer._optimize(self.soc, action_bounded) # action KW, soc %
        # ===============================================
        # R_pack_step = (self.params.environment.battery.Ns / self.params.environment.battery.Np) * resistance_cell
        # I_pack_step = self.params.environment.battery.Np * current_cell

        p_loss_step, delta_soh = self._coupled_battery_degradation(resistance_pack, current_pack)


        reward_step = self._get_reward(p_loss_step, delta_soh, action_bounded)

        self._get_soc(current_pack, resistance_pack)

        # =================== CSV =========================
        self.info_csv["P_batt"].append(action.item())
        self.info_csv["current"].append(current_pack)

        # =================================================

        if self.episode_length < len(self.data_handler):
            next_state = self.data_handler[self.episode_length]
            self.p_l = next_state["load"]
            self.p_g = next_state["solar"]
            self.time = next_state["datetime"]
            self.day_of_week = next_state["day_of_week"]
            self.hour = next_state["hour"]
            dataset_complete = False
            # =================== CSV ========================
            self.info_csv["P_l"].append(self.p_l)
            self.info_csv["P_g"].append(self.p_g)
            self.info_csv["SoC"].append(self.soc) 
            self.info_csv["SoH"].append(self.SOH)
            self.info_csv["Q_loss"].append(self.Q_loss)
            # =================================================
        
        else:
            dataset_complete = True

        obs = self._get_obs()
        wasted = (self.SOH <= self.params.simulation.soh_min)

        self.info["number_of_batteries"] =+ 1 if wasted else 0 
        # if  wasted or dataset_complete:
        #     done = True
        # else:
        #     done = False 

        # if (self.episode_length >= self.params.environment.max_steps_per_episode or 
        #     self.SOH <= self.params.environment.economic.soh_replacement_threshold or dataset_complete):
        if (self.SOH <= self.params.environment.economic.soh_replacement_threshold or dataset_complete):
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
        """
            Resistance: is at pack level
            Current: is at pack level
            soc: is in percent 
        """
        # Add basic sanity checks
        assert 0.01 <= resistance <= 1.0, f"Resistance {resistance}Ω out of range"
        assert -200 <= current <= 200, f"Current {current}A out of range"
        assert 0 <= self.soc <= 100, f"SOC {self.soc}% out of range"

        # Power loss (Ohmic)
        p_loss_step = resistance * current**2  # Watts

        # ===============================================
         # ----------------- Convert to *cell* quantities -----------------------
        Np          = self.params.battery.Np                   # 32
        I_cell      = current / Np                             # A
        Q_cell_nom  = self.nominal_ah                          # 1.69 Ah
        dt_h        = self.time_step                           # 0.25 h

        dAh_cell    = abs(I_cell) * dt_h                       # Ah
        self.charge_throughput += dAh_cell


        # C-rate
        c_rate = abs(I_cell) / Q_cell_nom
        # ===============================================

        # # Charge throughput increment (Ah)
        # delta_ah = abs(current) * self.time_step  # Ah
        # self.charge_throughput += delta_ah

        # # Calculate C-rate
        # c_rate = abs(current) / (self.nominal_ah * self.params.battery.Np)

        # Stress factor (using your existing sfunct)
        s_value = self._sfunct(c_rate, temperature=self.aging_model_config.constant_temperature)

        # Eq. 5 Ah_total
        ah_total = (50.0 / s_value) ** (1.0 / self.aging_model_config.exponent_z)
        # TODO
        # ah_total = (500.0 / s_value) ** (1.0 / self.aging_model_config.exponent_z)

        # C_use = self.SOH * self.nominal_ah
        # C_use_pack = self.params.battery.Np * self.SOH 
        # N_val = ah_total / C_use_pack if C_use_pack > 0 else 1e6  # Avoid zero-division
        
        # Remaining usable capacity (Ah) of one cell
        Q_use = self.SOH * Q_cell_nom
        N_val = ah_total / Q_use if Q_use > 0 else 1e6         # avoid /0

        # delta_soh = - ( |I|*dt ) / [2 * N_val * C_use ]
        # delta_soh = - (abs(current) * self.time_step) / (2 * N_val * C_use)
        delta_soh = - dAh_cell / (2 * N_val * Q_use)
        
        self.SOH  = np.clip(self.SOH + delta_soh, 0.0, 1.0)

        # ----------------- Update SOC (percent) -------------------------------
        dSOC = - dAh_cell / Q_cell_nom * 100* np.sign(I_cell)   # discharge ↓
        # Calculate new SOC before clipping
        new_soc = self.soc + dSOC
        
        # Apply clipping and store result
        self.soc = float(np.clip(new_soc,
                                self.observation_space.low[0],
                                self.observation_space.high[0]))
        
        # Log if clipping occurred
        if new_soc != self.soc:
            if self.versbose:
                self.journalist._process_smoothly(f"SOC clipped from {new_soc:.2f}% to {self.soc:.2f}%")  
        else:
            self.info["soc_violation"].append(False)

        # dSOC = - (current * self.time_step) / (C_use if C_use>0 else 1e-6) * 100.0
        # self.soc += dSOC
        # self.soc = np.clip(self.soc, self.observation_space.low[0], self.observation_space.high[0])

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
        if self.episode_length % 100 == 0 and self.versbose:
            self.journalist._process_smoothly(
                f"Step {self.episode_length} => "
                f"SOH: {self.SOH*100:.2f}%, "
                f"SOC: {self.soc:.2f}%, "
                f"Q_loss: {self.Q_loss_percent:.2f}%, "
                f"C_use: {Q_use:.2f}Ah, "
                f"N: {N_val:.3f}, "
                f"Ah_total: {ah_total:.2f}, "
                f"delta_soh: {delta_soh:.5f}"
            )

        self.journalist._process_smoothly(
        f"Icell={I_cell:.3f} A  C-rate={c_rate:.2f}  s={s_value:.1e}  "
        f"Ah_total={ah_total:.1f}  dAh={dAh_cell:.3f}  ΔSoH={delta_soh:.5f}"
            ) if self.versbose else None
        

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
        # self.info["soc_violation"].append(soc_boundary_violation)
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
        stress = (a * self.soc  + b) * np.exp(
            -(self.aging_model_config.Ea_J_per_mol + self.aging_model_config.h * c_rate) / 
            (self.aging_model_config.Rg_J_per_molK * (273.15 + temperature))
        )
        
        return stress
# =============================== End of degradation model ===============================

    def _get_energy_cost(self, action):

        E_batt_in = abs(action) * self.time_step if action < 0 else 0 # kwh  charge
        E_batt_out = abs(action) * self.time_step if action > 0 else 0  # kwh discharge

        price_buy, price_sell = self._get_price()

        net_load = (self.p_l - self.p_g) * self.time_step   # (kWh)> 0 => load is higher than generation, < 0 => generation is higher than load

        if net_load > 0:   # load is higher than generation
            cost_without_battery = - net_load * price_buy
            if action > 0:  # battery is discharging
                if E_batt_out >= net_load:
                    E_grid_import, E_grid_export = 0, E_batt_out - net_load
                else: 
                    E_grid_import, E_grid_export = net_load - E_batt_out, 0 

            else:  # battery is charging
                E_grid_import, E_grid_export = net_load + E_batt_in, 0
        
        else:  # Surplus energy
            cost_without_battery = - net_load * price_sell
            if action < 0:  # battery is charging 
                if E_batt_in >= abs(net_load):
                    E_grid_import, E_grid_export = E_batt_in - abs(net_load), 0
                else:
                    E_grid_import, E_grid_export = 0, abs(net_load) - E_batt_in
            else:  # battery is discharging
                E_grid_import, E_grid_export = 0, abs(net_load) + E_batt_out

        
        cost = -E_grid_import * price_buy + E_grid_export * price_sell

        self.info["profit_cost"].append(cost)
        self.info["profit_cost_without_battery"].append(cost_without_battery)


        # =================== CSV =========================
        self.info_csv["E_sell"].append(E_grid_export)
        self.info_csv["E_buy"].append(E_grid_import)
        self.info_csv["Cost_with_batt"].append(cost)
        self.info_csv["cost_without_batt"].append(cost_without_battery)
        # =================================================
        

        return cost, cost_without_battery
    

# =============================== Time ====================================
    def _get_cyclic_hour(self, time):
        hour_sin, hour_cos = np.sin(2 * np.pi * time/24), np.cos(2 * np.pi * time/24)
        return hour_sin, hour_cos
    
    
    def _get_cyclic_day_of_week(self, day_of_week):
        day_sin, day_cos = np.sin(2 * np.pi * day_of_week/7), np.cos(2 * np.pi * day_of_week/7)
        return day_sin, day_cos
    

    def _get_price(self):
        if self.time is None:
            self.journalist._get_error("self.time is None when calling _get_price()")
            # Use default prices as fallback
            return self.params.environment.economic.multiple_price_buy.mid, self.params.environment.economic.multiple_price_sell.mid
        
        if self.time.weekday() < 5:
            if 8 <= self.time.hour < 20:
                price_buy = self.params.environment.economic.multiple_price_buy.max
                price_sell = self.params.environment.economic.multiple_price_sell.max
            elif 7<= self.time.hour < 8 or 20<= self.time.hour < 23:
                price_buy = self.params.environment.economic.multiple_price_buy.mid
                price_sell = self.params.environment.economic.multiple_price_sell.mid
            else: 
                price_buy = self.params.environment.economic.multiple_price_buy.min
                price_sell = self.params.environment.economic.multiple_price_sell.min
        else:  # for days >= 5
            price_buy = self.params.environment.economic.multiple_price_buy.mid
            price_sell = self.params.environment.economic.multiple_price_sell.mid

        return price_buy, price_sell
    

# =============================== Debugging ===============================

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

        print(f"|{'-' * (box_width - 2)}|")
        print(format_line(f"Accumulated Reward: {self.info['accumulated_reward'][-1]}"))
        print(format_line(f"Accumulated Cost: {self.info['accumulated_cost'][-1]}"))
        print(format_line(f"Accumulated Battery Wear Cost: {self.info['accumulated_battery_wear_cost'][-1]}"))

        print(separator)



    def create_bar(self, percentage: float, width: int = 40) -> str:
            # Convert percentage to decimal (100% -> 1.0)
            decimal = percentage / 100.0
            # Ensure the value is between 0 and 1
            decimal = max(0, min(1, decimal))
            filled = int(decimal * width)
            return f"[{'█' * filled}{'-' * (width - filled)}] {percentage:.1f}%"

    def save_to_csv(self, filepath=None, episode_id=None):
        """
        Save step information to a CSV file.
        
        Args:
            filepath: Path to save the CSV file. If None, uses './results/csv/episode_{episode_id}.csv'
            episode_id: Optional episode identifier for the filename. If None, uses episode_length.
        """
        # Create default path if not provided
        if filepath is None:
            # Create directories if they don't exist
            os.makedirs("./results/csv", exist_ok=True)
            # Use episode_id if provided, otherwise use episode_length
            episode_num = episode_id if episode_id is not None else self.episode_length
            filepath = f"./results/csv/episode_{episode_num}.csv"
        
        # Convert info dictionary to DataFrame
        # Only include lists that have data (excluding single values)
        data_dict = {k: v for k, v in self.info.items() if isinstance(v, list) and len(v) > 0}
        
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        
        # Add episode metadata
        if len(df) > 0:
            df['episode_id'] = episode_id if episode_id is not None else self.episode_length
            
            # Save to CSV
            df.to_csv(filepath, index=False)
            self.journalist._process_smoothly(f"Episode data saved to {filepath}")
            
        return filepath
        
    def append_to_csv(self, filepath="/home/danial/Documents/Codes_new/N-IJCCI-BMS/Code/results/csvs"):
        """
        Append the current step's data to a CSV file.
        
        Args:
            filepath: Path to save the CSV file. If None, uses './results/csv/step_data.csv'
        
        Returns:
            filepath of the CSV file
        """
        # Create default path if not provided
        if filepath is None:
            # Create directories if they don't exist
            os.makedirs("./results/csv", exist_ok=True)
            filepath = "./results/csv/step_data.csv"
        
        # Get the latest data for each metric
        data = {}
        for key, values in self.info_csv.items():
            if isinstance(values, list) and len(values) > 0:
                data[key] = [values[-1]]  # Get only the latest value
            elif not isinstance(values, list):
                data[key] = [values]  # Include non-list values
        
        # Add step and episode information
        data['step'] = [self.episode_length]
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Check if file exists
        file_exists = os.path.isfile(filepath)
        
        # Write to CSV (append mode if file exists)
        df.to_csv(filepath, mode='a', header=not file_exists, index=False)
        
        # Comment out or remove this line to stop printing messages
        # self.journalist._process_smoothly(f"Step {self.episode_length} data appended to {filepath}")
        
        return filepath











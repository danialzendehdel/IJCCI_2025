
import numpy as np

class DegradationModel:
    def __init__(self, params):
        self.params = params

    def _coupled_battery_degradation(self, resistance, current, charge_throughput):
        # Add basic sanity checks
        assert 0.01 <= resistance <= 1.0, f"Resistance {resistance}Ω out of range"
        assert -200 <= current <= 200, f"Current {current}A out of range"
        assert 0 <= self.soc <= 100, f"SOC {self.soc}% out of range"

        # Power loss (Ohmic)
        p_loss_step = resistance * current**2  # Watts

        # Charge throughput increment (Ah)
        delta_ah = abs(current) * self.time_step  # Ah
        charge_throughput += delta_ah

        # Calculate C-rate
        c_rate = abs(current) / self.params.simulation.time_interval

        # Stress factor (using your existing sfunct)
        s_value = self._sfunct(c_rate, temperature=self.params.aging_model_config.constant_temperature)

        # Eq. 5 Ah_total
        ah_total = (20.0 / s_value) ** (1.0 / self.params.aging_model_config.exponent_z)
        # TODO
        # ah_total = (500.0 / s_value) ** (1.0 / self.aging_model_config.exponent_z)

        C_use = self.SOH * self.params.battery.nominal_Ah
        N_val = ah_total / C_use if C_use > 0 else 1e6  # Avoid zero-division

        # delta_soh = - ( |I|*dt ) / [2 * N_val * C_use ]
        delta_soh = - (abs(current) * self.params.simulation.time_interval) / (2 * N_val * C_use)
        self.SOH += delta_soh
        self.SOH = max(0.0, min(self.SOH, 1.0))

        dSOC = - (current * self.params.simulation.time_interval) / (C_use if C_use>0 else 1e-6) * 100.0
        self.soc += dSOC
        self.soc = np.clip(self.soc, self.params.simulation.soc_min, self.params.simulation.soc_max)

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

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import fsolve
import pygad
import os

# TODO C_rate or discharge rate 
# TODO: fix c_rate, start from 0 - i_max


class OptimizeSoC:
    def __init__(self, soc_lp, c_rate_lp, r_lp, ocv_lp, epsilon, nominal_current, nominal_capacity, params, journalist_ins, GA=False, verbose=False):

        # Lookup tables:
        self.soc_lp = soc_lp
        self.c_rate_lp = c_rate_lp
        self.r_lp = r_lp
        self.ocv_lp = ocv_lp
        
        self.interpolator = RegularGridInterpolator((self.soc_lp, self.c_rate_lp), self.r_lp)

        self.R = None
        self.nominal_current = nominal_current
        self.nominal_capacity = nominal_capacity
        self.epsilon = epsilon
        self.GA =  GA
        self.verbose = verbose
        self.params = params # Configuration parameters
        self.journalist = journalist_ins
        self.info = {
            "c_rate": [],
            "R": [],
            "current": [],
            "error_ga": [],
            'R_pack': [],
            'I_pack': []
        }


    def interpolate(self, soc, c_rate):

        if isinstance(soc, np.ndarray):
            soc = float(soc)
        if isinstance(c_rate, np.ndarray):
            c_rate = float(c_rate)

        point = np.array([[soc, c_rate]])
        try:
            r_interp = self.interpolator(point)
        except ValueError:
            print(f"Warning!: Interpolation failed for SoC: {soc}, c_rate: {c_rate}, ")
            return None
        return r_interp[0]
    

    def solve_current(self, R, OCV, P):
        """
        Solve for battery current using corrected power equation:
        - When P > 0 (deficit), battery should discharge (I < 0)
        - When P < 0 (surplus), battery should charge (I > 0)
        
        Battery perspective: P_batt = OCV * I - R * I²
        Grid perspective: P_net = P_load - P_gen = -P_batt
        
        P_net = P_load - P_gen  > 0 (deficit), battery should discharge, P_battx
        P_net = P_load - P_gen  < 0 (surplus), battery should charge, P_batt = +P_net




        For P_net > 0 (deficit): I should be negative (discharge)
        For P_net < 0 (surplus): I should be positive (charge)
        
        Rearranging: R * I² - OCV * I - P = 0
        """

        def equation(I, R, OCV, P): ## ⚠️⚠️⚠️⚠️ (sign)P ⚠️⚠️⚠️⚠️
            return R * I**2 - OCV * I - P
        
        # Better initial guess
        # I_initial_guess = P / OCV if abs(P / OCV) < self.nominal_current else np.sign(P) * self.nominal_current
        if P > 0:  # Deficit condition
            I_initial_guess = -P / OCV  # Negative current (discharge)
        else:  # Surplus condition
            I_initial_guess = -P / OCV  # Positive current (charge)

        I_solution_ = fsolve(equation, I_initial_guess, args=(R, OCV, P))
        I_solution = I_solution_[0]
        

        # ================= Debugging =================
        # Check both roots
        a, b, c = R, -OCV, P
        disc = b**2 - 4 * a * c
        if disc >= 0:
            I1 = (-b + np.sqrt(disc)) / (2 * a)
            I2 = (-b - np.sqrt(disc)) / (2 * a)
            # Pick root closest to expected power direction
            I_expected = P / OCV
            I_solution = I1 if abs(I1 - I_expected) < abs(I2 - I_expected) else I2
        
        power_calc = OCV * I_solution - R * I_solution**2
        if self.verbose:
            self.journalist._process_smoothly(f"Solve_current: P={P:.1f} W, I={I_solution:.3f} A, Calc={power_calc:.1f} W")
        # ================= Debugging =================
        # Double-check solution makes sense for power sign
        result = R * I_solution_[0]**2 - OCV * I_solution_[0] - P
        if abs(result) > 0.1:  # If solution is far from zero
            print(f"Warning: Current solution may be inaccurate. Error: {result}")
            
        # Check if direction makes sense
        if P > 0 and I_solution_[0] > 0:
            print(f"Warning: Power deficit ({P:.3f}) should result in discharge, but current is {I_solution_[0]:.3f}")
        elif P < 0 and I_solution_[0] < 0:
            print(f"Warning: Power surplus ({P:.3f}) should result in charge, but current is {I_solution_[0]:.3f}")
            
        return I_solution

    def find_R_init(self, soc, power):

       
        # c_rate_init = power / self.nominal_capacity # 0.1 ....
        c_rate_init = power / (self.nominal_current )
        ind_c_rate = np.abs(self.c_rate_lp - c_rate_init).argmin()
        c_rate_init = self.c_rate_lp[ind_c_rate]

        r_init = self.interpolate(soc, c_rate_init)
        self.info["R"].append(r_init)
        return r_init
    

    def _optimize(self, soc, power):

        if self.verbose:
            self.journalist._process_smoothly(f"Optimizer received power: {power:.3f} kW")
        ## SCALING ####
        power_cell = power * 1000 / (self.params.battery.Ns * self.params.battery.Np)
        ocv = interp1d(self.soc_lp, self.ocv_lp, kind='linear', fill_value="extrapolate")(soc)
        
        if not self.GA:
            self.R = self.find_R_init(soc, power_cell)
            iteration = 0
            I_cell = None
            while True:
                iteration += 1
                I_cell = self.solve_current(self.R, ocv, power_cell)
                cell_nominal_current = self.nominal_capacity / self.params.battery.Np # 4.9 A
                c_rate = abs(I_cell) / cell_nominal_current
                ind_c_rate = np.abs(self.c_rate_lp - c_rate).argmin()
                c_rate = self.c_rate_lp[ind_c_rate]
                self.info["c_rate"].append(c_rate)
                R_cell_new = self.interpolate(soc, c_rate)
                self.info["current"].append(float(I_cell))
                self.info["R"].append(R_cell_new)
                
                if iteration > 100 :
                    self.journalist._get_warning(f"Iteration #{iteration}, SOC: {soc}, Delta_R: {self.R - R_cell_new}")
                if abs(self.R - R_cell_new) < self.epsilon or iteration > 1000:
                    break
                self.R = R_cell_new
            
            # Cell-level power check
            power_cell_calc = ocv * I_cell - self.R * I_cell**2
            if abs(power_cell_calc - power_cell) > 0.1:
                self.journalist._get_warning(f"Cell power mismatch: Expected {power_cell:.1f} W, Got {power_cell_calc:.1f} W")
        
        else:
            R_cell_new, I_cell = self.optimize_GA(ocv, power_cell)
            self.info["current"].append(float(I_cell))
            self.info["R"].append(R_cell_new)
        
        R_pack = (self.params.battery.Ns / self.params.battery.Np) * R_cell_new
        I_pack = self.params.battery.Np * I_cell
        
        # Pack-level power for logging
        pack_voltage = ocv * self.params.battery.Ns
        power_pack_calc = I_pack * pack_voltage - R_pack * I_pack**2
        power_pack_expected = power * 1000
        
        self.info["R_pack"].append(R_pack)
        self.info["I_pack"].append(I_pack)
        
        
        
        if self.verbose:
            self._status()
            if self.verbose:
                self.journalist._process_smoothly(
                f"R_pack: {R_pack:.6f}, R_cell: {R_cell_new:.6f}, "
                f"I_pack: {I_pack:.3f}, I_cell: {I_cell:.3f}, "
            f"Power_expected: {power_pack_expected:.1f} W, Power_calc: {power_pack_calc:.1f} W"
        )
        
        return R_pack, I_pack, self.info
        
    
    def objective_func(self, ga_instance, solution, solution_idx, ocv, power):
        R, I = solution
        OCV = ocv
        P = power
        # P = OCV*I - RI² (Power equation from battery perspective)
        error = abs(R * pow(I,2) - OCV * I + P)
        self.info["error_ga"].append(error)
        return -error

    def optimize_GA(self, ocv, power):
        num_generations = 100
        num_parents_mating = 6
        sol_per_pop = 20
        num_genes = 2  # [R, I]

        # Update gene space to enforce current direction based on power
        gene_space = [
            {'low': 0.025, 'high': 0.282},  # R range
            {
                'low': -self.nominal_current if power < 0 else 0,  # Current range based on power
                'high': 0 if power < 0 else self.nominal_current
            }
        ]

        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=lambda ga, solution, idx: self.objective_func(ga, solution, idx, ocv, power),
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            gene_space=gene_space,  # Updated gene space
            crossover_type="two_points",
            mutation_type="random",
            mutation_percent_genes=10,
            mutation_num_genes=2,
            parallel_processing=["thread", max(4, os.cpu_count())],
            stop_criteria=["saturate_5"]
        )

        ga_instance.run()
        best_solution, best_fitness, solution_idx = ga_instance.best_solution()
        
        # Add debug print
        if self.verbose:
            print(f"GA Solution - Power: {power}, Current: {best_solution[1]}, R: {best_solution[0]}")
        
        return best_solution[0], best_solution[1]  # R, I


    def _status(self):

        box_width = 70
        separator = "=" * box_width
        def format_line(content:str) -> str:
            return f"|{content.ljust(box_width - 4)}|"

        print(separator)
        if self.GA:
            print(format_line(f"Approach: GA"))
        else:
            print(format_line(f"Approach: Deterministic"))

        # print(format_line(f"power: {self.power}"))
        # print(format_line(f"SoC: {self.soc}"))
        # print(format_line(f"OCV: {self.OCV}"))
        print(format_line(f"new_R: {self.info['R'][-1]}"))
        print(format_line(f"new_current: {self.info['current'][-1]}"))
        print(format_line(f"new_c_rate: {self.info['c_rate'][-1]}"))

        print(separator)
        print(separator)








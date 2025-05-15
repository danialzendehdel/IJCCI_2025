import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import fsolve
import pygad
import os
# TODO C_rate or discharge rate 
class OptimizeSoC:

    def __init__(self, soc, soc_lp, c_rate_lp, r_lp, ocv, power, epsilon, nominal_current, nominal_capacity, GA=False, verbose=False):

        self.soc_lp = soc_lp
        self.c_rate_lp = c_rate_lp
        self.r_lp = r_lp
        self.power = power
        self.epsilon = epsilon

        self.GA =  GA

        # self.soc = soc
        self.OCV = ocv

        self.interpolator = RegularGridInterpolator((self.soc_lp, self.c_rate_lp), self.r_lp)


        self.nominal_current = nominal_current
        self.nominal_capacity = nominal_capacity


        # self.r_init = 0.13
        c_rate_init = power / self.nominal_capacity # 0.1 ....
        ind_c_rate = np.abs(self.c_rate_lp - c_rate_init).argmin()
        self.c_rate_init = self.c_rate_lp[ind_c_rate]

        self.verbose = verbose

        # TODO: fix c_rate, start from 0 - i_max
        self.info = {
            "c_rate": [],
            "R": [],
            "current": [],
            "error_ga": []
        }

        self.R = self.find_R_init(soc)


    def interpolate(self, soc, c_rate):

        if isinstance(soc, np.ndarray):
            soc = float(soc)
        if isinstance(c_rate, np.ndarray):
            c_rate = float(c_rate)

        point = np.array([[soc, c_rate]])
        try:
            r_interp = self.interpolator(point)
        except ValueError:
            print(f"Warning!: Interpolation failed for SoC: {soc}, c_rate: {c_rate}, power: {self.power}")
            return None
        return r_interp[0]

    def solve_current(self, R, OCV, P):
        """
        Solve for battery current using corrected power equation:
        - When P > 0 (deficit), battery should discharge (I < 0)
        - When P < 0 (surplus), battery should charge (I > 0)
        
        Battery perspective: P_batt = OCV * I - R * I²
        Grid perspective: P_net = P_load - P_gen = -P_batt
        
        For P_net > 0 (deficit): I should be negative (discharge)
        For P_net < 0 (surplus): I should be positive (charge)
        
        Rearranging: R * I² - OCV * I - P = 0
        """
        def equation(I, R, OCV, P):
            # Crucial fix: Battery provides power when P > 0 (deficit)
            # So from battery's perspective, P_batt = -P_net
            return R * I ** 2 - OCV * I - P

        # Better initial guess based on power sign
        # For deficit (P > 0), battery should discharge (I < 0)
        if P > 0:  # Deficit condition
            I_initial_guess = -P / OCV  # Negative current (discharge)
        else:  # Surplus condition
            I_initial_guess = -P / OCV  # Positive current (charge)
            
        I_solution = fsolve(equation, I_initial_guess, args=(R, OCV, P))
        
        # Double-check solution makes sense for power sign
        result = R * I_solution[0]**2 - OCV * I_solution[0] - P
        if abs(result) > 0.1:  # If solution is far from zero
            print(f"Warning: Current solution may be inaccurate. Error: {result}")
            
        # Check if direction makes sense
        if P > 0 and I_solution[0] > 0:
            print(f"Warning: Power deficit ({P:.3f}) should result in discharge, but current is {I_solution[0]:.3f}")
        elif P < 0 and I_solution[0] < 0:
            print(f"Warning: Power surplus ({P:.3f}) should result in charge, but current is {I_solution[0]:.3f}")
            
        return I_solution[0]

    def find_R_init(self, soc):
        r_init = self.interpolate(soc, self.c_rate_init)
        self.info["R"].append(r_init)
        return r_init

    def _optimize(self, soc, ocv):
        if not self.GA:
            iteration = 0
            while True:
                iteration += 1

                I = self.solve_current(self.R, self.OCV, self.power)

                # find c_rate  index
                c_rate = I / self.nominal_current
                ind_c_rate = np.abs(self.c_rate_lp - c_rate).argmin()
                c_rate = self.c_rate_lp[ind_c_rate]
                self.info["c_rate"].append(c_rate)


                R_new = self.interpolate(soc, c_rate)
                self.info["current"].append(np.float64(I).item())
                self.info["R"].append(R_new)

                if iteration > 100:
                    print(f"WARNING! Iteration #{iteration}, {self.soc}, Delta_R:{self.R - R_new}")


                if abs(self.R - R_new) < self.epsilon or iteration > 1000:
                    # print(f"converged at R: {R_new}, current: {I}, OCV: {self.OCV}")
                    # print(f"# iteration: {iteration}")
                    break

                self.R = R_new


        else:
            R_new, I = self.optimize_GA(ocv)
            self.info["current"].append(np.float64(I).item())
            self.info["R"].append(R_new)

        if self.verbose:
            self._status()


        return R_new,I,  self.info

    def objective_func(self, ga_instance, solution, solution_idx, ocv):
        R, I = solution
        OCV = ocv
        P = self.power
        # FIXED: Correct equation for GA optimization
        error = abs(R * pow(I,2) - OCV * I - P)
        self.info["error_ga"].append(error)
        return -error

    def optimize_GA(self, ocv):
        num_generations = 100  # More generations for better convergence
        num_parents_mating = 6  # More parents to ensure diversity
        sol_per_pop = 20  #  population size
        num_genes = 2  # [R, I]
        num_parents_mating = 6 

        # FIXED: Initialize gene space based on power direction
        # For discharge (P > 0), current should be negative
        if self.power > 0:  # Discharge
            i_range = {'low': -self.nominal_current, 'high': 0}
        else:  # Charge
            i_range = {'low': 0, 'high': self.nominal_current}
            
        gene_space = [
            {'low': 0.025, 'high': 0.282},  # Range for R
            i_range  # Current range based on power direction
        ]
        
        num_workers = max(4, os.cpu_count())  # Use max 4 or available cores

        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=lambda ga, solution, idx: self.objective_func(ga, solution, idx, ocv),
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            gene_space=gene_space,
            crossover_type="two_points",
            mutation_type="random",
            mutation_percent_genes=10,
            mutation_num_genes=2,
            parallel_processing=["thread", num_workers],
            stop_criteria=["saturate_5"]
        )

        ga_instance.run()

        best_solution, best_fitness, solution_idx = ga_instance.best_solution()
        
        # Validate solution matches power sign expectation
        I = best_solution[1]
        if (self.power > 0 and I > 0) or (self.power < 0 and I < 0):
            print(f"WARNING: Current direction ({I:.4f}) doesn't match expected direction for power {self.power:.4f}")
            
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

        print(format_line(f"power: {self.power}"))
        # print(format_line(f"SoC: {self.soc}"))
        # print(format_line(f"OCV: {self.OCV}"))
        print(format_line(f"new_R: {self.info['R'][-1]}"))
        print(format_line(f"new_current: {self.info['current'][-1]}"))
        
        # FIXED: Show more details about power and current
        power_type = "DEFICIT (discharge)" if self.power > 0 else "SURPLUS (charge)"
        current_type = "DISCHARGE" if self.info['current'][-1] < 0 else "CHARGE"
        print(format_line(f"power type: {power_type}"))
        print(format_line(f"current type: {current_type}"))

        print(separator)
        print(separator) 
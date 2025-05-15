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

        def equation(I, R, OCV, P):
            return R * I ** 2 - OCV * I + P

        I_initial_guess = P / OCV  # Initial guess for fsolve
        I_solution = fsolve(equation, I_initial_guess, args=(R, OCV, P))
        # print(f"I = {I_solution}")
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
        error = abs(R * pow(I,2) - OCV * I + P)
        self.info["error_ga"].append(error)
        return -error

    def optimize_GA(self, ocv):
        num_generations = 100  # More generations for better convergence
        num_parents_mating = 6  # More parents to ensure diversity
        sol_per_pop = 20  #  population size
        num_genes = 2  # [R, I]
        num_parents_mating = 6 

        gene_space = [
            {'low': 0.025, 'high': 0.282},  # Wider range for R
            {'low': -self.nominal_current, 'high': self.nominal_current}  # I range
        ]
        num_workers = max(4, os.cpu_count())  # Use max 4 or available cores
        # print(num_workers)

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
            parallel_processing=["thread", num_workers],  # Enable parallel fitness evaluation
            stop_criteria=["saturate_5"]
        )

        ga_instance.run()

        # try:
        #     ga_instance.run()
        #     solution, solution_fitness, _ = ga_instance.best_solution()
        #     if solution is None:
        #         raise ValueError("No valid solution found")
        #     return solution, solution_fitness
        # except Exception as e:
        #     print(f"GA optimization failed: {str(e)}")
        #     # Return default/fallback values or raise the exception
        #     raise

        best_solution, best_fitness, solution_idx = ga_instance.best_solution()
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

        print(separator)
        print(separator)
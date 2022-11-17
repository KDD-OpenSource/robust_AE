import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch
import time
from maraboupy import Marabou
from maraboupy import MarabouCore

from .evaluation import evaluation
from .evaluation import extract_marabou_solution_point
from .evaluation import extract_marabou_solution_stats
from .evaluation import add_marabou_solution_stats
from .evaluation import update_to_tot_key
from .evaluation import test_for_collapsing

# eps is the area we are searching in, delta is the largest error value
class marabou_largest_error(evaluation):
    def __init__(
        self,
        name: str = "marabou_largest_error",
        # eps denotes the area around the input in which we search for the
        # largest error
        eps=0.1,
    ):
        self.name = name
        self.eps = eps
        self.surrounding_margin = 0.1

    def evaluate(self, dataset, algorithm, run_inst):
        collapsing = test_for_collapsing(dataset, algorithm)
        network = self.get_marabou_network(algorithm, dataset, run_inst)
        label_means = dataset.calc_label_means(subset="test")
        result_dict = {}
        for key in label_means.keys():
            input_sample = pd.DataFrame(label_means[key]).transpose()
            output_sample = algorithm.predict(input_sample)
            solution, tot_time, delta, tot_solution_stats = self.binary_search_delta(
                network=network,
                input_sample=input_sample,
                output_sample=output_sample,
                accuracy=0.0001,
            )

            self.plot_and_save(
                run_inst,
                result_dict,
                key,
                solution,
                network,
                input_sample,
                output_sample,
                delta,
                tot_time,
                tot_solution_stats,
                collapsing,
            )
            self.plot_and_save_surrounding_fcts(
                run_inst, result_dict, input_sample, algorithm, key
            )
        self.save_json(run_inst, result_dict, "results_marabou_largest")

    def binary_search_delta(self, network, input_sample, output_sample, accuracy):
        marabou_options = Marabou.createOptions(timeoutInSeconds=300)
        eps = self.eps
        numInputVars = len(network.inputVars[0][0])
        for ind in range(numInputVars):
            network.setLowerBound(
                network.inputVars[0][0][ind], input_sample.values[0][ind] - eps
            )
            network.setUpperBound(
                network.inputVars[0][0][ind], input_sample.values[0][ind] + eps
            )

        found_largest_delta = False
        delta = 1
        delta_change = delta / 2
        start_time = time.time()
        numOutputVars = len(network.outputVars[0])
        solution = None
        tot_solution_stats = 0

        # binary search over values of delta
        while not found_largest_delta:
            print(delta)
            disj_eqs = []
            for ind in range(numOutputVars):
                outputVar = network.outputVars[0][ind]
                inputVar = network.inputVars[0][0][ind]
                eq1 = MarabouCore.Equation(MarabouCore.Equation.GE)
                eq1.addAddend(-1, outputVar)
                eq1.addAddend(1, inputVar)
                eq1.setScalar(delta)

                eq2 = MarabouCore.Equation(MarabouCore.Equation.GE)
                eq2.addAddend(1, outputVar)
                eq2.addAddend(-1, inputVar)
                eq2.setScalar(delta)

                disj_eqs.append([eq1])
                disj_eqs.append([eq2])

            network.disjunctionList = []
            network.addDisjunctionConstraint(disj_eqs)
            network_solution = network.solve(options=marabou_options)
            solution_stats = extract_marabou_solution_stats(network_solution)
            tot_solution_stats = add_marabou_solution_stats(
                tot_solution_stats, solution_stats
            )
            if network_solution[1].hasTimedOut():
                solution = None
                break
            if len(network_solution[0]) > 0:
                extr_solution = extract_marabou_solution_point(
                    network_solution, network
                )
                diff_input = abs(
                    np.array(extr_solution[0]) - input_sample.values[0]
                ).max()
                larg_diff = abs(
                    np.array(extr_solution[1]) - np.array(extr_solution[0])
                ).max()
                solution = network_solution

                if (diff_input < eps + accuracy) and larg_diff > delta - accuracy:
                    delta = delta + delta_change
                else:
                    delte = delta - delta_change

            else:
                delta = delta - delta_change

            if delta_change <= accuracy:
                found_largest_delta = True
            delta_change = delta_change / 2

        end_time = time.time()
        tot_time = end_time - start_time
        return solution, tot_time, delta, tot_solution_stats

    def plot_and_save(
        self,
        run_inst,
        result_dict,
        key,
        solution,
        network,
        input_sample,
        output_sample,
        delta,
        tot_time,
        tot_solution_stats,
        collapsing,
    ):
        if solution is not None:

            # extract and calculate information for plots
            solution_stat_dict = extract_marabou_solution_stats(solution)
            extr_solution = extract_marabou_solution_point(solution, network)
            diff_input = abs(np.array(extr_solution[0]) - input_sample.values[0]).max()
            larg_diff = abs(
                np.array(extr_solution[1]) - np.array(extr_solution[0])
            ).max()

            # save figures
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=[20, 20])
            fig.suptitle(f"Calculation took {tot_time}")
            ax[0].plot(input_sample.values[0], label="input_sample")
            ax[0].plot(extr_solution[0], label="input_solution")
            ax[0].set_title(
                f"""L_infty dist is at most {self.eps}. (real:
                    {diff_input})"""
            )
            ax[0].legend()
            ax[0].set_ylim(-1.25, 1.25)
            ax[1].plot(extr_solution[0], label="input_solution")
            ax[1].plot(extr_solution[1], label="output_solution")
            ax[1].set_title(
                f"""L_infty dist is at least {delta} (up to
                accuracy) real: {larg_diff}"""
            )
            ax[1].legend()
            ax[1].set_ylim(-1.25, 1.25)
            self.save_figure(
                run_inst, fig, f"marabou_largest_error_close_to_sample_{key}"
            )
            plt.close("all")

            # save all samples
            self.save_csv(
                run_inst,
                input_sample,
                "input_sample",
                subfolder=f"results_largest_error_{key}",
            )
            self.save_csv(
                run_inst,
                output_sample,
                "output_sample",
                subfolder=f"results_largest_error_{key}",
            )
            self.save_csv(
                run_inst,
                pd.DataFrame(extr_solution[0]),
                "input_solution",
                subfolder=f"results_largest_error_{key}",
            )
            self.save_csv(
                run_inst,
                pd.DataFrame(extr_solution[1]),
                "output_solution",
                subfolder=f"results_largest_error_{key}",
            )

            # save statistics
            result_dict["label_" + str(key)] = {
                "calc_time": tot_time,
                "dist_to_y": self.eps,
                "error": delta,
                "real_dist_to_y": diff_input,
                "real_error": larg_diff,
            }
            result_dict["label_" + str(key)].update(solution_stat_dict)
            tot_solution_stats = update_to_tot_key(tot_solution_stats)
            result_dict["label_" + str(key)].update(tot_solution_stats)
            result_dict["collapsing"] = collapsing
        else:
            # save statistics (in case there is no solution)
            result_dict["label_" + str(key)] = {
                "calc_time": tot_time,
                "dist_to_y": self.eps,
                "error": None,
                "real_dist_to_y": None,
                "real_error": None,
            }
            tot_solution_stats = update_to_tot_key(tot_solution_stats)
            result_dict["label_" + str(key)].update(tot_solution_stats)
            result_dict["collapsing"] = collapsing

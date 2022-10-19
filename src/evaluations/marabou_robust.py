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


class marabou_robust(evaluation):
    def __init__(
        self,
        name: str = "marabou_robust",
        # delta denotes the margin by which the outputs have to differ
        delta=0.1,
    ):
        self.name = name
        self.desired_delta = delta

    def evaluate(self, dataset, algorithm, run_inst):
        collapsing = test_for_collapsing(dataset, algorithm)
        network = elf.get_marabou_network(algorithm, dataset)
        label_means = dataset.calc_label_means(subset="test")
        result_dict = {}

        for key in label_means.keys():
            input_sample = pd.DataFrame(label_means[key]).transpose()
            output_sample = algorithm.predict(input_sample)
            (
                    solution,
                    tot_time,
                    eps,
                    tot_solution_stats
            ) = self.binary_search_eps(eps=2, delta = self.desired_delta,
                    accuracy = 0.0000005, network = network, input_sample =
                    input_sample, output_sample = output_sample)

            self.plot_and_save(
                result_dict,
                key,
                solution,
                network,
                input_sample,
                output_sample,
                eps,
                tot_time,
                tot_solution_stats,
                collapsing,
            )

            self.plot_and_save_surrounding_fcts(
                result_dict, input_sample, algorithm, key
            )

        self.save_json(run_inst, result_dict, "results_marabou_robust")

    def binary_search_eps(
        self, eps, delta, accuracy, network, input_sample, output_sample
    ):
        start_time = time.time()

        numOutputVars = len(network.outputVars[0])
        disj_eqs = []
        for ind in range(numOutputVars):
            outputVar = network.outputVars[0][ind]
            eq1 = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq1.addAddend(-1, outputVar)
            eq1.setScalar(delta - output_sample.values[0][ind])

            eq2 = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq2.addAddend(1, outputVar)
            eq2.setScalar(delta + output_sample.values[0][ind])

            disj_eqs.append([eq1, eq2])

        network.disjunctionList = []
        network.addDisjunctionConstraint(disj_eqs)

        solution = None
        marabou_options = Marabou.createOptions(timeoutInSeconds=300)
        found_closest_eps = False
        numInputVars = len(network.inputVars[0][0])
        eps_change = eps / 2
        summed_solution_stats = 0
        while not found_closest_eps:
            # add eps constraints
            print(eps)
            for ind in range(numInputVars):
                network.setLowerBound(
                    network.inputVars[0][0][ind], input_sample.values[0][ind] - eps
                )
                network.setUpperBound(
                    network.inputVars[0][0][ind], input_sample.values[0][ind] + eps
                )

            network_solution = network.solve(options=marabou_options)
            solution_stats = extract_marabou_solution_stats(network_solution)
            summed_solution_stats = add_marabou_solution_stats(
                summed_solution_stats, solution_stats
            )
            if network_solution[1].hasTimedOut():
                break
            if len(network_solution[0]) > 0:
                extr_solution = extract_marabou_solution_point(network_solution, network)
                solution = network_solution
                diff_input = abs(
                    np.array(extr_solution[0]) - input_sample.values[0]
                ).max()
                diff_output_max = abs(
                    np.array(extr_solution[1]) - output_sample.values[0]
                ).max()
                # found solution
                if (diff_input < eps + accuracy) and (
                    diff_output_max > delta - accuracy
                ):
                    eps = eps - eps_change
                else:
                    eps = eps + eps_change
            else:
                eps = eps + eps_change
            if eps_change <= accuracy:
                found_closest_eps = True
            eps_change = eps_change / 2
        end_time = time.time()
        tot_time = end_time - start_time
        return solution, tot_time, eps, summed_solution_stats

    def plot_and_save(
        self,
        result_dict,
        key,
        solution,
        network,
        input_sample,
        output_sample,
        eps,
        tot_time,
        tot_solution_stats,
        collapsing,
    ):
        if solution is not None:
            solution_stat_dict = extract_marabou_solution_stats(solution)
            extr_solution = extract_marabou_solution_point(solution, network)

            # calculate respective differences
            diff_input = abs(np.array(extr_solution[0]) - input_sample.values[0]).max()
            diff_output_max = abs(
                np.array(extr_solution[1]) - output_sample.values[0]
            ).max()
            diff_input_output_sample = (
                abs(input_sample.values[0] - output_sample.values[0])
                .max()
                .astype(np.float64)
            )

            # save figures
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=[20, 20])
            fig.suptitle(f"Calculation took {tot_time}")
            ax[0].plot(input_sample.values[0], label="input_sample")
            ax[0].plot(extr_solution[0], label="input_solution")
            ax[0].set_title(
                f"""L_infty dist between x and y is at most {eps}
                    (up to accuracy), real: {diff_input}"""
            )
            ax[0].legend()
            ax[0].set_ylim(-1.25, 1.25)
            ax[1].plot(output_sample.values[0], label="output_sample")
            ax[1].plot(extr_solution[1], label="output_solution")
            ax[1].set_title(
                f"""L_infty dist between f(x) and f(y) is at
                    least {self.delta} (real: {diff_output_max})"""
            )
            ax[1].legend()
            ax[1].set_ylim(-1.25, 1.25)
            self.save_figure(run_inst, fig, f"marabou_robust_same_pairs_label_{key}")

            # save statistics
            result_dict["label_" + str(key)] = {
                "calc_time": tot_time,
                "robustness": eps,
                "delta": self.delta,
                "real_robustness": diff_input,
                "real_delta": diff_output_max,
                "input_output_sample_diff": diff_input_output_sample,
            }
            result_dict["label_" + str(key)].update(solution_stat_dict)
            tot_solution_stats = update_to_tot_key(tot_solution_stats)
            result_dict["label_" + str(key)].update(tot_solution_stats)
            result_dict["collapsing"] = collapsing

            # save exact solutions
            self.save_csv(
                run_inst,
                input_sample,
                "input_sample",
                subfolder=f"results_robust_{key}",
            )
            self.save_csv(
                run_inst,
                output_sample,
                "output_sample",
                subfolder=f"results_robust_{key}",
            )
            self.save_csv(
                run_inst,
                pd.DataFrame(extr_solution[0]),
                "input_solution",
                subfolder=f"results_robust_{key}",
            )
            self.save_csv(
                run_inst,
                pd.DataFrame(extr_solution[1]),
                "output_solution",
                subfolder=f"results_robust_{key}",
            )
        else:
            result_dict["label_" + str(key)] = {
                "calc_time": tot_time,
                "robustness": None,
                "delta": self.delta,
                "real_robustness": None,
                "real_delta": None,
                "input_output_sample_diff": None,
            }
            tot_solution_stats = update_to_tot_key(tot_solution_stats)
            result_dict["label_" + str(key)].update(tot_solution_stats)
            result_dict["collapsing"] = collapsing

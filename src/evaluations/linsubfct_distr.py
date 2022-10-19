import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import evaluation


class linsubfct_distr(evaluation):
    def __init__(self, name: str = "linsubfct_distr"):
        self.name = name

    def evaluate(self, dataset, algorithm, run_inst):
        # sample indices
        linsubfct_distr = algorithm.count_lin_subfcts(
            algorithm.module, dataset.test_data()
        )
        # import pdb; pdb.set_trace()
        perc = 0.9
        fig = plt.figure(figsize=(20, 10))
        fctIndices = range(len(linsubfct_distr))
        values = list(map(lambda x: x[1], linsubfct_distr))
        plt.bar(fctIndices, values)
        self.save_figure(run_inst, fig, "model_linsubfct_distr")
        self.save_csv(run_inst, np.array(values), name="linsubfct_distr")
        plt.close("all")
        res_values = self.get_top_perc(values, perc)
        fctIndices = range(len(res_values))
        fig = plt.figure(figsize=(20, 10))
        plt.bar(fctIndices, res_values)
        self.save_figure(run_inst, fig, f"model_linsubfct_distr_{perc}")
        self.save_csv(run_inst, np.array(values), name=f"linsubfct_distr_{perc}")
        plt.close("all")
        result_dict = {}
        result_dict["lin_subfct"] = len(values)
        result_dict[f"lin_subfct_{perc}"] = len(res_values)
        self.save_json(run_inst, result_dict, "num_subfcts")

    def get_top_perc(self, values, perc):
        values = sorted(values, reverse=True)
        tot = sum(values)
        perc_tot = perc * tot
        cur_sum = 0
        res_values = []
        i = 0
        while cur_sum < perc_tot:
            cur_sum += values[i]
            res_values.append(values[i])
            i = i + 1
        return res_values

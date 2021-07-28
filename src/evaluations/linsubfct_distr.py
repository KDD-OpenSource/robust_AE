import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import evaluation


class linsubfct_distr:
    def __init__(self, eval_inst: evaluation, name: str = "linsubfct_distr"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        # sample indices
        linsubfct_distr = algorithm.count_lin_subfcts(algorithm.module,
                dataset.data())
        #import pdb; pdb.set_trace()
        fig = plt.figure(figsize=(20,10))
        fctIndices = range(len(linsubfct_distr))
        values = list(map(lambda x: x[1], linsubfct_distr))
        plt.bar(fctIndices, values)
        self.evaluation.save_figure(fig, "model_linsubfct_distr")
        plt.close("all")

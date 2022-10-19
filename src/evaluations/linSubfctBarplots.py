import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import evaluation


class linSubfctBarplots(evaluation):
    def __init__(self, name: str = "linSubfctBarplots"):
        self.name = name

    def evaluate(self, dataset, algorithm, run_inst):
        # sample indices
        import pdb

        pdb.set_trace()
        linSubfctDist = algorithm.lin_sub_fct_Counters
        for ind in range(len(linSubfctDist)):
            fig = plt.figure()
            fctIndices = range(len(linSubfctDist[ind]))
            values = list(map(lambda x: x[1], linSubfctDist[ind]))
            plt.bar(fctIndices, values)
            self.save_figure(run_inst, fig, "barplot_" + str(ind))
            plt.close("all")

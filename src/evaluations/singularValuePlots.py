import matplotlib.pyplot as plt
import numpy as np

from .evaluation import evaluation


class singularValuePlots(evaluation):
    def __init__(
        self,
        name: str = "singularValuePlots",
    ):
        self.name = name

    def evaluate(self, dataset, algorithm, run_inst):
        functions = algorithm.lin_sub_fct_Counters[-1]
        svds = []
        for function in functions:
            svds.append(np.linalg.svd(function[0].matrix))
        ymin = np.array(list(map(lambda x: x[1], svds))).min()
        ymax = np.array(list(map(lambda x: x[1], svds))).max()
        for ind in range(len(svds)):
            num_points = functions[ind][1]
            fig = plt.figure()
            plt.ylim(ymin, ymax)
            plt.title(f"Numpoints: {num_points}")
            plt.plot(svds[ind][1])
            self.save_figure(
                run_inst,
                fig,
                f"plot_{num_points}_{ind}",
                subfolder="svd_plots",
            )
            plt.close("all")

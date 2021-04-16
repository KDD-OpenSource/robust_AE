import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import evaluation


class parallelQualplots:
    def __init__(
        self, eval_inst: evaluation, name: str = "parallelQualplot", num_plots: int = 20
    ):
        self.name = name
        self.evaluation = eval_inst
        self.num_plots = num_plots
        self.plots = []

    def evaluate(self, dataset, algorithm):
        # input_points: pd.DataFrame,
        # output_points: pd.DataFrame):
        # sample indices
        input_points = dataset.data()
        output_points = algorithm.predict(input_points)
        rand_ind = np.random.choice(
            input_points.shape[0], replace=False, size=self.num_plots
        )
        for ind in rand_ind:
            fig = plt.figure()
            plt.plot(input_points.iloc[ind, :], color="blue")
            plt.plot(output_points.iloc[ind, :], color="orange")
            self.evaluation.save_figure(fig, "plot_" + str(ind))
            plt.close("all")

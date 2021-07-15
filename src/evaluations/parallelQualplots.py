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
        for label in dataset.labels.unique():
            label_indices = dataset.labels[dataset.labels ==
                    label].sample(10).index
            label_data = dataset.data().loc[
                    dataset.labels[dataset.labels==label].index]
            label_mean = label_data.mean()
            for ind in label_indices:
                fig = plt.figure(figsize=(20,10))
                mean_squared_error = (
                        (input_points.iloc[ind,:]-output_points.iloc[ind,:])**2
                        ).sum()/input_points.shape[1]
                dist_to_mean = (
                        (input_points.iloc[ind,:]-label_mean)**2
                        ).sum()/input_points.shape[1]
                # change limits to -1, 1
                plt.ylim(-1,1)
                plt.plot(input_points.iloc[ind, :], color="blue", label='Orig')
                plt.plot(output_points.iloc[ind, :], color="orange",
                        label='Reconstr')
                plt.plot(label_mean,
                        label='label_mean')
                plt.legend()
                plt.title(f'''MSE: {mean_squared_error}; Dist_to_mean:
                        {dist_to_mean}''')
                self.evaluation.save_figure(fig, "parallelPlot_"+str(label) +
                    '_' + str(ind))
                plt.close("all")

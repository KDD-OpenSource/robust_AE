import matplotlib.pyplot as plt
import matplotlib.lines as ln
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd

from .evaluation import evaluation


class inst_area_2d_plot(evaluation):
    def __init__(self, name: str = "inst_area_2d_plot", num_points=20000):
        self.name = name
        self.num_points = num_points

    def evaluate(self, dataset, algorithm, run_inst):
        input_dim = algorithm.topology[0]
        if input_dim != 2:
            raise Exception("cannot plot in 2d unless input dim is 2d too")
        fig = plt.figure(figsize=[20, 20])

        # plot background points
        randPoints = pd.DataFrame(
            np.random.uniform(low=-1, high=1, size=(self.num_points, input_dim))
        )
        inst_func_pairs = algorithm.assign_lin_subfcts_ind(algorithm.module, randPoints)
        points = pd.DataFrame(map(lambda x: x[0], inst_func_pairs))
        colors = pd.DataFrame(map(lambda x: x[1], inst_func_pairs), columns=[2])
        num_colors = len(colors[2].unique())
        joined = pd.concat([points, colors], axis=1)
        fig.suptitle(f"Number of colors is {num_colors}")
        plt.scatter(joined[0], joined[1], c=joined[2], alpha=0.5, cmap="tab20")
        color_repr = []
        for elem in joined[2].unique():
            color_repr.append(pd.DataFrame(joined[joined[2] == elem].iloc[0]))
        joined_color_repr = pd.concat(color_repr, axis=1).transpose()
        plt.scatter(joined_color_repr[0], joined_color_repr[1], c="blue")

        # save data
        points_np = pd.DataFrame(map(lambda x: x[0].numpy(), inst_func_pairs))
        joined_np = pd.concat([points_np, colors], axis=1)
        self.save_csv(run_inst, joined_np, f"scatter_2d_bound_data_{num_colors}")
        self.save_figure(run_inst, fig, "scatter_2d_boundaries")
        plt.close("all")

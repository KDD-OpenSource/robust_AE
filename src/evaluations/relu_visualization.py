import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd

from .evaluation import evaluation


class relu_visualization(evaluation):
    def __init__(self, name: str = "relu_visualization", num_points=20000):
        self.name = name
        self.num_points = num_points

    def evaluate(self, dataset, algorithm, run_inst):
        input_dim = algorithm.topology[0]
        if input_dim != 2:
            raise Exception("cannot plot in 2d unless input dim is 2d too")

        import pdb

        pdb.set_trace()

        custom_ds = np.random.uniform([-1, -1], [1, 1], size=(10000, 2)).astype(
            np.float32
        )

        #        data_loader = DataLoader(
        #            dataset=dataset.test_data().values,
        #            batch_size=1,
        #            drop_last=False,
        #            pin_memory=True,
        #        )
        data_loader = DataLoader(
            dataset=custom_ds,
            batch_size=1,
            drop_last=False,
            pin_memory=True,
        )
        res_list = []
        for inst_batch in data_loader:
            for inst in inst_batch:
                inst_list = [np.array(inst)]
                res = algorithm.module(inst)
                for key in res[1].keys():
                    if int(key) % 2 == 1:
                        inst_list.append(res[1][key].detach().numpy())
                inst_list.append(res[1][list(res[1].keys())[-1]].detach().numpy())
                res_list.append(np.hstack(inst_list))
        # results has the points as the first two dimensions. Thereafter all
        # the layers (starting from the first non-input layer) are given
        # including the final (non-relu) layer
        results = np.vstack(res_list)
        relu_outcomes = results[:, 2:]
        relu_outcomes[relu_outcomes > 0] = 1
        relu_outcomes[relu_outcomes < 0] = 0

        plot_data = pd.DataFrame(results)
        column_names = []
        for layer_num, layer in enumerate(algorithm.topology):
            for node in range(layer):
                column_names.append(str(layer_num) + "_" + str(node))

        plot_data.columns = column_names
        # plot background points

        for column in plot_data.columns[2:]:
            fig = plt.figure(figsize=[10, 10])
            plt.scatter(
                plot_data["0_0"], plot_data["0_1"], c=plot_data[column], alpha=0.5
            )
            self.save_figure(run_inst, fig, f"relu_vis_{column}")
            plt.close("all")

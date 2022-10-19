import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import evaluation


class linSub_unifPoints(evaluation):
    def __init__(self, name: str = "linSub_unifPoints", num_points=100):
        self.name = name
        self.num_points = num_points

    def evaluate(self, dataset, algorithm, run_inst):
        # sample indices
        input_dim = algorithm.topology[0]
        randPoints = pd.DataFrame(
            np.random.uniform(low=-1, high=1, size=(self.num_points, input_dim))
        )
        linsubfctCtr = algorithm.count_lin_subfcts(algorithm.module, randPoints)
        fig = plt.figure()
        fctIndices = range(len(linsubfctCtr))
        values = list(map(lambda x: x[1], linsubfctCtr))
        plt.bar(fctIndices, values)
        self.save_figure(run_inst, fig, "unifPoints_fctbarplot")
        plt.close("all")

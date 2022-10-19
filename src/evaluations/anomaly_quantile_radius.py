import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

from .evaluation import evaluation


class anomaly_quantile_radius(evaluation):
    def __init__(self, name: str = "anomaly_quantile_radius"):
        self.name = name

    def evaluate(self, dataset, algorithm, run_inst):
        test_data = dataset.test_data()
        quantiles = [0.99 + i / 1000 for i in range(10)]

        anomaly_scores_normal = algorithm.calc_anomalyScores(
            algorithm.module, test_data
        )
        import pdb

        pdb.set_trace()
        radii = np.quantile(anomaly_scores_normal, quantiles)
        result_dict = dict(zip(quantiles, radii))
        self.run_inst.save_json(result_dict, "quantiles")

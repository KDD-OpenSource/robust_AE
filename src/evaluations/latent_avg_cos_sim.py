import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist

from .evaluation import evaluation


class latent_avg_cos_sim(evaluation):
    def __init__(self, name: str = "latent_avg_cos_sim"):
        self.name = name

    def evaluate(self, dataset, algorithm, run_inst):
        predictions = algorithm.predict(dataset.train_data())
        center_diffs = predictions - algorithm.center
        pw_cosines = pdist(center_diffs, "cosine")
        cos_sim_sum = abs(pw_cosines - 1).sum()
        num_preds = predictions.shape[0]
        num_combs = (num_preds * (num_preds - 1)) / 2
        cos_sim_sum = cos_sim_sum / num_combs
        result_dict = {}
        result_dict["avg_cos_sim"] = cos_sim_sum
        self.save_json(run_inst, result_dict, "latent_avg_cos_sim")

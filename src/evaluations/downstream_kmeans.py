import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from .evaluation import evaluation


class downstream_kmeans:
    def __init__(
        self, eval_inst: evaluation, name: str = "downstream_kmeans"
    ):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        latent_repr = algorithm.extract_latent(dataset.data())
        num_clusters = len(dataset.labels.unique())
        kmeans_latent = KMeans(n_clusters = num_clusters).fit(latent_repr)
        kmeans_orig = KMeans(n_clusters = num_clusters).fit(dataset.data())
        adj_rand_ind_orig_latent = adjusted_rand_score(kmeans_latent.labels_,
                kmeans_orig.labels_)
        adj_rand_ind_label_orig = adjusted_rand_score(dataset.labels,
                kmeans_orig.labels_)
        adj_rand_ind_label_latent = adjusted_rand_score(dataset.labels,
                kmeans_latent.labels_)
        result_dict = {}
        result_dict['orig_latent'] = adj_rand_ind_orig_latent
        result_dict['label_orig'] = adj_rand_ind_label_orig
        result_dict['label_latent'] = adj_rand_ind_label_latent
        self.evaluation.save_json(result_dict, 'adj_rand_index')
        # additional metrics could be evaluated

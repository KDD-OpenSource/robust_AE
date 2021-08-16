import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from .evaluation import evaluation


class downstream_naiveBayes:
    def __init__(self, eval_inst: evaluation, name: str = "downstream_naiveBayes"):
        self.name = name
        self.evaluation = eval_inst

    def evaluate(self, dataset, algorithm):
        latent_repr_train = algorithm.extract_latent(dataset.train_data())
        latent_repr_test = algorithm.extract_latent(dataset.test_data())
        # num_classes = len(dataset.test_labels.unique())
        gaussian_orig = GaussianNB()
        gaussian_orig.fit(dataset.train_data(), dataset.train_labels)
        orig_res = gaussian_orig.predict(dataset.test_data())
        # compare orig_res to dataset.test_labels

        gaussian_latent = GaussianNB()
        gaussian_latent.fit(latent_repr_train, dataset.train_labels)
        latent_res = gaussian_latent.predict(latent_repr_test)

        acc_orig = accuracy_score(dataset.test_labels, orig_res)
        acc_latent = accuracy_score(dataset.test_labels, latent_res)
        result_dict = {}
        result_dict["acc_orig"] = acc_orig
        result_dict["acc_latent"] = acc_latent
        self.evaluation.save_json(result_dict, "accuracy_naiveBayes")

        # kmeans_latent = KMeans(n_clusters = num_clusters).fit(latent_repr)
        # kmeans_orig = KMeans(n_clusters = num_clusters).fit(dataset.test_data())
        # adj_rand_ind_orig_latent = adjusted_rand_score(kmeans_latent.labels_,
        #        kmeans_orig.labels_)
        # adj_rand_ind_label_orig = adjusted_rand_score(dataset.test_labels,
        #        kmeans_orig.labels_)
        # adj_rand_ind_label_latent = adjusted_rand_score(dataset.test_labels,
        #        kmeans_latent.labels_)
        # result_dict = {}
        # result_dict['orig_latent'] = adj_rand_ind_orig_latent
        # result_dict['label_orig'] = adj_rand_ind_label_orig
        # result_dict['label_latent'] = adj_rand_ind_label_latent
        # self.evaluation.save_json(result_dict, 'adj_rand_index')
        # additional metrics could be evaluated
        # for instance one could calculate calculate

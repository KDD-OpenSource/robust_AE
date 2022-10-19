import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from .evaluation import evaluation


class downstream_knn(evaluation):
    def __init__(self, name: str = "downstream_knn"):
        self.name = name

    def evaluate(self, dataset, algorithm, run_inst):
        latent_repr_train = algorithm.extract_latent(dataset.train_data())
        latent_repr_test = algorithm.extract_latent(dataset.test_data())

        knn_orig = KNeighborsClassifier(n_neighbors=10)
        knn_orig.fit(dataset.train_data(), dataset.train_labels)
        orig_res = knn_orig.predict(dataset.test_data())

        knn_latent = KNeighborsClassifier(n_neighbors=10)
        knn_latent.fit(latent_repr_train, dataset.train_labels)
        latent_res = knn_latent.predict(latent_repr_test)

        acc_orig = accuracy_score(dataset.test_labels, orig_res)
        acc_latent = accuracy_score(dataset.test_labels, latent_res)
        result_dict = {}
        result_dict["acc_orig"] = acc_orig
        result_dict["acc_latent"] = acc_latent
        self.save_json(run_inst, result_dict, "accuracy_knn")

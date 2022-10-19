import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from .evaluation import evaluation


class downstream_rf(evaluation):
    def __init__(self, name: str = "downstream_rf"):
        self.name = name

    def evaluate(self, dataset, algorithm, run_inst):
        latent_repr_train = algorithm.extract_latent(dataset.train_data())
        latent_repr_test = algorithm.extract_latent(dataset.test_data())
        rf_orig = RandomForestClassifier(n_estimators=10)
        rf_orig.fit(dataset.train_data(), dataset.train_labels)
        orig_res = rf_orig.predict(dataset.test_data())

        rf_latent = RandomForestClassifier(n_estimators=10)
        rf_latent.fit(latent_repr_train, dataset.train_labels)
        latent_res = rf_latent.predict(latent_repr_test)

        acc_orig = accuracy_score(dataset.test_labels, orig_res)
        acc_latent = accuracy_score(dataset.test_labels, latent_res)
        result_dict = {}
        result_dict["acc_orig"] = acc_orig
        result_dict["acc_latent"] = acc_latent
        self.save_json(run_inst, result_dict, "accuracy_rf")

import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from sklearn.metrics import accuracy_score

from .evaluation import evaluation


class superv_accuracy(evaluation):
    def __init__(self, name: str = "superv_accuracy"):
        self.name = name

    def evaluate(self, dataset, algorithm, run_inst):
        test_data = dataset.test_data()
        test_labels = dataset.test_labels
        predictions = algorithm.predict(test_data)["pred_label"]
        accuracy = accuracy_score(test_labels, predictions)
        result_dict = {}
        result_dict["accuracy"] = accuracy
        self.save_json(run_inst, result_dict, "test_accuracy")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import evaluation


class calc_linfct_volume(evaluation):
    def __init__(self, name: str = "calc_linfct_volume"):
        self.name = name

    def evaluate(self, dataset, algorithm, run_inst):
        # sample indices
        """calculates the  fct volume of all class means"""
        label_means = dataset.calc_label_means(subset="test")
        import pdb

        pdb.set_trace()
        fct_volumes = algorithm.calc_linfct_volume(label_means)

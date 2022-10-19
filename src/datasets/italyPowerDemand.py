import numpy as np
import pandas as pd
from .dataset import dataset
from sklearn.datasets import fetch_openml


class italyPowerDemand(dataset):
    def __init__(
        self,
        name: str = "ItalyPowerDemand",
        file_path: str = None,
        subsample: int = None,
    ):
        super().__init__(name, file_path, subsample)
        self.balance = True
        self.delimiter = ","
        self.header = None
        self.index_col = None
        self.label_col_train = 0
        self.label_col_test = 0

import numpy as np
import pandas as pd
from .dataset import dataset
from sklearn.datasets import fetch_openml


class syntheticControl(dataset):
    def __init__(
        self,
        name: str = "SyntheticControl",
        file_path: str = None,
        subsample: int = None,
    ):
        super().__init__(name, file_path, subsample)
        self.balance = True
        self.delimiter = "\t"
        self.header = None
        self.index_col = None
        self.label_col_train = 0
        self.label_col_test = 0

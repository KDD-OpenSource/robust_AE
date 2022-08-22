import numpy as np
import re
import random
import pandas as pd
from .dataset import dataset
from sklearn.datasets import fetch_openml


class crop(dataset):
    def __init__(
        self,
        name: str = "Crop",
        class1: int = None,
        class2: int = None,
        file_path: str = None,
        subsample: int = None,
    ):
        # note that in a previous git version there exists old code to create two classes
        super().__init__(name, file_path, subsample)
        self.delimiter = '\t'
        self.header = None
        self.index_col = None
        self.label_col_train = 0
        self.label_col_test = 0

import numpy as np
import pandas as pd
from .dataset import dataset
from src.utils.utils import get_proj_root


class moteStrain(dataset):
    def __init__(
        self,
        name: str = "MoteStrain",
        file_path: str = None,
        subsample: int = None,
    ):
        super().__init__(name, file_path, subsample)
        self.balance = True
        self.label_col_train = 0
        self.index_col = None
        self.header = None
        self.delimiter = '\t'
        self.label_col_test = 0

 #   def create(self):
 #       root = get_proj_root()

 #       dataset_train = pd.read_csv(
 #           str(root) + "/datasets/MoteStrain/MoteStrain_TRAIN.csv",
 #           header=None,
 #           delimiter="\t",
 #       )
 #       dataset_test = pd.read_csv(
 #           str(root) + "/datasets/MoteStrain/MoteStrain_TEST.csv",
 #           header=None,
 #           delimiter="\t",
 #       )

 #       dataset = self.join_and_shuffle(dataset_train, dataset_test)
 #       dataset_train, dataset_test = self.balance_split(
 #           dataset
 #       )

 #       self.train_labels = dataset_train[0]
 #       dataset_train = dataset_train.drop([0], axis=1)
 #       self._train_data = dataset_train
 #       self.test_labels = dataset_test[0]
 #       dataset_test = dataset_test.drop([0], axis=1)
 #       self._test_data = dataset_test

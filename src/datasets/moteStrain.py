import numpy as np
import pandas as pd
from .dataset import dataset
from sklearn.datasets import fetch_openml
from ..utils.utils import get_proj_root


class moteStrain(dataset):
    def __init__(
        self,
        name: str = "moteStrain",
        file_path: str = None,
        subsample: int = None,
    ):
        super().__init__(name, file_path, subsample)

    def create(self):
        import pdb; pdb.set_trace()
        root = get_proj_root()

        dataset_train = pd.read_csv(
            "./datasets/MoteStrain/MoteStrain_TRAIN",
            header=None,
            delimiter="\t",
        )
        dataset_test = pd.read_csv(
            "./datasets/MoteStrain/MoteStrain_TEST",
            header=None,
            delimiter="\t",
        )

        dataset_train, dataset_test = self.rebalance_train_test(
            dataset_train, dataset_test
        )
        self.train_labels = dataset_train[0]
        dataset_train.drop([0], inplace=True, axis=1)
        self._train_data = dataset_train
        self.test_labels = dataset_test[0]
        dataset_test.drop([0], inplace=True, axis=1)
        self._test_data = dataset_test

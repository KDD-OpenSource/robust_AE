import pandas as pd
from .dataset import dataset
from src.utils.utils import get_proj_root


class twoLeadEcg(dataset):
    def __init__(
        self,
        name: str = "twoLeadEcg",
        file_path: str = None,
        subsample: int = None,
    ):
        super().__init__(name, file_path, subsample)

    def create(self):
        root = get_proj_root()
        dataset_train = pd.read_csv(
            str(root) + "/datasets/TwoLeadECG/TwoLeadECG_TRAIN",
            header=None,
            delimiter="\t",
        )
        dataset_test = pd.read_csv(
            str(root) + "/datasets/TwoLeadECG/TwoLeadECG_TEST",
            header=None,
            delimiter="\t",
        )
        dataset = self.join_and_shuffle(dataset_train, dataset_test)
        dataset_train, dataset_test = self.balance_split(
            dataset
        )

        self.train_labels = dataset_train[0]
        dataset_train = dataset_train.drop([0], axis=1)
        self._train_data = dataset_train
        self.test_labels = dataset_test[0]
        dataset_test = dataset_test.drop([0], axis=1)
        self._test_data = dataset_test

import pandas as pd
from .dataset import dataset
from src.utils.utils import get_proj_root


class proximalPhalanxOutlineCorrect(dataset):
    def __init__(
        self,
        name: str = "ProximalPhalanxOutlineCorrect",
        file_path: str = None,
        subsample: int = None,
    ):
        super().__init__(name, file_path, subsample)
        self.balance = True
        self.label_col_train = 0
        self.index_col = None
        self.header = None
        self.delimiter = "\t"
        self.label_col_test = 0

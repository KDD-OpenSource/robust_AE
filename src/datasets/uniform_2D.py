import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from .dataset import dataset


class uniform_2D(dataset):
    def __init__(
        self,
        name: str = "uniform_2D",
        file_path: str = None,
        subsample: int = None,
        scale: bool = False,
        x_min=0,
        x_max=1,
        y_min=0,
        y_max=1,
        num_samples: int = 10000,
        num_anomalies: int = 0,
        num_testpoints: int = 0,
    ):
        super().__init__(name, file_path, subsample, scale)
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.num_samples = num_samples
        self.num_anomalies = num_anomalies
        self.num_testpoints = num_testpoints

    def create(self):
        """
        creates a cloud of points that are uniformly distributed in 2D between
        the given limits
        """
        data = pd.DataFrame(
            np.random.uniform(
                [self.x_min, self.y_min],
                [self.x_max, self.y_max],
                size=(self.num_samples, 2),
            )
        )
        self._train_data = data
        self.train_labels = pd.Series(0, range(self.num_samples))

        test_data = pd.DataFrame(
            np.random.uniform(
                [self.x_min, self.y_min],
                [self.x_max, self.y_max],
                size=(self.num_testpoints, 2),
            )
        )
        self._test_data = test_data
        self.test_labels = pd.Series(0, range(self.num_testpoints))

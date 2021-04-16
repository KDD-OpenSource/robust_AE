import numpy as np
import pandas as pd
from .dataset import dataset


class gaussianClouds(dataset):
    def __init__(
        self,
        name: str = "gaussianCloud",
        spacedim: int = 20,
        clouddim: int = 5,
        num_clouds: int = 5,
        num_datapoints: int = 2000,
    ):
        super().__init__(name)
        self.spacedim = spacedim
        self.clouddim = clouddim
        self.num_clouds = num_clouds
        self.num_datapoints = num_datapoints

    def load(self):
        """
        creates a synthetic DS by projecting a gaussian cloud with unit
        variance into a high dim space
        """
        points_per_cloud = int(self.num_datapoints / self.num_clouds)
        random_matrices = []
        random_centers = []
        random_points = []
        for _ in range(self.num_clouds):
            random_matrices.append(
                np.random.uniform(low=-1, high=1, size=(self.spacedim, self.clouddim))
            )
            random_centers.append(
                np.random.uniform(low=0, high=1, size=(self.clouddim))
            )
            random_points.append(
                np.random.normal(
                    loc=random_centers[-1],
                    scale=1,
                    size=(points_per_cloud, self.clouddim),
                ).transpose()
            )
            random_points[-1] = np.dot(random_matrices[-1], random_points[-1])

        data = np.hstack(random_points).transpose()
        data = pd.DataFrame(data)
        import pdb

        pdb.set_trace()
        # check if correct
        data = self.scale_data(data)
        self._data = data

    def save(self):
        pass

import numpy as np
import pandas as pd
from .dataset import dataset


class gaussianClouds(dataset):
    def __init__(
        self,
        name: str = "gaussianClouds",
        file_path: str = None,
        subsample: int = None,
        spacedim: int = 20,
        clouddim: int = 5,
        num_clouds: int = 5,
        num_samples: int = 2000,
        num_anomalies: int = 20,
    ):
        super().__init__(name, file_path, subsample)
        self.spacedim = spacedim
        self.clouddim = clouddim
        self.num_clouds = num_clouds
        self.num_samples = num_samples
        self.num_anomalies = num_anomalies

    def create(self):
        """
        creates a synthetic DS by projecting a gaussian cloud with unit
        variance into a high dim space
        """
        points_per_cloud = int(self.num_samples / self.num_clouds)
        random_matrices = []
        random_centers = []
        random_points = []
        label = 0
        self.labels = pd.Series()
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
            labels = pd.Series(label, range(points_per_cloud))
            self.labels = self.labels.append(labels, ignore_index = True)
            label += 1
        data = np.hstack(random_points).transpose()
        anomalies = np.random.uniform(
            low=data.min(), high=data.max(), size=(self.num_anomalies, self.spacedim)
        )
        data = np.vstack([data, anomalies])
        anom_labels = pd.Series(-1, range(self.num_anomalies))
        self.labels = self.labels.append(anom_labels, ignore_index=True)
        data = pd.DataFrame(data)
        self._data = data
        # add diffs to means by adding a series (self.dists_to_means)

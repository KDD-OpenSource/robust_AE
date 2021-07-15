import numpy as np
import pandas as pd
from .dataset import dataset
from sklearn.datasets import fetch_openml


class mnist(dataset):
    def __init__(
        self,
        name: str = "mnist",
        file_path: str = None,
        subsample: int = None,
        num_samples: int = 2000,
        num_anomalies: int = 20,
    ):
        super().__init__(name, file_path, subsample)
        self.num_samples = num_samples
        self.num_anomalies = num_anomalies

    def create(self):
        mnist_data_file_train = pd.read_csv('./datasets/mnist/mnist_train.csv')
        mnist_data_file_test = pd.read_csv('./datasets/mnist/mnist_test.csv')
        mnist_data_file_tot = pd.concat([mnist_data_file_train,
            mnist_data_file_test])
        mnist_data_file_tot['new_index'] = range(mnist_data_file_tot.shape[0])
        mnist_data_file_tot.set_index('new_index', inplace=True)

        #mnist_data_file_tot.drop('label', inplace=True, axis=1)
        mnist_data = mnist_data_file_tot.sample(n=self.num_samples)
        #self.labels = mnist_data['label']
        self.labels = pd.DataFrame(mnist_data['label'])
        self.labels['ind'] = range(self.labels.shape[0])
        self.labels.set_index('ind', inplace=True)
        self.labels = self.labels['label']
        anomaly_labels = pd.Series(-1, range(self.num_anomalies))
        self.labels = self.labels.append(anomaly_labels, ignore_index = True)
        mnist_data.drop('label', inplace=True, axis=1)
        anomalies = np.random.uniform(low=0, high=255, size=(self.num_anomalies,
            mnist_data.shape[1]))
        data = np.vstack([mnist_data, anomalies])
        #data = data.astype(np.float32)
        data = pd.DataFrame(data)
        self._data = data
        #data = self.scale_data(data, min_val = -1, max_val=1)
        #anomaly_labels = pd.Series(-1, range(self.num_samples,
            #self.num_samples + self.num_anomalies))
        
        # add label '-1' for anomalous points
        #self.anomalies = pd.DataFrame(data.iloc[-self.num_anomalies :])
        #self.normal_data = pd.DataFrame(data.iloc[:-self.num_anomalies])
import abc
import os
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class dataset:
    def __init__(self, name: str, file_path: str, subsample: int = None):
        self.name = name
        self.file_path = file_path
        self.subsample = subsample
        self._data = None

    def __str__(self) -> str:
        return self.name

    def load(self):
        # note that self.file_path will change when reading from the properties
        # file
        file_path = self.file_path
        for dataset_file in os.listdir(file_path):
            if 'Properties' in dataset_file:
                with open(os.path.join(self.file_path,dataset_file)) as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        for dataset_prop in self.__dict__.keys():
                            if row[0] == dataset_prop:
                                self.__dict__[dataset_prop] = row[1]
            else:
                data_df = pd.read_csv(os.path.join(file_path, dataset_file),
                        index_col=0)
                self.labels = data_df['label']
                data_df.drop(['label'], axis=1, inplace = True)
                self._data = data_df

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        joined_df = self.data().merge(pd.DataFrame(self.labels, columns =
            ['label']), left_index=True, right_index=True)
        joined_df.to_csv(path+'/' + self.name + '.csv')
        writer = csv.writer(open(path+'/' + self.name + '_Properties.csv','w'))
        for key, val in self.__dict__.items():
            if key == '_data' or key == 'labels':
                continue
            writer.writerow([key, val])

    def preprocess(self):
        if self.subsample:
            self._data = self._data.sample(self.subsample)
            self.labels = self.labels[self._data.index]
            self.labels = pd.Series(self.labels.values)
        data = self._data.values
        data = data.astype(np.float32)
        data = pd.DataFrame(data)
        data = self.scale_data(data)
        self._data = data

    def scale_data(self, data: pd.DataFrame, min_val=-1, max_val=1):
        scaler = MinMaxScaler(feature_range=(min_val, max_val))
        scaler.fit(data)
        return pd.DataFrame(scaler.transform(data), columns=data.columns)

    def calc_dist_to_label_mean(self):
        self.data()
        self.dists_to_label_mean = pd.Series(0, self.labels.index)
        for label in self.labels.unique():
            label_mean = self.data().loc[self.labels==label].mean()
            diffs = self.data().loc[self.labels==label] - label_mean
            dists = ((diffs**2).sum(axis=1))**(1/2)
            self.dists_to_label_mean.loc[self.labels==label] = dists


    def add_anomalies(self):
        # to be implemented later: adds anomalies to already existing datasets
        pass

    def data(self):
        if self._data is None:
            if self.file_path is None:
                self.create()
            else:
                self.load()
            self.preprocess()
        return self._data

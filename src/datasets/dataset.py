import abc
import os
import torch
import csv
import pandas as pd
import numpy as np
from src.utils.utils import get_proj_root
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors


class dataset:
    def __init__(self, name: str, file_path: str, subsample: int = None, scale=True):
        self.name = name
        self.file_path = file_path
        self.subsample = subsample
        self._train_data = None
        self._test_data = None
        self.scale = scale

    def __str__(self) -> str:
        return self.name

    def create(self):
        root = get_proj_root()
        train_file = '/datasets/' + self.name + '/' + self.name + '_TRAIN.csv'
        test_file = '/datasets/' + self.name + '/' + self.name + '_TEST.csv'

        dataset_train = pd.read_csv(str(root) + train_file,
                header = self.header, delimiter = self.delimiter, index_col =
                self.index_col)
        dataset_test = pd.read_csv(str(root) + test_file,
                header = self.header, delimiter = self.delimiter, index_col =
                self.index_col)

        if hasattr(self, 'balance'):
            dataset = self.join_and_shuffle(dataset_train, dataset_test)
            dataset_train, dataset_test = self.balance_split(
                dataset
            )

        if self.label_col_train is None:
            self.train_labels = pd.Series(0, index=dataset_train.index)
            #dataset_train['label'] = 0
        else:
            self.train_labels = dataset_train.loc[:,self.label_col_train]
            #dataset_train.loc[:,'label'] = dataset_train.loc[:,self.label_col_train]
            dataset_train = dataset_train.drop([self.label_col_train], axis = 1)


        #self.train_labels = dataset_train['label']
        #dataset_train = dataset_train.drop(['label'], axis=1)
        self._train_data = dataset_train

        self.test_labels = dataset_test[self.label_col_test]
        dataset_test = dataset_test.drop([self.label_col_test], axis=1)
        self._test_data = dataset_test


    def load(self):
        # note that self.file_path will change when reading from the properties
        # file
        file_path = self.file_path
        for dataset_file in os.listdir(file_path):
            if "Properties.csv" in dataset_file:
                with open(os.path.join(self.file_path, dataset_file)) as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        for dataset_prop in self.__dict__.keys():
                            if row[0] == dataset_prop and dataset_prop not in [
                                "_train_data",
                                "_test_data",
                                "train_labels",
                                "test_labels",
                                "scale"
                            ]:
                                self.__dict__[dataset_prop] = row[1]
            elif "train.csv" in dataset_file:
                data_df = pd.read_csv(
                    os.path.join(file_path, dataset_file), index_col=0
                )
                self.train_labels = data_df["label"]
                data_df.drop(["label"], axis=1, inplace=True)
                self._train_data = data_df
            elif "test.csv" in dataset_file:
                data_df = pd.read_csv(
                    os.path.join(file_path, dataset_file), index_col=0
                )
                self.test_labels = data_df["label"]
                data_df.drop(["label"], axis=1, inplace=True)
                self._test_data = data_df
            elif 'readme.md' in dataset_file:
                pass
            else:
                print("No appropriate keyword in datafile")

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        joined_df_train = self.train_data().merge(
            pd.DataFrame(self.train_labels.rename("label"), columns=["label"]),
            left_index=True,
            right_index=True,
        )
        joined_df_train.to_csv(path + "/" + self.name + "train.csv")
        joined_df_test = self.test_data().merge(
            pd.DataFrame(self.test_labels.rename("label"), columns=["label"]),
            left_index=True,
            right_index=True,
        )
        joined_df_test.to_csv(path + "/" + self.name + "test.csv")
        with open(path + "/" + self.name + "_Properties.csv", "w") as csv_file:
        #writer = csv.writer(open(path + "/" + self.name + "_Properties.csv", "w"))
            writer = csv.writer(csv_file)
            # add saving the test file
            for key, val in self.__dict__.items():
                if key == "_train_data" or key == "labels":
                    continue
                writer.writerow([key, val])

    def preprocess(self):
        if self.subsample:
            self._train_data = self._train_data.sample(self.subsample)
            self.train_labels = self.train_labels[self._train_data.index]
            self.train_labels = pd.Series(self.train_labels.values)
        train_index = self._train_data.index
        test_index = self._test_data.index
        train_data = self._train_data.values
        train_data = train_data.astype(np.float32)
        train_data = pd.DataFrame(train_data, index=train_index)
        if self.scale == True:
            train_data, scaler = self.scale_data(train_data,
                    return_scaler=True)
                    #scale_type = self.scale_type, min_val =
                    #self.scale_min, max_val = self.scale_max)
        self._train_data = train_data
        test_data = self._test_data.values
        test_data = test_data.astype(np.float32)
        test_data = pd.DataFrame(test_data, index=test_index)
        if self.scale == True:
            test_data = pd.DataFrame(
                scaler.transform(test_data), columns=test_data.columns, index=test_index
            )
        self._test_data = test_data
        self.train_labels = self.train_labels.astype(np.float32)
        self.test_labels = self.test_labels.astype(np.float32)

    def kth_nearest_neighbor_dist(self, k):
        neigh = NearestNeighbors(n_neighbors = k)
        neigh.fit(self.train_data())
        dist, ind = neigh.kneighbors(self.train_data())
        dists = [dist[i][k-1] for i in range(len(dist))]
        return dists

    def kth_nearest_neighbor_model(self, k):
        neigh = NearestNeighbors(n_neighbors = k)
        neigh.fit(self.train_data())
        return neigh

    def get_last_nearest_neighbor_dist(self, neigh, instance):
        # instance is supposed to be a single sample
        dist, ind = neigh.kneighbors(instance.reshape(1,-1))
        return dist.flatten()[-1]

    def get_nearest_neighbor_insts(self, neigh, instance):
        # instance is supposed to be a single sample
        dist, ind = neigh.kneighbors(instance.reshape(1,-1))
        res = torch.tensor(self.train_data().loc[ind.flatten()[1:]].values)
        return res

    def scale_data(
        self, data: pd.DataFrame,
        return_scaler=False
        #, min_val=-1, max_val=1, return_scaler=False,
        #scale_type = 'MinMax'
    ):
        if self.scale_type == 'MinMax':
            scaler = MinMaxScaler(feature_range=(self.scale_min, self.scale_max))
        elif self.scale_type == 'centered': # x - mean(x)/ max_value
            scaler = CenteredMaxAbsScaler()

        data_index = data.index
        scaler.fit(data)
        if not return_scaler:
            return pd.DataFrame(
                scaler.transform(data), columns=data.columns, index=data_index
            )
        else:
            return (
                pd.DataFrame(
                    scaler.transform(data), columns=data.columns, index=data_index
                ),
                scaler,
            )

    def calc_label_means(self, subset):
        label_means = {}
        if subset == "train":
            if self._train_data is None:
                self.train_data()
            for label in self.train_labels.unique():
                label_mean = self.train_data().loc[self.train_labels == label].mean()
                label_means[label] = label_mean
        elif subset == "test":
            if self._test_data is None:
                self.test_data()
            for label in self.test_labels.unique():
                label_mean = self.test_data().loc[self.test_labels == label].mean()
                label_means[label] = label_mean
        else:
            raise Exception("Not yet implemented")
        return label_means

    def calc_dist_to_label_mean(self, subset):
        if subset == "train":
            self.train_data()
            self.dists_to_label_mean = pd.Series(0, self.train_labels.index)
            for label in self.train_labels.unique():
                label_mean = self.train_data().loc[self.train_labels == label].mean()
                diffs = self.train_data().loc[self.train_labels == label] - label_mean
                dists = ((diffs ** 2).sum(axis=1)) ** (1 / 2)
                self.dists_to_label_mean.loc[self.train_labels == label] = dists
        elif subset == "test":
            self.test_data()
            self.dists_to_label_mean = pd.Series(0, self.test_labels.index)
            for label in self.test_labels.unique():
                label_mean = self.test_data().loc[self.test_labels == label].mean()
                diffs = self.test_data().loc[self.test_labels == label] - label_mean
                dists = ((diffs ** 2).sum(axis=1)) ** (1 / 2)
                self.dists_to_label_mean.loc[self.test_labels == label] = dists
        else:
            raise Exception("Not yet implemented")

    def add_anomalies(self):
        # to be implemented later: adds anomalies to already existing datasets
        pass

    def train_data(self):
        if self._train_data is None:
            if self.file_path is None:
                self.create()
            else:
                self.load()
            self.preprocess()
        return self._train_data

    def test_data(self):
        # note that we never create test data (this must happen when the
        # traindata is created as it should follow its distribution
        if self._test_data is None:
            self.load()
            # we do not preprocess any more. Preprocessing is done during
            # training as it is based on training data's values
            # self.preprocess()
        return self._test_data

    def get_anom_labels_from_test_labels(self):
        # returns 1 for anomalies, 0 for normal data
        anom_series = (self.test_labels<0).astype(int)
        return anom_series

    def join_and_shuffle(self, dataset_train, dataset_test):
        dataset_train = dataset_train.sample(frac=1)
        dataset_test = dataset_test.sample(frac=1)
        dataset = pd.concat([dataset_train, dataset_test])
        return dataset

    #def rebalance_train_test(self, dataset_train, dataset_test):
    def balance_split(self, dataset):
        tot_num_samples = dataset.shape[0]
        train_num_samples = int(tot_num_samples / 2)
        test_num_samples = int(tot_num_samples / 2)
        dataset_train = dataset[:train_num_samples]
        dataset_test = dataset[train_num_samples : train_num_samples + test_num_samples]
        return dataset_train, dataset_test

    def mirror_dataset_frac(self):
        # function to obtain e.g. a negative sine curve
        # arguments: value to be mirrored at, fraction of dataset, 'behaviour'
        # (random/starting from an index etc.)
        self.train_labels = pd.Series(0, range(int(self.num_samples / 2)))
        tmp = pd.Series(0, range(int(self.num_samples / 2)))
        # tmp = pd.Series(1, range(int(self.num_samples/2)))
        self.train_labels = self.train_labels.append(tmp, ignore_index=True)

class CenteredMaxAbsScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.max_abs_ = None
        self.mean_ = None

    def fit(self, X, y=None):
        self.max_abs_ = X.max().max()
        self.mean_ = X.mean()
        return self

    def transform(self, X):
        return (X - self.mean_)/self.max_abs_

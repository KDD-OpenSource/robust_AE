import abc
import os
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class dataset:
    def __init__(self, name: str, file_path: str, subsample: int = None,
            scale=True):
        self.name = name
        self.file_path = file_path
        self.subsample = subsample
        self._train_data = None
        self._test_data = None
        self.scale = scale

    def __str__(self) -> str:
        return self.name

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
        writer = csv.writer(open(path + "/" + self.name + "_Properties.csv", "w"))
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
            train_data, scaler = self.scale_data(train_data, return_scaler=True)
        self._train_data = train_data
        test_data = self._test_data.values
        test_data = test_data.astype(np.float32)
        test_data = pd.DataFrame(test_data, index=test_index)
        if self.scale == True:
            test_data = pd.DataFrame(
                scaler.transform(test_data), columns=test_data.columns, index=test_index
            )
        self._test_data = test_data

    def scale_data(
        self, data: pd.DataFrame, min_val=-1, max_val=1, return_scaler=False
    ):
        scaler = MinMaxScaler(feature_range=(min_val, max_val))
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
            self.preprocess()
        return self._test_data

    def rebalance_train_test(self, dataset_train, dataset_test):
        # shuffle:
        dataset_train = dataset_train.sample(frac = 1)
        dataset_test = dataset_test.sample(frac = 1)
        # concat and divide
        tot_num_samples = dataset_train.shape[0] + dataset_test.shape[0]
        train_num_samples = int(tot_num_samples/2)
        test_num_samples = int(tot_num_samples/2)
        df_tot = pd.concat([dataset_train, dataset_test], ignore_index =
                True)
        dataset_train = df_tot[:train_num_samples]
        dataset_test = df_tot[train_num_samples:train_num_samples + test_num_samples]
        return dataset_train, dataset_test

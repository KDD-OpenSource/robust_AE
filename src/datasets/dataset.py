import abc
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class dataset:
    def __init__(self, name: str):
        self.name = name
        self._data = None

    def __str__(self) -> str:
        return self.name

    @abc.abstractmethod
    def load(self):
        pass
        # fills _data field

    @abc.abstractmethod
    def save(self):
        pass

    def data(self):
        if self._data is None:
            self.load()
        return self._data

    def scale_data(self, data: pd.DataFrame):
        # scaler = MinMaxScaler(feature_range = (-1,1))
        scaler = MinMaxScaler(feature_range=(-0.9, 0.9))
        scaler.fit(data)
        return pd.DataFrame(scaler.transform(data), columns=data.columns)

    # Todo: implement a scaler such that all points are within -1 and 1
    # (min/max scaler?)

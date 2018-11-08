import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def get_flatten(array: np.ndarray):
    tri = []
    top = array[:-array.shape[1]]
    z = array.shape[1] - 1
    for i in range(array.shape[1] - 1):
        tri.append(array[-(1 + i)][:z])
        z -= 1
    tri = np.concatenate(tri, axis=0)
    return np.concatenate((top.flatten(), tri.flatten()))

def flatten_state(a: np.ndarray):
    # flattens the covariance matrix
    assert a.shape[0] != a.shape[1], "array has to be quadratic"
    return a[np.tri(a.shape[0]) == True]


def get_scaler(train: np.ndarray, feature_range: tuple=(0, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    return scaler.fit(train)


def read_data(file, nb_assets: int=8, lin_return: bool=False, return_array: bool=True):
    data = pd.read_csv(file, parse_dates=True, index_col=0)
    if lin_return:
        data.iloc[:, :8] = data.iloc[:, :8].pct_change(1)
        data = data.iloc[1:]
    if return_array:
        return data.values
    else:
        return data

def get_split(x: np.ndarray, y: np.ndarray, split: float=0.85):
    train_x, train_y = x[: int(x.shape[0] * split)], y[: int(x.shape[0] * split)]
    test_x, test_y = x[len(train_x):], y[len(train_y):]
    return train_x, train_y, test_x, test_y


class PrepData:
    def __int__(
        self,
        file_name: str,
        horizon: int=10,
        window_size: int=100,
        split: float=0.85,
        nb_assets: int=10
    ):
        """
        Data preparation class
        :param file_name: data set (path + filename)
        :param horizon: investment horizon
        :param window_size: evaluation window
        :param split: train/test split
        :param nb_assets: number of assets
        """

        self.horizon = horizon
        self.nb_assets = nb_assets
        self.window_size = window_size

        _data = read_data(file_name)
        self.x_trn, self.y_trn, self.x_test, self.y_test = get_split(_data[:, :-1], _data[:, -1], split=split)
        self.scaler = get_scaler(self.x_trn)

    def reshape_data(self, x, y):
        y = y[1:]
        x = self.scaler.transform(x[: -(1 + self.horizon)])
        xdata, ydata = [], []
        for step in range(x.shape[0] - self.window_size):
            xdata.append(x[0 + step: self.window_size + step])
            ydata.append(y[self.window_size + step: self.window_size + step + 1])
        xdata = np.array(xdata)
        ydata = np.array(ydata).reshape((x.shape[0] - self.window_size, y.shape[1]))
        return xdata, ydata


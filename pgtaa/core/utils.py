import pandas as pd
import numpy as np

def flatten_state(a: np.ndarray):
    # flattens the covariance matrix
    assert a.shape[0] != a.shape[1], "array has to be quadratic"
    return a[np.tri(a.shape[0]) == True]


def read_data(file, nb_assets: int=8, lin_return: bool=False, return_array: bool=True, nrows: int=None):
    data = pd.read_csv(file, parse_dates=True, index_col=0, nrows=nrows)
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

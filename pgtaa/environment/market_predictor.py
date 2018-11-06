import pandas as pd  
import numpy as np

from tensorflow.keras.models import Model
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.manifold import TSNE
from sklearn.externals import joblib

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier

from pgtaa.config import *



def reshape_data(x, y, horizon, window_size, scaler):
    y = y[1:]
    x = scaler.transform(x[: -horizon])
    xdata, ydata = [], []
    for step in range(x.shape[0] - window_size):
        xdata.append(x[0 + step: window_size + step])
        ydata.append(y[window_size + step: window_size + step + 1])
    xdata = np.array(xdata)
    ydata = np.array(ydata).reshape((x.shape[0] - window_size, y.shape[1]))
    return xdata, ydata

# Support Vector Regression
svr = SVR()

digits = load_digits()
clf = SGDClassifier(max_iter=5, tol=None).fit(digits.data, digits.target)
print(clf.score(digits.data, digits.target))
filename = 'saves/digits_classifier.joblib.pkl'
joblib.dump(clf, filename, compress=9)
# joblib.load(filename)

# up down Classifier

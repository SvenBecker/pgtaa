from tensorflow.keras.models import Model
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.manifold import TSNE
from sklearn.externals import joblib

from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier

# Support Vector Regression
svr = SVR()

digits = load_digits()
clf = SGDClassifier(max_iter=5, tol=None).fit(digits.data, digits.target)
print(clf.score(digits.data, digits.target))
filename = 'saves/digits_classifier.joblib.pkl'
joblib.dump(clf, filename, compress=9)
# joblib.load(filename)

# up down Classifier

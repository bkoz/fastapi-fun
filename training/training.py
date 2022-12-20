# -*- coding: utf-8 -*-
"""
=========================================================
SVM 3-class Classifier Training Script
=========================================================

Show below is an SVM classifier on the
four dimensions (sepal length and width, pedal length and width) of the `iris
<https://en.wikipedia.org/wiki/Iris_flower_data_set>`_ dataset.
"""

from sklearn import datasets
from sklearn import svm
import joblib

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :]  # Use all rows and all four features of the Iris data.
Y = iris.target

clf = svm.SVC()
clf.fit(X, Y)

joblib.dump(clf, 'iris-svc.joblib')

clf = joblib.load('iris-svc.joblib')

# Make a prediction using the "last" X 
y = clf.predict(X[-1:, :])

print(f"X = {X[-1:, :]}")
print(f"y = {y}")
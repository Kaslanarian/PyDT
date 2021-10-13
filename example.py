from ID3 import ID3Classifier
from C4_5 import C4_5Classifier
from CART import CARTClassifier, CARTRegressor
from plot import tree_plot
from data import load_watermelon2, load_watermelon3

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import *

import numpy as np
import matplotlib.pyplot as plt

RANDOM_STATE = 2021


def ID3test(plot=True):
    dataset = load_watermelon2()
    X, y = dataset.data, dataset.target
    train_X, test_X, train_y, test_y = train_test_split(
        X,
        y,
        train_size=0.7,
        random_state=RANDOM_STATE,
    )
    model = ID3Classifier()
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print("ID3's perf : {}".format(accuracy_score(test_y, pred)))
    if plot:
        tree_plot(
            model,
            filename="ID3",
            feature_names=dataset.feature_names,
            target_names=dataset.target_names,
            font="SimHei",
        )


def C4_5test(plot=True):
    dataset = load_watermelon3()
    X, y = dataset.data, dataset.target
    train_X, test_X, train_y, test_y = train_test_split(
        X,
        y,
        train_size=0.7,
        random_state=RANDOM_STATE,
    )
    model = C4_5Classifier()
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print("C4.5's perf : {}".format(accuracy_score(test_y, pred)))
    if plot:
        tree_plot(
            model,
            filename="C4.5",
            feature_names=dataset.feature_names,
            target_names=dataset.target_names,
            font="SimHei",
        )


def CARTclassify_test(dataset="iris", plot=True):
    name = dataset
    dataset = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "digits": load_digits,
    }[dataset]()
    X, y = dataset.data, dataset.target
    train_X, test_X, train_y, test_y = train_test_split(
        X,
        y,
        train_size=0.7,
        random_state=RANDOM_STATE,
    )
    model = CARTClassifier()
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print("dataset : {}\nCART classifier's perf : {}".format(
        name, accuracy_score(test_y, pred)))
    if plot:
        tree_plot(
            model,
            filename="CART",
            feature_names=dataset.feature_names,
            target_names=dataset.target_names,
        )


def CARTregression_test(dataset="boston", plot=True):
    name = dataset
    dataset = {
        "boston": load_boston,
        "diabetes": load_diabetes,
    }[dataset]()
    X, y = dataset.data, dataset.target
    train_X, test_X, train_y, test_y = train_test_split(
        X,
        y,
        train_size=0.7,
        random_state=RANDOM_STATE,
    )
    model = CARTRegressor()
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print("dataset : {}\nCART regressor's perf : {}".format(
        name, mean_squared_error(test_y, pred)))
    if plot:
        tree_plot(
            model,
            filename="CART",
            feature_names=dataset.feature_names,
        )


def CARTregression_visual():
    X, y = make_regression(n_features=1, noise=10, random_state=RANDOM_STATE)
    y = y**2

    model = CARTRegressor()
    model.fit(X, y)

    test_x = np.linspace(X.min(), X.max(), 500)
    pred = model.predict(test_x)

    plt.scatter(X, y, label="dataset")
    plt.plot(test_x, pred, label="predict", color='orange')
    plt.legend()
    plt.show()


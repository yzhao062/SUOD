# -*- coding: utf-8 -*-
"""Cost predictor function for forecasting base model training and prediction
cost.
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: MIT
import sys
import os

import numpy as np
import pandas as pd
from joblib import dump

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels.

    Parameters
    ----------
    data : list
        The raw data.

    nb_classes : int
        The number of targeted classes.

    Returns
    -------

    """
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def build_cost_predictor(file_name, output_file, save_to_local=True):
    """Build cost predictor from the scratch. In general, this does not need
    to be used.

    Parameters
    ----------
    file_name : string
        The training table of algorithm performance.

    output_file
    save_to_local
    """

    # read in the data file，
    # WS = pd.read_excel(os.path.join('saved_models', file_name),
    #                    sheet_name='sheet1').drop(columns=['File'])
    #
    # WS_np = WS.to_numpy().astype('float')
    WS_np = np.loadtxt(file_name, delimiter=',')

    X = []
    y = []

    for i in range(WS_np.shape[0]):
        for j in range(4, 14):
            X.append([WS_np[i][0], WS_np[i][1], WS_np[i][2], j - 4])
            y.append(WS_np[i][j])

        X.append([WS_np[i][0], WS_np[i][1], WS_np[i][2], j - 3])
        y.append(np.mean(y[-10:]))

    X = np.asarray(X)

    nb_classes = 11
    data = X[:, 3].astype(int)

    # build embeddings
    w = indices_to_one_hot(data, nb_classes)
    # p = indices_to_one_hot([2,3], 11)

    X = np.concatenate((X[:, [0, 1, 2]], w), axis=1)
    # this is currently a hardcoded string
    indices = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    X = X[:, indices]

    r2 = []
    mse = []
    pearson = []
    spearman = []

    # fix for 10 fold random CV now
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X[:, ], y,
                                                            test_size=0.1)
        clf = RandomForestRegressor(n_estimators=100)

        clf.fit(X_train, y_train)
        test_pred = clf.predict(X_test)
        test_pred[test_pred < 0] = 0

        r2.append([r2_score(y_test, test_pred)])
        mse.append([mean_squared_error(y_test, test_pred)])
        pearson.append(pearsonr(y_test, test_pred)[0])
        spearman.append(spearmanr(y_test, test_pred)[0])

    # print('Spearman Rank', np.mean(spearman))

    clf.fit(X, y)

    if save_to_local:
        # save to the local
        dump(clf, os.path.join("saved_models", output_file))


if __name__ == "__main__":
    # this should be only executed if the pre-trained model is missing.
    build_cost_predictor(
        file_name=os.path.join('saved_models', 'summary_train.txt'),
        output_file="bps_train_curr.joblib",
        save_to_local=False)
    build_cost_predictor(
        file_name=os.path.join('saved_models', 'summary_prediction.txt'),
        output_file="bps_prediction_curr.joblib",
        save_to_local=False)

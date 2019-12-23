import sys
import os

import numpy as np
import pandas as pd
from joblib import dump, load

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
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


idx_clf_mapping = {
    1: 'ABOD',
    2: 'CBLOF',
    3: 'FeatureBagging',
    4: 'HBOS',
    5: 'IForest',
    6: 'KNN',
    7: 'LOF',
    8: 'MCD',
    9: 'OCSVM',
    10: 'PCA',
    11: 'UNK'
}

clf_idx_mapping = {
    'ABOD': 1,
    'CBLOF': 2,
    'FeatureBagging': 3,
    'HBOS': 4,
    'IForest': 5,
    'KNN': 6,
    'LOF': 7,
    'MCD': 8,
    'OCSVM': 9,
    'PCA': 10,
    'UNK': 11
}


def build_cost_predictor(file_name, output_file):
    # read in the data file
    WS = pd.read_excel(os.path.join('saved_models', file_name),
                       sheet_name='sheet1').drop(columns=['File'])

    WS_np = WS.to_numpy().astype('float')

    X1 = []
    y1 = []

    for i in range(WS_np.shape[0]):
        for j in range(4, 14):
            X1.append([WS_np[i][0], WS_np[i][1], WS_np[i][2], j - 4])
            y1.append(WS_np[i][j])

        X1.append([WS_np[i][0], WS_np[i][1], WS_np[i][2], j - 3])
        y1.append(np.mean(y1[-10:]))

    X1 = np.asarray(X1)

    # build the second dataset
    WS = pd.read_excel(os.path.join('saved_models', file_name),
                       sheet_name='sheet2').drop(columns=['File'])
    WS_np = WS.to_numpy().astype('float')

    X2 = []
    y2 = []
    for i in range(WS_np.shape[0]):
        for j in range(4, 14):
            X2.append([WS_np[i][0], WS_np[i][1], WS_np[i][2], j - 4])
            y2.append(WS_np[i][j])

        X2.append([WS_np[i][0], WS_np[i][1], WS_np[i][2], j - 3])
        y2.append(np.mean(y1[-10:]))

    X2 = np.asarray(X2)

    # combine both datasets
    X = np.concatenate([X1, X2])
    y = np.concatenate([y1, y2])

    nb_classes = 11
    data = X[:, 3].astype(int)

    # build embeddings
    w = indices_to_one_hot(data, nb_classes)
    # p = indices_to_one_hot([2,3], 11)

    X = np.concatenate((X[:, [0, 1, 2]], w), axis=1)
    indices = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    X = X[:, indices]

    clf_mapping = {
        1: 'ABOD',
        2: 'CBLOF',
        3: 'FeatureBagging',
        4: 'HBOS',
        5: 'IForest',
        6: 'KNN',
        7: 'LOF',
        8: 'MCD',
        9: 'OCSVM',
        10: 'PCA',
        11: 'UKN'
    }

    r2 = []
    mse = []
    pearson = []
    spearman = []

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

    print('Spearman Rank', np.mean(spearman))

    clf.fit(X, y)

    # save to the local
    dump(clf, os.path.join("saved_models", output_file))


if __name__ == "__main__":
    build_cost_predictor(file_name="summary_train.xlsx",
                         output_file="bps_train.joblib")
    build_cost_predictor(file_name="summary_prediction.xlsx",
                         output_file="bps_prediction.joblib")

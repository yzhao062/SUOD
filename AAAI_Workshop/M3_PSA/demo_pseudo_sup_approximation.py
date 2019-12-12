# %%
import os
import time
import numpy as np
import scipy as sp
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from pyod.models.iforest import IForest
from pyod.models.abod import ABOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.mcd import MCD
from sklearn.svm import SVR

from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import lightgbm as lgb

import warnings

warnings.filterwarnings("ignore")

# not all the mat files are included
mat_file_list = [
    # 'annthyroid.mat',
    # 'BreastW.mat',
    'cardio.mat',
    # 'glass.mat',
    # 'http_n.mat',
    # 'ionosphere.mat',
    # 'letter.mat',
    # 'lympho.mat',
    # 'mammography.mat',
    # 'mnist.mat',
    # 'musk.mat',
    # 'optdigits.mat',
    # 'pendigits.mat',
    # 'pima.mat',
    'satellite.mat',
    'satimage-2.mat',
    # 'seismic.mat',
    # 'shuttle.mat',
    # 'smtp_n.mat',
    # 'speech.mat',
    # 'vertebral.mat',
    # 'vowels.mat',
    # 'thyroid.mat',
    # 'wbc.mat',
    # 'wine.mat'
]

n_iter = 10

# for mat_file in mat_file_list:
mat_file = mat_file_list[0]
mat_file_name = mat_file.replace('.mat', '')
print("\n... Processing", mat_file_name, '...')
mat = sp.io.loadmat(os.path.join('../datasets', mat_file))

X = mat['X']
y = mat['y'].ravel()
outliers_fraction = np.sum(y) / len(y)
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

classifiers = {
    'Angle-based Outlier Detector (ABOD)': ABOD(n_neighbors=10,
                                                contamination=outliers_fraction),
    'Cluster-based Local Outlier Factor (CBLOF)':
        CBLOF(contamination=outliers_fraction, check_estimator=False),
    'Feature Bagging': FeatureBagging(LOF(), contamination=outliers_fraction),
    'Histogram-base Outlier Detection (HBOS)': HBOS(
        contamination=outliers_fraction),
    'Isolation Forest': IForest(contamination=outliers_fraction),
    'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
    'Average KNN': KNN(method='mean', contamination=outliers_fraction),
    'Local Outlier Factor (LOF)': LOF(contamination=outliers_fraction),
    'Minimum Covariance Determinant (MCD)': MCD(
        contamination=outliers_fraction),
    'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
    'Principal Component Analysis (PCA)': PCA(contamination=outliers_fraction)
}

stat_mat_all = np.zeros([len(classifiers), 10])
report_list = ['train_roc_orig', 'train_p@n_orig', 'train_roc_kd',
               'train_p@n_kd', 'test_time_orig', 'test_roc_orig',
               'test_p@n_orig', 'test_time_kd', 'test_roc_kd', 'test_p@n_kd']

classifier_names = ['ABOD', 'CBLOF', 'FB', 'HBOS', 'IF', 'KNN', 'AKNN', 'LOF',
                    'MCD', 'OCSVM', 'PCA']

for j in range(n_iter):
    stat_mat = np.zeros([len(classifiers), 10])

    for i, (clf_name, clf) in enumerate(classifiers.items()):
        ################## original version
        clf.fit(X_train)
        pseudo_labels = clf.decision_scores_
        # replace nan by mean
        np_mean = np.nanmean(pseudo_labels)
        pseudo_labels[np.isnan(pseudo_labels)] = np_mean

        print('Iter', j + 1, i + 1, clf_name, '|', 'train stat',
              np.round(roc_auc_score(y_train, pseudo_labels), decimals=4), '|',
              np.round(precision_n_scores(y_train, pseudo_labels), decimals=4))

        stat_mat[i, 0] = np.round(roc_auc_score(y_train, pseudo_labels),
                                  decimals=4)
        stat_mat[i, 1] = np.round(precision_n_scores(y_train, pseudo_labels),
                                  decimals=4)

        ################## xgb train scores

        regressor = lgb.LGBMRegressor()
        regressor.fit(X_train, pseudo_labels)
        pseudo_scores = regressor.predict(X_train)
        print('Iter', j + 1, i + 1, 'kd', clf_name, '|', 'train stat',
              np.round(roc_auc_score(y_train, pseudo_scores), decimals=4), '|',
              np.round(precision_n_scores(y_train, pseudo_scores), decimals=4))

        stat_mat[i, 2] = np.round(roc_auc_score(y_train, pseudo_scores),
                                  decimals=4)
        stat_mat[i, 3] = np.round(precision_n_scores(y_train, pseudo_scores),
                                  decimals=4)

        ################## original test time, roc, prn
        start = time.time()
        y_predict = clf.decision_function(X_test)
        end = time.time()

        # replace nan by mean
        np_mean = np.nanmean(y_predict)
        y_predict[np.isnan(y_predict)] = np_mean

        print('Iter', j + 1, i + 1, clf_name,
              np.round(end - start, decimals=4), '|',
              np.round(roc_auc_score(y_test, y_predict), decimals=4), '|',
              np.round(precision_n_scores(y_test, y_predict), decimals=4))

        stat_mat[i, 4] = np.round(end - start, decimals=4)
        stat_mat[i, 5] = np.round(roc_auc_score(y_test, y_predict), decimals=4)
        stat_mat[i, 6] = np.round(precision_n_scores(y_test, y_predict),
                                  decimals=4)

        ################## original test time, roc, prn
        start = time.time()
        y_predict_xgb = regressor.predict(X_test)
        end = time.time()
        print('Iter', j + 1, i + 1, 'kd', clf_name,
              np.round(end - start, decimals=4), '|',
              np.round(roc_auc_score(y_test, y_predict_xgb), decimals=4), '|',
              np.round(precision_n_scores(y_test, y_predict_xgb), decimals=4))
        stat_mat[i, 7] = np.round(end - start, decimals=4)
        stat_mat[i, 8] = np.round(roc_auc_score(y_test, y_predict_xgb),
                                  decimals=4)
        stat_mat[i, 9] = np.round(precision_n_scores(y_test, y_predict_xgb),
                                  decimals=4)

        print()

    stat_mat_all = stat_mat_all + stat_mat

stat_mat_all = stat_mat_all / n_iter

roc_summary = pd.DataFrame(stat_mat_all, columns=report_list)
roc_summary['clf'] = classifier_names
print(roc_summary)
roc_summary.to_csv(mat_file_name + '_XGB_' + '.csv', index=False)

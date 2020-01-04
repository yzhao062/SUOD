# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:09:18 2019

@author: yuezh
"""
# %%
import numpy as np
import pandas as pd
import sys
import warnings
import scipy as sp
import os
from sklearn.preprocessing import StandardScaler
import time

if not sys.warnoptions:
    warnings.simplefilter("ignore")

WS = pd.read_excel(os.path.join('saved_models', 'summary.xlsx'),
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

WS = pd.read_excel(os.path.join('saved_models', 'summary.xlsx'),
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

X = np.concatenate([X1, X2])
y = np.concatenate([y1, y2])

nb_classes = 11
data = X[:, 3].astype(int)


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


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

# X = X[:, [0,1,3]]

# X = StandardScaler().fit_transform(X)

# %%
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

r2 = []
mse = []
pearson = []
spearman = []

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X[:, ], y,
                                                        test_size=0.1)
    clf = RandomForestRegressor()
    #    clf = xgb.XGBRegressor(n_estimators=200, max_depth=5)
    #    clf = LinearRegression()
    #    clf = DecisionTreeRegressor(max_depth=20)

    clf.fit(X_train, y_train)
    test_pred = clf.predict(X_test)
    test_pred[test_pred < 0] = 0

    r2.append([r2_score(y_test, test_pred)])
    mse.append([mean_squared_error(y_test, test_pred)])
    pearson.append(pearsonr(y_test, test_pred)[0])
    spearman.append(spearmanr(y_test, test_pred)[0])

print(np.mean(r2), np.mean(mse), np.mean(pearson), np.mean(pearson),
      np.mean(spearman))

clf.fit(X, y)

from joblib import dump, load

dump(clf, os.path.join('saved_models', 'rf_predictor.joblib'))

# %%
from joblib import effective_n_jobs
from joblib import Parallel, delayed
from copy import deepcopy


def _parallel_build_estimators(n_estimators, clfs, X, total_n_estimators,
                               verbose):
    # Build estimators
    estimators = []
    for i in range(n_estimators):
        estimator = deepcopy(clfs[i])
        if verbose > 1:
            print("Building estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))

        estimator.fit(X)
        estimators.append(estimator)

    return estimators


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs,
                                   dtype=np.int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


# %%

from joblib import load
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
from pyod.models.lscp import LSCP

from scipy.stats import rankdata
import arff

# def read_arff(file_path, misplaced_list):
#    misplaced = False
#    for item in misplaced_list:
#        if item in file_path:
#            misplaced = True
#
#    file = arff.load(open(file_path))
#    data_value = np.asarray(file['data'])
#    attributes = file['attributes']
#
#    X = data_value[:, 0:-2]
#    if not misplaced:
#        y = data_value[:, -1]
#    else:
#        y = data_value[:, -2]
#    y[y == 'no'] = 0
#    y[y == 'yes'] = 1
#    y = y.astype('float').astype('int').ravel()
#
#    if y.sum() > len(y):
#        print(attributes)
#        raise ValueError('wrong sum')
#
#    return X, y, attributes
#
#
# misplaced_list = ['Arrhythmia', 'Cardiotocography', 'Hepatitis', 'ALOI',
#                  'KDDCup99']
#
# arff_file = os.path.join('../semantic', 'SpamBase', 'SpamBase_withoutdupl_40.arff')
# X, y, attributes = read_arff(arff_file, misplaced_list)

mat_file = 'cardio.mat'
mat_file_name = mat_file.replace('.mat', '')
print("\n... Processing", mat_file_name, '...')
mat = sp.io.loadmat(os.path.join('../datasets', mat_file))

X = mat['X']
y = mat['y']

X = StandardScaler().fit_transform(X)

clf = load('rf_predictor.joblib')

classifiers = {
    1: ABOD(n_neighbors=10),
    2: CBLOF(check_estimator=False),
    3: FeatureBagging(LOF()),
    4: HBOS(),
    5: IForest(),
    6: KNN(),
    7: KNN(method='mean'),
    8: LOF(),
    9: MCD(),
    10: OCSVM(),
    11: PCA(),
}

n_estimators_total = 100
clfs = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        size=n_estimators_total)
clfs_real = []

for estimator in clfs:
    clfs_real.append(classifiers[estimator])
X_w = indices_to_one_hot(clfs - 1, 11)
X_d1 = np.array([X.shape[0], X.shape[1]]).reshape(1, 2)
X_d = np.repeat(X_d1, len(clfs), axis=0)

X_c = np.concatenate((X_d, X_w), axis=1)
predict_time = clf.predict(X_c)

n_jobs = 6
n_estimators_list = []
ranks = rankdata(predict_time)
rank_sum = np.sum(ranks)
chunk_sum = rank_sum / n_jobs

# %%
starts = [0]
index_track = 0
sum_check = []

for i in range(len(ranks) + 1):
    if np.sum(ranks[starts[index_track]:i]) >= chunk_sum:
        starts.append(i)
        index_track += 1
starts.append(len(ranks))

for j in range(n_jobs):
    sum_check.append(np.sum(ranks[starts[j]:starts[j + 1]]))
    print(sum_check[j])
    n_estimators_list.append(starts[j + 1] - starts[j])

assert (np.sum(n_estimators_list) == n_estimators_total)
assert (rank_sum == np.sum(sum_check))

n_jobs = min(effective_n_jobs(n_jobs), n_estimators_total)
total_n_estimators = total_n_estimators = sum(n_estimators_list)
print(starts)

start = time.time()
all_results = Parallel(n_jobs=n_jobs, verbose=True)(
    delayed(_parallel_build_estimators)(
        n_estimators_list[i],
        clfs_real[starts[i]:starts[i + 1]],
        X,
        total_n_estimators,
        verbose=True)
    for i in range(n_jobs))
print('total time', time.time() - start)
time.sleep(10)

# %%
# print()
#
# n_jobs, n_estimators, starts = _partition_estimators(len(clfs), n_jobs=n_jobs)
# total_n_estimators = sum(n_estimators)
#
# start = time.time()
# all_results = Parallel(n_jobs=n_jobs, verbose=True)(
#    delayed(_parallel_build_estimators)(
#        n_estimators[i],
#        clfs_real[starts[i]:starts[i + 1]], 
#        X,
#        total_n_estimators,
#        verbose=True)
#    for i in range(n_jobs))
# print('total time', time.time()-start)
# time.sleep(10)
# %%
print()
clfs = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        size=n_estimators_total)
clfs = np.sort(clfs)
clfs_real = []

for estimator in clfs:
    clfs_real.append(classifiers[estimator])

n_jobs, n_estimators, starts = _partition_estimators(len(clfs), n_jobs=n_jobs)
total_n_estimators = sum(n_estimators)

start = time.time()
all_results = Parallel(n_jobs=n_jobs, verbose=True)(
    delayed(_parallel_build_estimators)(
        n_estimators[i],
        clfs_real[starts[i]:starts[i + 1]],
        X,
        total_n_estimators,
        verbose=True)
    for i in range(n_jobs))
print('total time', time.time() - start)

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 18:04:41 2020

@author: yuezh
"""
import os
import sys
import time
import warnings
import numpy as np
import scipy as sp
from scipy.stats import rankdata
from sklearn.base import clone
import joblib

import numpy as np
from scipy.stats import rankdata
from joblib import effective_n_jobs
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from sklearn.preprocessing import StandardScaler
from joblib import effective_n_jobs
from joblib import Parallel, delayed
from copy import deepcopy
import arff

from pyod.utils.utility import score_to_label
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

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def read_arff(file_path, misplaced_list):
    misplaced = False
    for item in misplaced_list:
        if item in file_path:
            misplaced = True

    file = arff.load(open(file_path))
    data_value = np.asarray(file['data'])
    attributes = file['attributes']

    X = data_value[:, 0:-2]
    if not misplaced:
        y = data_value[:, -1]
    else:
        y = data_value[:, -2]
    y[y == 'no'] = 0
    y[y == 'yes'] = 1
    y = y.astype('float').astype('int').ravel()

    if y.sum() > len(y):
        print(attributes)
        raise ValueError('wrong sum')

    return X, y, attributes


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def balanced_scheduling(time_cost_pred, n_estimators, n_jobs):
    """Conduct balanced scheduling based on the sum of rank, for both train
    and prediction. The algorithm will enforce the equal sum of ranks among
    workers.

    Parameters
    ----------
    time_cost_pred
    n_estimators
    n_jobs

    Returns
    -------

    """
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)
    # Conduct Balanced Task Scheduling
    n_estimators_list = []  # track the number of estimators for each worker
    ranks = rankdata(time_cost_pred)
    ##########################################
    # #todo: the fastest is at most 2 times costly than the slowest
    ranks = 1 + ranks / n_estimators
    ##########################################
    rank_sum = np.sum(ranks)
    chunk_sum = rank_sum / n_jobs

    starts_orig = [0]
    index_track = 0
    sum_check = []

    for i in range(len(ranks) + 1):
        if np.sum(ranks[starts_orig[index_track]:i]) >= chunk_sum:
            starts_orig.append(i)
            index_track += 1
    starts_orig.append(len(ranks))

    starts = starts_orig

    # # offset for the last worker's load
    # starts = [0]
    # for k in range(1, n_jobs+1):
    #     starts.append(starts_orig[k]-np.random.randint(low=1, high=k+1))

    # print(starts)
    # starts[-1] = n_estimators
    # print(starts)

    for j in range(n_jobs):
        sum_check.append(np.sum(ranks[starts[j]:starts[j + 1]]))
        print('Worker', j + 1, 'sum of ranks:', sum_check[j])
        n_estimators_list.append(starts[j + 1] - starts[j])

    print()

    # Confirm the length of the estimators is consistent
    assert (np.sum(n_estimators_list) == n_estimators)
    assert (np.abs(rank_sum - np.sum(sum_check)) < 0.1)

    xdiff = [starts[n] - starts[n - 1] for n in range(1, len(starts))]

    print("Split among workers BPS:", starts, xdiff)
    return n_estimators_list, starts, n_jobs


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs,
                                   dtype=np.int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    xdiff = [starts[n] - starts[n - 1] for n in range(1, len(starts))]

    print("Split among workers default:", starts, xdiff)
    return n_estimators_per_job.tolist(), [0] + starts.tolist(), n_jobs


def _parallel_fit(n_estimators, clfs, X, total_n_estimators, verbose):
    X = check_array(X)
    # Build estimators
    estimators = []
    rp_transformers = []
    for i in range(n_estimators):
        estimator = clone(clfs[i])
        if verbose > 1:
            print("Building estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))

        estimator.fit(X)
        estimators.append(estimator)
    return estimators


def cost_forecast_meta(clf, X, base_estimator_names):
    # convert base estimators to the digestible form
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
        'UNK': 11}

    clf_idx = np.asarray(
        list(map(clf_idx_mapping.get, base_estimator_names)))

    X_detector_code = indices_to_one_hot(clf_idx - 1, 11)
    X_shape_code_s = np.array([X.shape[0], X.shape[1]]).reshape(1, 2)
    X_shape_code = np.repeat(X_shape_code_s, len(base_estimator_names), axis=0)
    X_code = np.concatenate((X_shape_code, X_detector_code), axis=1)
    time_cost_pred = clf.predict(X_code)

    return time_cost_pred


if __name__ == "__main__":
    #############################################################################
    misplaced_list = ['Arrhythmia', 'Cardiotocography', 'Hepatitis', 'ALOI',
                      'KDDCup99']
    arff_list = [
        # os.path.join('../semantic', 'Annthyroid', 'Annthyroid_withoutdupl_07.arff'),
        # os.path.join('../semantic', 'Arrhythmia', 'Arrhythmia_withoutdupl_46.arff'),
        # os.path.join('../semantic', 'Cardiotocography', 'Cardiotocography_withoutdupl_22.arff'),
        # os.path.join('../semantic', 'HeartDisease', 'HeartDisease_withoutdupl_44.arff'),
        # os.path.join('../semantic', 'Hepatitis', 'Hepatitis_withoutdupl_16.arff'),
        # os.path.join('../semantic', 'InternetAds', 'InternetAds_withoutdupl_norm_19.arff'),
        # os.path.join('../semantic', 'PageBlocks', 'PageBlocks_withoutdupl_09.arff'),
        # os.path.join('../semantic', 'Parkinson', 'Parkinson_withoutdupl_75.arff'),
        # os.path.join('../semantic', 'Pima', 'Pima_withoutdupl_35.arff'),
        # os.path.join('semantic', 'SpamBase', 'SpamBase_withoutdupl_40.arff'),
        # os.path.join('../semantic', 'Stamps', 'Stamps_withoutdupl_09.arff'),
        # os.path.join('../semantic', 'Wilt', 'Wilt_withoutdupl_05.arff'),
        #    #
        # os.path.join('../literature', 'ALOI', 'ALOI_withoutdupl.arff'),
        # os.path.join('../literature', 'Glass', 'Glass_withoutdupl_norm.arff'),
        # os.path.join('literature', 'Ionosphere', 'Ionosphere_withoutdupl_norm.arff'),
        # os.path.join('../literature', 'KDDCup99', 'KDDCup99_original.arff'),
        # os.path.join('../literature', 'Lymphography', 'Lymphography_original.arff'),
        # os.path.join('../literature', 'PenDigits', 'PenDigits_withoutdupl_norm_v01.arff'),
        # os.path.join('../literature', 'Shuttle', 'Shuttle_withoutdupl_v01.arff'),
        # os.path.join('../literature', 'Waveform', 'Waveform_withoutdupl_v01.arff'),
        # os.path.join('../literature', 'WBC', 'WBC_withoutdupl_v01.arff'),
        # os.path.join('../literature', 'WDBC', 'WDBC_withoutdupl_v01.arff'),
        # os.path.join('../literature', 'WPBC', 'WPBC_withoutdupl_norm.arff')
    ]

    file_names = [
        # 'Annthyroid',
        # 'Arrhythmia',
        # 'Cardiotocography',
        # 'HeartDisease',  # too small
        # 'Hepatitis',  # too small
        # 'InternetAds',
        # 'PageBlocks',
        # 'Parkinson',  # too small
        # 'Pima',
        # 'SpamBase',
        # 'Stamps',
        # 'Wilt',
        #    #
        # 'ALOI', # too large
        # 'Glass', # too small
        # 'Ionosphere',
        # 'KDDCup99', # too large
        # 'Lymphography', # data type X contains categorical
        # 'PenDigits',
        # 'Shuttle',
        # 'Waveform',
        # 'WBC', # too small
        # 'WDBC', # too small
        # 'WPBC', # too small
    ]

    assert (len(arff_list) == len(file_names))

    # arff_file = arff_list[0]
    # arff_file_name = file_names[0]
    # X, y, attributes = read_arff(arff_file, misplaced_list)

    n_jobs = 4
    n_estimators_total = 500

    mat_file = 'pendigits.mat'
    mat_file_name = mat_file.replace('.mat', '')
    print("\n... Processing", mat_file_name, '...')
    mat = sp.io.loadmat(os.path.join('../datasets', mat_file))

    X = mat['X']
    y = mat['y']

    X = StandardScaler().fit_transform(X)

    classifiers = {
        1: ABOD(n_neighbors=10),
        2: CBLOF(check_estimator=False),
        3: FeatureBagging(LOF()),
        4: HBOS(),
        5: IForest(),
        6: KNN(),
        7: LOF(),
        8: MCD(),
        9: OCSVM(),
        10: PCA(),
    }

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
    }

    clfs = np.random.choice([1, 3, 4, 5, 6, 7, 8, 9, 10],
                            size=n_estimators_total)

    clfs = np.sort(clfs)

    base_estimators = []
    base_estimator_names = []

    for i in clfs:
        estimator = classifiers[i]
        base_estimators.append(estimator)
        base_estimator_names.append(idx_clf_mapping[i])

    this_directory = os.path.abspath(os.path.dirname(__file__))
    cost_forecast_loc_fit_ = os.path.join(
        this_directory, 'saved_models', 'bps_train.joblib')

    cost_predictor = joblib.load(cost_forecast_loc_fit_)

    #############################################
    start = time.time()
    time_cost_pred = cost_forecast_meta(cost_predictor, X,
                                        base_estimator_names)

    n_estimators_list, starts, n_jobs = balanced_scheduling(
        time_cost_pred, n_estimators_total, n_jobs)

    # TODO: code cleanup. There is an existing bug for joblib on Windows:
    # https://github.com/joblib/joblib/issues/806
    # max_nbytes can be dropped on other OS
    all_results = Parallel(n_jobs=n_jobs, max_nbytes=None, batch_size=1,
                           verbose=True)(
        delayed(_parallel_fit)(
            n_estimators_list[i],
            base_estimators[starts[i]:starts[i + 1]],
            X,
            n_estimators_total,
            verbose=True)
        for i in range(n_jobs))

    BPS = time.time() - start

    #############################################
    start = time.time()
    n_estimators_list, starts, n_jobs = _partition_estimators(
        n_estimators_total, n_jobs)
    xdiff = [starts[n] - starts[n - 1] for n in range(1, len(starts))]
    print(starts, xdiff)
    all_results = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=True)(
        delayed(_parallel_fit)(
            n_estimators_list[i],
            base_estimators[starts[i]:starts[i + 1]],
            X,
            n_estimators_total,
            verbose=True)
        for i in range(n_jobs))

    NS = time.time() - start

    ###########################################################################
    start = time.time()
    n_estimators_list, starts, n_jobs = _partition_estimators(
        n_estimators_total, n_jobs)
    xdiff = [starts[n] - starts[n - 1] for n in range(1, len(starts))]
    print(starts, xdiff)
    all_results = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=True,
                           batch_size=1)(
        delayed(_parallel_fit)(
            n_estimators_list[i],
            base_estimators[starts[i]:starts[i + 1]],
            X,
            n_estimators_total,
            verbose=True)
        for i in range(n_jobs))

    BS = time.time() - start

    print('Balanced Scheduling Total Train Time:', BPS)
    print('Batch Sampling Train Time:', BS)
    print('Naive Split Total Time', NS)

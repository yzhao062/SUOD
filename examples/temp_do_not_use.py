import os
import sys
import time
from copy import deepcopy

import numpy as np
import scipy as sp

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor


from joblib import Parallel, delayed

from pyod.utils.utility import standardizer
from pyod.utils.data import evaluate_print

from combo.models.score_comb import majority_vote, maximization, average
from combo.models.score_comb import aom, moa

# temporary solution for relative imports in case combo is not installed
# if combo is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from suod.models.base import SUOD
from suod.models.parallel_processes import _partition_estimators
from suod.models.parallel_processes import cost_forecast_meta
from suod.models.parallel_processes import _parallel_fit
from suod.models.parallel_processes import _parallel_predict
from suod.models.parallel_processes import _parallel_decision_function
from suod.models.parallel_processes import _partition_estimators
from suod.models.parallel_processes import _parallel_approx_estimators
from suod.models.parallel_processes import balanced_scheduling
from suod.models.utils.utility import _unfold_parallel
from suod.models.utils.utility import get_estimators

# suppress warnings
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    n_jobs = 20
    # load files
    mat_file_list = [
        # 'cardio.mat',
        # 'satellite.mat',
        # 'satimage-2.mat',
        'mnist.mat',
    ]

    mat_file = mat_file_list[0]
    mat_file_name = mat_file.replace('.mat', '')
    print("\n... Processing", mat_file_name, '...')
    mat = sp.io.loadmat(os.path.join('', 'datasets', mat_file))

    X = mat['X']
    y = mat['y']


    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.4, random_state=42)
        
    # standardize data to be digestible for most algorithms
    X_train, X_test = standardizer(X_train, X_test)

    contamination = y.sum() / len(y)

    base_estimators = deepcopy(get_estimators(contamination=contamination))

    ##########################################################################
    model = SUOD(base_estimators=base_estimators, rp_flag_global=True, 
                 approx_clf=RandomForestRegressor(),
                 n_jobs=n_jobs, bps_flag=True, contamination=contamination,
                 approx_flag_global=True)

    start = time.time()
    model.fit(X_train)  # fit all models with X
    print('Fit time:', time.time() - start)
    print()

    start = time.time()
    model.approximate(X_train)  # conduct model approximation if it is enabled
    print('Approximation time:', time.time() - start)
    print()

    start = time.time()
    predicted_labels = model.predict(X_test)  # predict labels
    print('Predict time:', time.time() - start)
    print()

    start = time.time()
    predicted_scores = model.decision_function(X_test)  # predict scores
    print('Decision Function time:', time.time() - start)
    print()

    ##########################################################################
    # compare with no projection, no bps, and no approximation
    print("******************************************************************")
    start = time.time()
    n_estimators = len(base_estimators)
    n_estimators_list, starts, n_jobs = _partition_estimators(n_estimators,
                                                              n_jobs)

    rp_flags = np.zeros([n_estimators, 1])
    approx_flags = np.zeros([n_estimators, 1])
    objective_dim = None
    rp_method = None

    all_results = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=True)(
        delayed(_parallel_fit)(
            n_estimators_list[i],
            base_estimators[starts[i]:starts[i + 1]],
            X_train,
            n_estimators,
            rp_flags[starts[i]:starts[i + 1]],
            objective_dim,
            rp_method=rp_method,
            verbose=True)
        for i in range(n_jobs))

    print('Orig Fit time:', time.time() - start)
    print()

    all_results = list(map(list, zip(*all_results)))
    trained_estimators = _unfold_parallel(all_results[0], n_jobs)
    jl_transformers = _unfold_parallel(all_results[1], n_jobs)

    ##########################################################################
    start = time.time()
    n_estimators = len(base_estimators)
    n_estimators_list, starts, n_jobs = _partition_estimators(n_estimators,
                                                              n_jobs)
    # model prediction
    all_results_pred = Parallel(n_jobs=n_jobs, max_nbytes=None,
                                verbose=True)(
        delayed(_parallel_predict)(
            n_estimators_list[i],
            trained_estimators[starts[i]:starts[i + 1]],
            None,
            X_test,
            n_estimators,
            # rp_flags[starts[i]:starts[i + 1]],
            jl_transformers,
            approx_flags[starts[i]:starts[i + 1]],
            contamination,
            verbose=True)
        for i in range(n_jobs))

    print('Orig Predict time:', time.time() - start)
    print()

    # unfold and generate the label matrix
    predicted_labels_orig = np.zeros([X_test.shape[0], n_estimators])
    for i in range(n_jobs):
        predicted_labels_orig[:, starts[i]:starts[i + 1]] = np.asarray(
            all_results_pred[i]).T

    start = time.time()
    n_estimators = len(base_estimators)
    n_estimators_list, starts, n_jobs = _partition_estimators(n_estimators,
                                                              n_jobs)
    # model prediction
    all_results_scores = Parallel(n_jobs=n_jobs, max_nbytes=None,
                                  verbose=True)(
        delayed(_parallel_decision_function)(
            n_estimators_list[i],
            trained_estimators[starts[i]:starts[i + 1]],
            None,
            X_test,
            n_estimators,
            # rp_flags[starts[i]:starts[i + 1]],
            jl_transformers,
            approx_flags[starts[i]:starts[i + 1]],
            verbose=True)
        for i in range(n_jobs))

    print('Orig decision_function time:', time.time() - start)
    print()

    # unfold and generate the label matrix
    predicted_scores_orig = np.zeros([X_test.shape[0], n_estimators])
    for i in range(n_jobs):
        predicted_scores_orig[:, starts[i]:starts[i + 1]] = np.asarray(
            all_results_scores[i]).T
    ##########################################################################
    predicted_scores = standardizer(predicted_scores)
    predicted_scores_orig = standardizer(predicted_scores_orig)

    evaluate_print('orig', y_test, np.mean(predicted_scores_orig, axis=1))
    evaluate_print('new', y_test, np.mean(predicted_scores, axis=1))

    evaluate_print('orig max', y_test, np.max(predicted_scores_orig, axis=1))
    evaluate_print('new max', y_test, np.max(predicted_scores, axis=1))
    
    evaluate_print('orig aom', y_test, aom(predicted_scores_orig))
    evaluate_print('new aom', y_test, aom(predicted_scores))

    evaluate_print('orig moa', y_test, moa(predicted_scores_orig))
    evaluate_print('new moa', y_test, moa(predicted_scores))

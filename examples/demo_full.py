# %%
import os
import sys
import time
import numpy as np
import scipy as sp

import joblib
from joblib import Parallel, delayed

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.lscp import LSCP
from pyod.utils.data import evaluate_print

from combo.models.score_comb import majority_vote, maximization, average

# temporary solution for relative imports in case combo is not installed
# if combo is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from suod.models.base import build_codes
from suod.models.parallel_processes import cost_forecast_train
from suod.models.parallel_processes import _parallel_train
from suod.models.parallel_processes import _parallel_predict
from suod.models.parallel_processes import _parallel_decision_function
from suod.models.parallel_processes import _partition_estimators
from suod.models.parallel_processes import _parallel_approx_estimators
from suod.models.parallel_processes import balanced_scheduling
from suod.models.utils.utility import _unfold_parallel

# suppress warnings
import warnings

warnings.filterwarnings("ignore")

###############################################################################
# load files
mat_file_list = [
    'cardio.mat',
    # 'satellite.mat',
    # 'satimage-2.mat',
    # 'mnist.mat',
]

mat_file = mat_file_list[0]
mat_file_name = mat_file.replace('.mat', '')
print("\n... Processing", mat_file_name, '...')
mat = sp.io.loadmat(os.path.join('', 'datasets', mat_file))

X = mat['X']
y = mat['y']

# standardize data to be digestible for most algorithms
X = StandardScaler().fit_transform(X)

##############################################################################
# initialize a set of anomaly detectors
base_estimators = [
    LOF(n_neighbors=5), LOF(n_neighbors=15),
    LOF(n_neighbors=25), LOF(n_neighbors=35),
    LOF(n_neighbors=45),
    HBOS(),
    PCA(),
    OCSVM(),
    KNN(n_neighbors=5), KNN(n_neighbors=15),
    KNN(n_neighbors=25), KNN(n_neighbors=35),
    KNN(n_neighbors=45),
    IForest(n_estimators=50),
    IForest(n_estimators=100),
    LOF(n_neighbors=5), LOF(n_neighbors=15),
    LOF(n_neighbors=25), LOF(n_neighbors=35),
    LOF(n_neighbors=45),
    HBOS(),
    PCA(),
    OCSVM(),
    KNN(n_neighbors=5), KNN(n_neighbors=15),
    KNN(n_neighbors=25), KNN(n_neighbors=35),
    KNN(n_neighbors=45),
    IForest(n_estimators=50),
    IForest(n_estimators=100),
    LOF(n_neighbors=5), LOF(n_neighbors=15),
    LOF(n_neighbors=25), LOF(n_neighbors=35),
    LOF(n_neighbors=45),
    HBOS(),
    PCA(),
    OCSVM(),
    KNN(n_neighbors=5), KNN(n_neighbors=15),
    KNN(n_neighbors=25), KNN(n_neighbors=35),
    KNN(n_neighbors=45),
    IForest(n_estimators=50),
    IForest(n_estimators=100),
    LSCP(detector_list=[LOF(), LOF()])
]

# number of the parallel jobs
n_jobs = 6
n_estimators = len(base_estimators)

# the algorithms that should be be using random projection
rp_clf_list = ['LOF', 'KNN', 'ABOD']
# the algorithms that should NOT use random projection
rp_ng_clf_list = ['IForest', 'PCA', 'HBOS']
# global flag for random projection
rp_flag_global = True
objective_dim = 6
rp_method = 'discrete'

# build flags for random projection
rp_flags, base_estimator_names = build_codes(n_estimators, base_estimators,
                                             rp_clf_list, rp_ng_clf_list,
                                             rp_flag_global)

# load the pre-trained cost predictor to forecast the train cost
clf_train = joblib.load(
    os.path.join('../suod', 'models', 'saved_models', 'bps_train.joblib'))

time_cost_pred = cost_forecast_train(clf_train, X, base_estimator_names)

# schedule the tasks
n_estimators_list, starts, n_jobs = balanced_scheduling(time_cost_pred,
                                                        n_estimators, n_jobs)

print(starts)  # this is the list of being split
start = time.time()

print('Parallel Training...')

# TODO: code cleanup. There is an existing bug for joblib on Windows:
# https://github.com/joblib/joblib/issues/806
# max_nbytes can be dropped on other OS
all_results = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=True)(
    delayed(_parallel_train)(
        n_estimators_list[i],
        base_estimators[starts[i]:starts[i + 1]],
        X,
        n_estimators,
        rp_flags[starts[i]:starts[i + 1]],
        objective_dim,
        rp_method=rp_method,
        verbose=True)
    for i in range(n_jobs))

print('Balanced Scheduling Total Train Time:', time.time() - start)

# reformat and unfold the lists. Save the trained estimators and transformers
all_results = list(map(list, zip(*all_results)))
trained_estimators = _unfold_parallel(all_results[0], n_jobs)
jl_transformers = _unfold_parallel(all_results[1], n_jobs)

###############################################################################
# %% Model Approximation
approx_clf_list = ['LOF', 'KNN', 'ABOD']
approx_ng_clf_list = ['IForest', 'PCA', 'HBOS', 'ABOD']
approx_flag_global = True

# build approx code
# this can be a pre-defined list and directly supply to the system
approx_flags = np.zeros([n_estimators, 1], dtype=int)

# this can be supplied by the user
approx_clf = RandomForestRegressor(n_estimators=100)

approx_flags, base_estimator_names = build_codes(n_estimators, base_estimators,
                                                 approx_clf_list,
                                                 approx_ng_clf_list,
                                                 approx_flag_global)

n_jobs, n_estimators_list, starts = _partition_estimators(n_estimators,
                                                          n_jobs=n_jobs)
print(starts)  # this is the list of being split
start = time.time()

all_approx_results = Parallel(n_jobs=n_jobs, verbose=True)(
    delayed(_parallel_approx_estimators)(
        n_estimators_list[i],
        trained_estimators[starts[i]:starts[i + 1]],
        X,  # if it is a PyOD model, we do not need this
        n_estimators,
        approx_flags,
        approx_clf,
        verbose=True)
    for i in range(n_jobs))

print('Balanced Scheduling Total Test Time:', time.time() - start)

approximators = _unfold_parallel(all_approx_results, n_jobs)

# %% Second BPS for prediction
###############################################################################
# still build the rank sum by BPS
# load the pre-trained cost predictor to forecast the prediction cost
clf_prediction = joblib.load(
    os.path.join('../suod', 'models', 'saved_models', 'bps_prediction.joblib'))

time_cost_pred = cost_forecast_train(clf_prediction, X, base_estimator_names)

# TODO: add a second-stage tuner for prediction stage

n_estimators_list, starts, n_jobs = balanced_scheduling(time_cost_pred,
                                                        n_estimators, n_jobs)

print('Parallel Label Predicting without Approximators...')

all_results_pred = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=True)(
    delayed(_parallel_predict)(
        n_estimators_list[i],
        trained_estimators[starts[i]:starts[i + 1]],
        X,
        n_estimators,
        rp_flags[starts[i]:starts[i + 1]],
        jl_transformers[starts[i]:starts[i + 1]],
        verbose=True)
    for i in range(n_jobs))

# unfold and generate the label matrix
predicted_labels = np.zeros([X.shape[0], n_estimators])
for i in range(n_jobs):
    predicted_labels[:, starts[i]:starts[i + 1]] = np.asarray(
        all_results_pred[i]).T

print('Parallel Score Predicting without Approximators...')

all_results_scores = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=True)(
    delayed(_parallel_decision_function)(
        n_estimators_list[i],
        trained_estimators[starts[i]:starts[i + 1]],
        X,
        n_estimators,
        rp_flags[starts[i]:starts[i + 1]],
        jl_transformers[starts[i]:starts[i + 1]],
        verbose=True)
    for i in range(n_jobs))

# unfold and generate the label matrix
predicted_scores = np.zeros([X.shape[0], n_estimators])
for i in range(n_jobs):
    predicted_scores[:, starts[i]:starts[i + 1]] = np.asarray(
        all_results_scores[i]).T

# %% Check point to see whether it is working
###############################################################################
evaluate_print('majority vote', y, majority_vote(predicted_labels))
evaluate_print('average', y, average(predicted_scores))
evaluate_print('maximization', y, maximization(predicted_scores))

clf = LOF()
clf.fit(X)
evaluate_print('LOF', y, clf.decision_scores_)

clf = IForest()
clf.fit(X)
evaluate_print('IForest', y, clf.decision_scores_)
###############################################################################

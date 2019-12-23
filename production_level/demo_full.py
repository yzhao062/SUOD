# %%
import os
import warnings
import time
import numpy as np
import scipy as sp

import joblib
from joblib import effective_n_jobs
from joblib import Parallel, delayed
from copy import deepcopy

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as PCA_sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import check_array
from scipy.stats import rankdata

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
from pyod.utils.utility import precision_n_scores
from pyod.utils.utility import generate_bagging_indices
from pyod.utils.data import evaluate_print
from sklearn.metrics import roc_auc_score

from combo.models.score_comb import majority_vote, maximization, average

from sklearn.random_projection import johnson_lindenstrauss_min_dim
from jl_projection import jl_fit_transform, jl_transform

import warnings

warnings.filterwarnings("ignore")


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def _parallel_train(n_estimators, clfs, X, total_n_estimators,
                    rp_flag, objective_dim, verbose):
    X = check_array(X)
    # Build estimators
    estimators = []
    rp_transformers = []
    for i in range(n_estimators):
        estimator = deepcopy(clfs[i])
        if verbose > 1:
            print("Building estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))
        if rp_flag[i] == 1:
            X_scaled, jlt_transformer = jl_fit_transform(X, objective_dim)
            rp_transformers.append(jlt_transformer)

            estimator.fit(X_scaled)
            estimators.append(estimator)
        else:
            rp_transformers.append(None)
            estimator.fit(X)
            estimators.append(estimator)
    return estimators, rp_transformers


def _parallel_predict(n_estimators, clfs, X, total_n_estimators,
                      rp_flag, rp_transformers, verbose):
    X = check_array(X)

    pred = []
    for i in range(n_estimators):
        estimator = clfs[i]
        if verbose > 1:
            print("predicting with estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))

        if rp_flag[i] == 1:
            X_scaled = jl_transform(X, rp_transformers[i])
            pred.append(estimator.predict(X_scaled))

        else:
            pred.append(estimator.predict(X))

    return pred

def _parallel_decision_function(n_estimators, clfs, X, total_n_estimators,
                      rp_flag, rp_transformers, verbose):
    X = check_array(X)

    pred = []
    for i in range(n_estimators):
        estimator = clfs[i]
        if verbose > 1:
            print("predicting with estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))

        if rp_flag[i] == 1:
            X_scaled = jl_transform(X, rp_transformers[i])
            pred.append(estimator.decision_function(X_scaled))

        else:
            pred.append(estimator.decision_function(X))

    return pred


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs in 
    scikit-learn."""

    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs,
                                   dtype=np.int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


mat_file_list = [
    'cardio.mat',
        # 'satellite.mat',
    #    'satimage-2.mat',
    # 'mnist.mat',
]

mat_file = mat_file_list[0]
mat_file_name = mat_file.replace('.mat', '')
print("\n... Processing", mat_file_name, '...')
mat = sp.io.loadmat(os.path.join('datasets', mat_file))

X = mat['X']
y = mat['y']

X = StandardScaler().fit_transform(X)


# predict on the dataset and detection algorithm pairs
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

base_estimator_names = []

n_jobs = 6
n_estimators = len(base_estimators)

rp_clf_list = ['LOF', 'KNN', 'ABOD']
rp_ng_clf_list = ['IForest', 'PCA', 'HBOS']
proj_enabled = True
objective_dim = 6

# build RP code
# this can be a pre-defined list and directly supply to the system
rp_flag = np.zeros([n_estimators, 1], dtype=int)
rp_method = 'basic'
# rp_method = 'discrete'
# rp_method = 'circulant'
# rp_method = 'toeplitz'


for i in range(n_estimators):

    try:
        clf_name = type(base_estimators[i]).__name__

    except TypeError:
        print('Unknown detection algorithm.')
        clf_name = 'UNK'
        
    if clf_name not in list(clf_idx_mapping):
        # print(clf_name)
        clf_name = 'UNK'
    # build the estimator list
    base_estimator_names.append(clf_name)

    # check whether the projection is needed
    if clf_name in rp_clf_list:
        rp_flag[i] = 1
    elif clf_name in rp_ng_clf_list:
        continue
    else:
        warnings.warn("{clf_name} does not have a predefined projection code. "
                      "RP disabled.".format(clf_name=clf_name))

    
if not proj_enabled:
    # revert back
    rp_flag = np.zeros([n_estimators, 1], dtype=int)
##############################################################################

# load cost predictor
clf = joblib.load(os.path.join('saved_models', 'rf_predictor.joblib'))

# TODO: there should be a seperate predictor for prediction cost



# convert base estimators to the digestable form
clf_idx = np.asarray(list(map(clf_idx_mapping.get, base_estimator_names)))

X_detector_code = indices_to_one_hot(clf_idx - 1, 11)
X_shape_code_s = np.array([X.shape[0], X.shape[1]]).reshape(1, 2)
X_shape_code = np.repeat(X_shape_code_s, len(base_estimators), axis=0)
#
X_code = np.concatenate((X_shape_code, X_detector_code), axis=1)
time_cost_pred = clf.predict(X_code)

# Conduct Balanced Task Scheduling
n_estimators_list = []  # track the number of estimators for each worker
ranks = rankdata(time_cost_pred)
rank_sum = np.sum(ranks)
chunk_sum = rank_sum / n_jobs

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
    print('Worker', j + 1, 'sum of ranks:', sum_check[j])
    n_estimators_list.append(starts[j + 1] - starts[j])

print()

# Confirm the length of the estimators is consistent
assert (np.sum(n_estimators_list) == n_estimators)
assert (rank_sum == np.sum(sum_check))

n_jobs = min(effective_n_jobs(n_jobs), n_estimators)
print(starts)  # this is the list of being split

start = time.time()

# TODO: code cleanup. There is an existing bug for joblib on Windows:
# https://github.com/joblib/joblib/issues/806
# max_nbytes can be dropped on other OS

print('Parallel Training...')

all_results = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=True)(
    delayed(_parallel_train)(
        n_estimators_list[i],
        base_estimators[starts[i]:starts[i + 1]],
        X,
        n_estimators,
        rp_flag[starts[i]:starts[i + 1]],
        objective_dim,
        verbose=True)
    for i in range(n_jobs))

print('Balanced Scheduling Total Time:', time.time() - start)

trained_estimators = []
jl_transformers = []

# unfold the fitted models and the transformers 
for i in range(n_jobs):
    trained_estimators.extend(all_results[i][0])
    jl_transformers.extend(all_results[i][1])

print('Parallel Predicting...')

all_results_pred = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=True)(
    delayed(_parallel_predict)(
        n_estimators_list[i],
        trained_estimators[starts[i]:starts[i + 1]],
        X,
        n_estimators,
        rp_flag[starts[i]:starts[i + 1]],
        jl_transformers[starts[i]:starts[i + 1]],
        verbose=True)
    for i in range(n_jobs))

# unfold and generate the label matrix
predicted_labels = np.zeros([X.shape[0], n_estimators])
for i in range(n_jobs):
    predicted_labels[:, starts[i]:starts[i + 1]] = np.asarray(all_results_pred[i]).T
    

print('Parallel Score Prediciting...')

all_results_scores = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=True)(
    delayed(_parallel_decision_function)(
        n_estimators_list[i],
        trained_estimators[starts[i]:starts[i + 1]],
        X,
        n_estimators,
        rp_flag[starts[i]:starts[i + 1]],
        jl_transformers[starts[i]:starts[i + 1]],
        verbose=True)
    for i in range(n_jobs))

# unfold and generate the label matrix
predicted_scores = np.zeros([X.shape[0], n_estimators])
for i in range(n_jobs):
    predicted_scores[:, starts[i]:starts[i + 1]] = np.asarray(all_results_scores[i]).T
    
#%% Check point to see whether it is working
evaluate_print('majority vote', y, majority_vote(predicted_labels))
evaluate_print('average', y, average(predicted_scores))
evaluate_print('maximization', y, maximization(predicted_scores))

clf = LOF()
clf.fit(X)
evaluate_print('LOF', y, clf.decision_scores_)

clf = IForest()
clf.fit(X)
evaluate_print('IForest', y, clf.decision_scores_)

#%% Model Approximation
approx_clf_list = ['LOF', 'KNN', 'ABOD']
approx_ng_clf_list = ['IForest', 'PCA', 'HBOS', 'ABOD']
approx_enabled = True


# build approx code
# this can be a pre-defined list and directly supply to the system
approx_flag = np.zeros([n_estimators, 1], dtype=int)
# this can be supplied by the user
approx_clf = RandomForestRegressor()


# this may be combined with the first step
for i in range(n_estimators):

    try:
        clf_name = type(base_estimators[i]).__name__

    except TypeError:
        print('Unknown detection algorithm.')
        clf_name = 'UNK'

    # print(clf_name)
    base_estimator_names.append(clf_name)

    if clf_name in approx_clf_list:
        approx_flag[i] = 1
    elif clf_name in approx_ng_clf_list:
        continue
    else:
        warnings.warn("{clf_name} does not have a predefined approximation code. "
                      "Approximation disabled.".format(clf_name=clf_name))

if not approx_enabled:
    # revert back
    approx_flag = np.zeros([n_estimators, 1], dtype=int)


n_jobs, n_estimators_list, starts = _partition_estimators(n_estimators, n_jobs=n_jobs)


def _parallel_approx_estimators(n_estimators, clfs, X, total_n_estimators,
                                approx_flag, approximator, verbose):
    X = check_array(X)
    # Build estimators
    approximators = []

    for i in range(n_estimators):
        # estimator = deepcopy(clfs[i])
        estimator = clfs[i]
        approximater = deepcopy(approximator)
        if verbose > 1:
            print("Building estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))
        if approx_flag[i] == 1:
            
            pseudo_scores = estimator.decision_scores_
            # pseudo_scores = estimator.decision_function(X)
            approximater.fit(X, pseudo_scores)
            approximators.append(approximater)
        else:
            approximators.append(None)

    return approximators

all_approx_results = Parallel(n_jobs=n_jobs, verbose=True)(
    delayed(_parallel_approx_estimators)(
        n_estimators_list[i],
        trained_estimators[starts[i]:starts[i + 1]], 
        X, # if it is PyOD model, we do not need this
        n_estimators,
        approx_flag,
        approx_clf,
        verbose=True)
    for i in range(n_jobs))

approximators = []
# unfold the fitted approximators
for i in range(n_jobs):
    approximators.extend(all_approx_results[i])
    
#%% Second BPS for prediction

    
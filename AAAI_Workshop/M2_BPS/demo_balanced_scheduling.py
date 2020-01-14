import os
import sys
import time
import warnings
import numpy as np
import scipy as sp
from scipy.stats import rankdata

from sklearn.preprocessing import StandardScaler
from joblib import effective_n_jobs
from joblib import Parallel, delayed
from copy import deepcopy

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


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


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


##############################################################################


n_jobs = 5
n_estimators_total = 1000

mat_file = 'cardio.mat'
mat_file_name = mat_file.replace('.mat', '')
print("\n... Processing", mat_file_name, '...')
mat = sp.io.loadmat(os.path.join('../datasets', mat_file))

X = mat['X']
y = mat['y']

X = StandardScaler().fit_transform(X)

# load the pre-trained model cost predictor
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


# Conduct Balanced Task Scheduling
n_estimators_list = []
ranks = rankdata(predict_time)
##########################################
ranks = 1 + ranks/len(clfs)
##########################################

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
    print('Worker', j+1, 'sum of ranks:', sum_check[j])
    n_estimators_list.append(starts[j + 1] - starts[j])

print()

# Confirm the length of the estimators is consistent
assert (np.sum(n_estimators_list) == n_estimators_total)
assert (np.abs(rank_sum - np.sum(sum_check)) < 0.1 )

n_jobs = min(effective_n_jobs(n_jobs), n_estimators_total)
total_n_estimators = sum(n_estimators_list)
xdiff = [starts[n] - starts[n - 1] for n in range(1, len(starts))]
print(starts, xdiff)

start = time.time()
# https://github.com/joblib/joblib/issues/806
# max_nbytes can be dropped on other OS
all_results = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=True)(
    delayed(_parallel_build_estimators)(
        n_estimators_list[i],
        clfs_real[starts[i]:starts[i + 1]],
        X,
        total_n_estimators,
        verbose=True)
    for i in range(n_jobs))
print('Balanced Scheduling Total Time:', time.time() - start)


#############################################################################

print()
clfs = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        size=n_estimators_total)
clfs = np.sort(clfs)
clfs_real = []

for estimator in clfs:
    clfs_real.append(classifiers[estimator])

n_jobs, n_estimators, starts = _partition_estimators(len(clfs), n_jobs=n_jobs)
total_n_estimators = sum(n_estimators)
xdiff = [starts[n] - starts[n - 1] for n in range(1, len(starts))]
print(starts, xdiff)

start = time.time()
# https://github.com/joblib/joblib/issues/806
# max_nbytes can be dropped on other OS
all_results = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=True)(
    delayed(_parallel_build_estimators)(
        n_estimators[i],
        clfs_real[starts[i]:starts[i + 1]],
        X,
        total_n_estimators,
        verbose=True)
    for i in range(n_jobs))
print('Naive Split Total Time', time.time() - start)

print(mat_file_name, n_jobs, n_estimators)

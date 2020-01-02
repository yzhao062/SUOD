from copy import deepcopy

import numpy as np
from scipy.stats import rankdata
from joblib import effective_n_jobs
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .cost_predictor import clf_idx_mapping
from .jl_projection import jl_fit_transform, jl_transform


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def balanced_scheduling(time_cost_pred, n_estimators, n_jobs):
    """Conduct balanced scheduling based on the sum of rank, for both train
    and prediction.

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

    return n_estimators_per_job.tolist(), [0] + starts.tolist(), n_jobs


def cost_forecast_meta(clf, X, base_estimator_names):
    # convert base estimators to the digestible form
    clf_idx = np.asarray(
        list(map(clf_idx_mapping.get, base_estimator_names)))

    X_detector_code = indices_to_one_hot(clf_idx - 1, 11)
    X_shape_code_s = np.array([X.shape[0], X.shape[1]]).reshape(1, 2)
    X_shape_code = np.repeat(X_shape_code_s, len(base_estimator_names), axis=0)
    #
    X_code = np.concatenate((X_shape_code, X_detector_code), axis=1)
    time_cost_pred = clf.predict(X_code)

    return time_cost_pred


def _parallel_fit(n_estimators, clfs, X, total_n_estimators,
                  rp_flags, objective_dim, rp_method, verbose):
    X = check_array(X)
    # Build estimators
    estimators = []
    rp_transformers = []
    for i in range(n_estimators):
        estimator = deepcopy(clfs[i])
        if verbose > 1:
            print("Building estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))
        if rp_flags[i] == 1:
            X_scaled, jlt_transformer = jl_fit_transform(X, objective_dim,
                                                         rp_method)
            rp_transformers.append(jlt_transformer)

            estimator.fit(X_scaled)
            estimators.append(estimator)
        else:
            rp_transformers.append(None)
            estimator.fit(X)
            estimators.append(estimator)
    return estimators, rp_transformers


def _parallel_predict(n_estimators, clfs, X, total_n_estimators,
                      rp_flags, rp_transformers, verbose):
    X = check_array(X)

    pred = []
    for i in range(n_estimators):
        estimator = clfs[i]
        if verbose > 1:
            print("predicting with estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))

        if rp_flags[i] == 1:
            X_scaled = jl_transform(X, rp_transformers[i])
            pred.append(estimator.predict(X_scaled))

        else:
            pred.append(estimator.predict(X))

    return pred


def _parallel_decision_function(n_estimators, clfs, X, total_n_estimators,
                                rp_flags, rp_transformers, verbose):
    X = check_array(X)

    pred = []
    for i in range(n_estimators):
        estimator = clfs[i]
        if verbose > 1:
            print("predicting with estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))

        if rp_flags[i] == 1:
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

    return n_estimators_per_job.tolist(), [0] + starts.tolist(), n_jobs


def _parallel_approx_estimators(n_estimators, clfs, X, total_n_estimators,
                                approx_flag, approximator, verbose):
    """

    Parameters
    ----------
    n_estimators
    clfs
    X
    total_n_estimators
    approx_flag
    approximator
    verbose

    Returns
    -------

    """
    X = check_array(X)
    # Build estimators
    approximators = []

    # TODO: approximators can be different
    for i in range(n_estimators):
        estimator = clfs[i]

        check_is_fitted(estimator, ['decision_scores_'])

        # use the same type of approximator for all models
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

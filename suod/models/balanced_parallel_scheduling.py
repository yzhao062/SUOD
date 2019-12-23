import numpy as np
from scipy.stats import rankdata
from joblib import effective_n_jobs

from .cost_predictor import clf_idx_mapping


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def balanced_scheduling(time_cost_pred, n_estimators, n_jobs):
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


def cost_forecast_train(clf, X, base_estimator_names):
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

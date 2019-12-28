import numpy as np
import warnings
from .cost_predictor import clf_idx_mapping


def build_codes(n_estimators, base_estimators, clf_list, ng_clf_list,
                flag_global):
    """Core function for building codes for deciding whether enable random
    projection and supervised approximation.

    Parameters
    ----------
    n_estimators
    base_estimators
    clf_list
    ng_clf_list
    flag_global

    Returns
    -------

    """
    base_estimator_names = []
    flags = np.zeros([n_estimators, 1], dtype=int)

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
        if clf_name in clf_list:
            flags[i] = 1
        elif clf_name in ng_clf_list:
            continue
        else:
            warnings.warn(
                "{clf_name} does not have a predefined code. "
                "Code sets to 0.".format(clf_name=clf_name))

    if not flag_global:
        # revert back
        flags = np.zeros([n_estimators, 1], dtype=int)

    return flags, base_estimator_names

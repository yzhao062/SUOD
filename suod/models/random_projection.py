import numpy as np
import warnings
from .cost_predictor import clf_idx_mapping


def build_rp_codes(n_estimators, base_estimators, rp_clf_list, rp_ng_clf_list,
                   proj_enabled):
    base_estimator_names = []
    rp_flag = np.zeros([n_estimators, 1], dtype=int)

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
            warnings.warn(
                "{clf_name} does not have a predefined projection code. "
                "RP disabled.".format(clf_name=clf_name))

    if not proj_enabled:
        # revert back
        rp_flag = np.zeros([n_estimators, 1], dtype=int)

    return rp_flag, base_estimator_names

# base_estimator_names = []

# build RP code
# this can be a pre-defined list and directly supply to the system
# rp_flags = np.zeros([n_estimators, 1], dtype=int)
# rp_method = 'basic'
# # rp_method = 'discrete'
# # rp_method = 'circulant'
# # rp_method = 'toeplitz'
#
# for i in range(n_estimators):
#
#     try:
#         clf_name = type(base_estimators[i]).__name__
#
#     except TypeError:
#         print('Unknown detection algorithm.')
#         clf_name = 'UNK'
#
#     if clf_name not in list(cost_predictor.clf_idx_mapping):
#         # print(clf_name)
#         clf_name = 'UNK'
#     # build the estimator list
#     base_estimator_names.append(clf_name)
#
#     # check whether the projection is needed
#     if clf_name in rp_clf_list:
#         rp_flags[i] = 1
#     elif clf_name in rp_ng_clf_list:
#         continue
#     else:
#         warnings.warn("{clf_name} does not have a predefined projection code. "
#                       "RP disabled.".format(clf_name=clf_name))
#
# if not rp_flag_global:
#     # revert back
#     rp_flags = np.zeros([n_estimators, 1], dtype=int)
##############################################################################

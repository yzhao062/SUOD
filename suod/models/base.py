# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause
import os
import sys
import time

import warnings
from collections import defaultdict
import numbers

import numpy as np
from numpy import percentile
from scipy.special import erf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import check_array

import joblib
from joblib import Parallel, delayed

from .cost_predictor import clf_idx_mapping

from pyod.models.sklearn_base import _pprint
from pyod.utils.utility import _get_sklearn_version

from suod.models.parallel_processes import cost_forecast_meta
from suod.models.parallel_processes import balanced_scheduling
from suod.models.parallel_processes import _parallel_fit
from suod.models.parallel_processes import _parallel_predict
from suod.models.parallel_processes import _parallel_decision_function
from suod.models.parallel_processes import _partition_estimators
from suod.models.parallel_processes import _parallel_approx_estimators
from suod.models.utils.utility import _unfold_parallel

import warnings
from collections import defaultdict

if _get_sklearn_version() > 20:
    from inspect import signature
else:
    from sklearn.externals.funcsigs import signature


class SUOD(object):
    """Abstract class for all combination classes.

    Parameters
    ----------
    base_estimators: list, length must be greater than 1
        A list of base estimators. Certain methods must be present, e.g.,
        `fit` and `predict`.

    pre_fitted: bool, optional (default=False)
        Whether the base estimators are trained. If True, `fit`
        process may be skipped.
    """

    def __init__(self, base_estimators, contamination=0.05, n_jobs=None,
                 rp_clf_list=None, rp_ng_clf_list=None, rp_flag_global=True,
                 max_features=0.5, rp_method='basic', bps_flag=False,
                 approx_clf_list=None, approx_ng_clf_list=None,
                 approx_flag_global=True, approx_clf=None, verbose=False):

        assert (isinstance(base_estimators, (list)))
        if len(base_estimators) < 2:
            raise ValueError('At least 2 estimators are required')
        self.base_estimators = base_estimators
        self.n_estimators = len(base_estimators)
        self.rp_flag_global = rp_flag_global
        self.max_features = max_features
        self.rp_method = rp_method
        self.bps_flag = bps_flag
        self.verbose = verbose
        self.approx_clf_list = approx_clf_list
        self.approx_ng_clf_list = approx_ng_clf_list
        self.approx_flag_global = approx_flag_global
        self.contamination = contamination

        if approx_clf is not None:
            self.approx_clf = approx_clf
        else:
            self.approx_clf = RandomForestRegressor(n_estimators=100)

        if n_jobs is None:
            self.n_jobs = 1
        else:
            self.n_jobs = n_jobs

        # validate random projection list
        if rp_clf_list is None:
            # the algorithms that should be be using random projection
            self.rp_clf_list = ['LOF', 'KNN', 'ABOD']
        else:
            self.rp_clf_list = rp_clf_list

        if rp_ng_clf_list is None:
            # the algorithms that should be be using random projection
            self.rp_ng_clf_list = ['IForest', 'PCA', 'HBOS']
        else:
            self.rp_ng_clf_list = rp_ng_clf_list

        # validate model approximation list
        if approx_clf_list is None:
            # the algorithms that should be be using random projection
            self.approx_clf_list = ['LOF', 'KNN', 'ABOD']
        else:
            self.approx_clf_list = approx_clf_list

        if approx_ng_clf_list is None:
            # the algorithms that should be be using random projection
            self.approx_ng_clf_list = ['IForest', 'PCA', 'HBOS', 'ABOD']
        else:
            self.approx_ng_clf_list = approx_ng_clf_list

        # build flags for random projection
        self.rp_flags, self.base_estimator_names = build_codes(
            self.n_estimators,
            self.base_estimators,
            self.rp_clf_list,
            self.rp_ng_clf_list,
            self.rp_flag_global)

    def fit(self, X, y=None):
        """Fit estimator. y is optional for unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples,), optional (default=None)
            The ground truth of the input samples (labels).

        Returns
        -------
        self
        """
        X = check_array(X)
        n_samples, n_features = X.shape[0], X.shape[1]

        # Validate max_features for random projection
        if isinstance(self.max_features, (numbers.Integral, np.integer)):
            self.max_features_ = self.max_features
        else:  # float
            self.max_features_ = int(self.max_features * n_features)

        # build flags for random projection
        self.rp_flags_, _ = build_codes(
            self.n_estimators, self.base_estimators, self.rp_clf_list,
            self.rp_ng_clf_list, self.rp_flag_global)

        # decide whether bps is needed
        # it is turned off
        if self.bps_flag:
            # load the pre-trained cost predictor to forecast the train cost
            cost_predictor = joblib.load(
                os.path.join('../suod', 'models', 'saved_models',
                             'bps_train.joblib'))

            time_cost_pred = cost_forecast_meta(cost_predictor, X,
                                                self.base_estimator_names)

            # use BPS
            n_estimators_list, starts, n_jobs = balanced_scheduling(
                time_cost_pred, self.n_estimators, self.n_jobs)
        else:
            # use the default sklearn equal split
            n_estimators_list, starts, n_jobs = _partition_estimators(
                self.n_estimators, self.n_jobs)

        # fit the base models
        print('Parallel Training...')
        start = time.time()

        # TODO: code cleanup. There is an existing bug for joblib on Windows:
        # https://github.com/joblib/joblib/issues/806
        # max_nbytes can be dropped on other OS
        all_results = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=True)(
            delayed(_parallel_fit)(
                n_estimators_list[i],
                self.base_estimators[starts[i]:starts[i + 1]],
                X,
                self.n_estimators,
                self.rp_flags[starts[i]:starts[i + 1]],
                self.max_features_,
                self.rp_method,
                verbose=self.verbose)
            for i in range(n_jobs))

        print('Balanced Scheduling Total Train Time:', time.time() - start)

        # reformat and unfold the lists. Save the trained estimators and transformers
        all_results = list(map(list, zip(*all_results)))

        # overwrite estimators
        self.base_estimators = _unfold_parallel(all_results[0], n_jobs)
        self.jl_transformers_ = _unfold_parallel(all_results[1], n_jobs)

        return self

    def approximate(self, X):

        # todo: X may be optional
        # todo: allow to use a list of scores for approximation, instead of
        # todo: decision_scores

        self.approx_flags, _ = build_codes(self.n_estimators,
                                           self.base_estimators,
                                           self.approx_clf_list,
                                           self.approx_ng_clf_list,
                                           self.approx_flag_global)

        n_estimators_list, starts, n_jobs = _partition_estimators(
            self.n_estimators, n_jobs=self.n_jobs)

        print(starts)  # this is the list of being split
        start = time.time()

        all_approx_results = Parallel(n_jobs=n_jobs, verbose=True)(
            delayed(_parallel_approx_estimators)(
                n_estimators_list[i],
                self.base_estimators[starts[i]:starts[i + 1]],
                X,  # if it is a PyOD model, we do not need this
                self.n_estimators,
                self.approx_flags[starts[i]:starts[i + 1]],
                self.approx_clf,
                verbose=True)
            for i in range(n_jobs))

        print('Balanced Scheduling Total Test Time:', time.time() - start)

        self.approximators = _unfold_parallel(all_approx_results, n_jobs)
        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        labels : numpy array of shape (n_samples,)
            Class labels for each data sample.
        """
        X = check_array(X)
        n_samples, n_features = X.shape[0], X.shape[1]

        # decide whether bps is needed
        # it is turned off
        if self.bps_flag:
            # load the pre-trained cost predictor to forecast the train cost
            cost_predictor = joblib.load(
                os.path.join('../suod', 'models', 'saved_models',
                             'bps_train.joblib'))

            time_cost_pred = cost_forecast_meta(cost_predictor, X,
                                                self.base_estimator_names)

            n_estimators_list, starts, n_jobs = balanced_scheduling(
                time_cost_pred, self.n_estimators, self.n_jobs)
        else:
            # use simple equal split by sklearn
            n_estimators_list, starts, n_jobs = _partition_estimators(
                self.n_estimators, self.n_jobs)

        # fit the base models
        print('Parallel label prediction...')
        start = time.time()

        # TODO: code cleanup. There is an existing bug for joblib on Windows:
        # https://github.com/joblib/joblib/issues/806
        # max_nbytes can be dropped on other OS
        all_results_pred = Parallel(n_jobs=n_jobs, max_nbytes=None,
                                    verbose=True)(
            delayed(_parallel_predict)(
                n_estimators_list[i],
                self.base_estimators[starts[i]:starts[i + 1]],
                self.approximators[starts[i]:starts[i + 1]],
                X,
                self.n_estimators,
                self.rp_flags[starts[i]:starts[i + 1]],
                self.jl_transformers_[starts[i]:starts[i + 1]],
                self.approx_flags[starts[i]:starts[i + 1]],
                self.contamination,
                verbose=True)
            for i in range(n_jobs))

        print('Parallel Label Predicting without Approximators Total Time:',
              time.time() - start)

        # unfold and generate the label matrix
        predicted_labels = np.zeros([n_samples, self.n_estimators])
        for i in range(n_jobs):
            predicted_labels[:, starts[i]:starts[i + 1]] = np.asarray(
                all_results_pred[i]).T

        return predicted_labels

    def decision_function(self, X):
        """Predict raw anomaly scores of X using the fitted detectors.

        The anomaly score of an input sample is computed based on the fitted
        detector. For consistency, outliers are assigned with
        higher anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        X = check_array(X)
        n_samples, n_features = X.shape[0], X.shape[1]

        # decide whether bps is needed
        # it is turned off
        if self.bps_flag:
            # load the pre-trained cost predictor to forecast the train cost
            cost_predictor = joblib.load(
                os.path.join('../suod', 'models', 'saved_models',
                             'bps_train.joblib'))

            time_cost_pred = cost_forecast_meta(cost_predictor, X,
                                                self.base_estimator_names)

            n_estimators_list, starts, n_jobs = balanced_scheduling(
                time_cost_pred, self.n_estimators, self.n_jobs)
        else:
            # use simple equal split by sklearn
            n_estimators_list, starts, n_jobs = _partition_estimators(
                self.n_estimators, self.n_jobs)

        # fit the base models
        if self.verbose:
            print('Parallel score prediction...')
            start = time.time()

        # TODO: code cleanup. There is an existing bug for joblib on Windows:
        # https://github.com/joblib/joblib/issues/806
        # max_nbytes can be dropped on other OS
        all_results_scores = Parallel(n_jobs=n_jobs, max_nbytes=None,
                                      verbose=True)(
            delayed(_parallel_decision_function)(
                n_estimators_list[i],
                self.base_estimators[starts[i]:starts[i + 1]],
                self.approximators[starts[i]:starts[i + 1]],
                X,
                self.n_estimators,
                self.rp_flags[starts[i]:starts[i + 1]],
                self.jl_transformers_[starts[i]:starts[i + 1]],
                self.approx_flags[starts[i]:starts[i + 1]],
                verbose=True)
            for i in range(n_jobs))

        # fit the base models
        if self.verbose:
            print('Parallel Score Prediction without Approximators '
                  'Total Time:', time.time() - start)

        # unfold and generate the label matrix
        predicted_scores = np.zeros([n_samples, self.n_estimators])
        for i in range(n_jobs):
            predicted_scores[:, starts[i]:starts[i + 1]] = np.asarray(
                all_results_scores[i]).T

        return predicted_scores

    def __len__(self):
        """Returns the number of estimators in the ensemble."""
        return len(self.base_estimators)

    def __getitem__(self, index):
        """Returns the index'th estimator in the ensemble."""
        return self.base_estimators[index]

    def __iter__(self):
        """Returns iterator over estimators in the ensemble."""
        return iter(self.base_estimators)

    # noinspection PyMethodParameters
    def _get_param_names(cls):
        # noinspection PyPep8
        """Get parameter names for the estimator

        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.
        """

        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    # noinspection PyPep8
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """

        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        # noinspection PyPep8
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.

        Returns
        -------
        self : object
        """

        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def __repr__(self):
        # noinspection PyPep8
        """
        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.
        """

        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False),
                                               offset=len(class_name), ),)


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

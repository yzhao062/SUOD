# -*- coding: utf-8 -*-
"""Base class and functions of SUOD (Scalable Unsupervised Outlier Detection)
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: MIT


import os
import sys
import time

import numbers

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import check_array

import joblib
from joblib import Parallel, delayed, effective_n_jobs

from pyod.models.sklearn_base import _pprint
from pyod.utils.utility import _get_sklearn_version
from pyod.utils.utility import check_parameter

from suod.models.parallel_processes import cost_forecast_meta
from suod.models.parallel_processes import balanced_scheduling
from suod.models.parallel_processes import _parallel_fit
from suod.models.parallel_processes import _parallel_predict
from suod.models.parallel_processes import _parallel_predict_proba
from suod.models.parallel_processes import _parallel_decision_function
from suod.models.parallel_processes import _partition_estimators
from suod.models.parallel_processes import _parallel_approx_estimators
from ..utils.utility import _unfold_parallel, build_codes

import warnings
from collections import defaultdict

# temporary solution for relative imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

if _get_sklearn_version() > 20:
    from inspect import signature
else:
    from sklearn.externals.funcsigs import signature


# noinspection PyPep8
class SUOD(object):
    """SUOD (Scalable Unsupervised Outlier Detection) is an acceleration
    framework for large scale unsupervised outlier detector training and
    prediction. The corresponding paper is under review in KDD 2020.

    Parameters
    ----------
    base_estimators: list, length must be greater than 1
        A list of base estimators. Certain methods must be present, e.g.,
        `fit` and `predict`.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    n_jobs : optional (default=1)
        The number of jobs to run in parallel for both `fit` and
        `predict`. If -1, then the number of jobs is set to the
        the number of jobs that can actually run in parallel.

    rp_clf_list : list, optional (default=None)
        The list of outlier detection models to use random projection. The
        detector name should be consistent with PyOD.

    rp_ng_clf_list : list, optional (default=None)
        The list of outlier detection models NOT to use random projection. The
        detector name should be consistent with PyOD.

    rp_flag_global : bool, optional (default=True)
        If set to False, random projection is turned off for all base models.
        
    target_dim_frac : float in (0., 1), optional (default=0.5)
        The target compression ratio.

    jl_method : string, optional (default = 'basic')
        The JL projection method:

        - "basic": each component of the transformation matrix is taken at
          random in N(0,1).
        - "discrete", each component of the transformation matrix is taken at
          random in {-1,1}.
        - "circulant": the first row of the transformation matrix is taken at
          random in N(0,1), and each row is obtained from the previous one
          by a one-left shift.
        - "toeplitz": the first row and column of the transformation matrix
          is taken at random in N(0,1), and each diagonal has a constant value
          taken from these first vector.

    bps_flag : bool, optional (default=True)
        If set to False, balanced parallel scheduling is turned off.

    approx_clf_list : list, optional (default=None)
        The list of outlier detection models to use pseudo-supervised
        approximation. The detector name should be consistent with PyOD.

    approx_ng_clf_list : list, optional (default=None)
        The list of outlier detection models NOT to use pseudo-supervised
        approximation. The detector name should be consistent with PyOD.

    approx_flag_global : bool, optional (default=True)
        If set to False, pseudo-supervised approximation is turned off.

    approx_clf : object, optional (default: sklearn RandomForestRegressor)
        The supervised model used to approximate unsupervised models.

    cost_forecast_loc_fit : str, optional
        The location of the pretrained cost prediction forecast for training.

    cost_forecast_loc_pred : str, optional
        The location of the pretrained cost prediction forecast for prediction.

    verbose : bool, optional (default=False)
        Controls the verbosity of the building process.
    """

    def __init__(self, base_estimators, contamination=0.1, n_jobs=None,
                 rp_clf_list=None, rp_ng_clf_list=None, rp_flag_global=True,
                 target_dim_frac=0.5, jl_method='basic', bps_flag=True,
                 approx_clf_list=None, approx_ng_clf_list=None,
                 approx_flag_global=True, approx_clf=None,
                 cost_forecast_loc_fit=None, cost_forecast_loc_pred=None,
                 verbose=False):

        assert (isinstance(base_estimators, (list)))
        if len(base_estimators) < 2:
            raise ValueError('At least 2 estimators are required')
        self.base_estimators = base_estimators
        self.n_estimators = len(base_estimators)
        self.rp_flag_global = rp_flag_global
        self.target_dim_frac = target_dim_frac
        self.jl_method = jl_method
        self.bps_flag = bps_flag
        self.verbose = verbose
        self.approx_flag_global = approx_flag_global
        self.contamination = contamination

        self._parameter_validation(contamination, n_jobs, rp_clf_list,
                                   rp_ng_clf_list, approx_clf_list,
                                   approx_ng_clf_list, approx_clf,
                                   cost_forecast_loc_fit,
                                   cost_forecast_loc_pred)

        # build flags for random projection
        self.rp_flags, self.base_estimator_names = build_codes(
            self.base_estimators, self.rp_clf_list, self.rp_ng_clf_list,
            self.rp_flag_global)

    def _parameter_validation(self, contamination, n_jobs, rp_clf_list,
                              rp_ng_clf_list, approx_clf_list,
                              approx_ng_clf_list, approx_clf,
                              cost_forecast_loc_fit, cost_forecast_loc_pred):
        """Internal function to valid the initial parameters

        Returns
        -------
        self : object
            Post-check estimator.
        """

        if not (0. < contamination <= 0.5):
            raise ValueError("contamination must be in (0, 0.5], "
                             "got: %f" % contamination)

        self.contamination = contamination

        if approx_clf is not None:
            self.approx_clf = approx_clf
        else:
            self.approx_clf = RandomForestRegressor(n_estimators=50)

        if n_jobs is None:
            self.n_jobs = 1
        elif n_jobs == -1:
            self.n_jobs = effective_n_jobs()
        else:
            self.n_jobs = n_jobs

        # validate random projection list
        if rp_clf_list is None:
            # the algorithms that should be be using random projection
            self.rp_clf_list = ['LOF', 'KNN', 'ABOD', 'COF']
        else:
            self.rp_clf_list = rp_clf_list

        if rp_ng_clf_list is None:
            # the algorithms that should not be using random projection
            self.rp_ng_clf_list = ['IForest', 'PCA', 'HBOS', 'MCD', 'LMDD']
        else:
            self.rp_ng_clf_list = rp_ng_clf_list

        # Validate target_dim_frac
        check_parameter(self.target_dim_frac, low=0, high=1,
                        include_left=False,
                        include_right=True, param_name='target_dim_frac')

        # validate model approximation list
        if approx_clf_list is None:
            # the algorithms that should be be using approximation
            self.approx_clf_list = ['LOF', 'KNN', 'CBLOF', 'OCSVM']
        else:
            self.approx_clf_list = approx_clf_list

        if approx_ng_clf_list is None:
            # the algorithms that should not be using approximation
            self.approx_ng_clf_list = ['PCA', 'HBOS', 'ABOD', 'MCD',
                                       'LMDD', 'LSCP', 'IForest']
        else:
            self.approx_ng_clf_list = approx_ng_clf_list

        this_directory = os.path.abspath(os.path.dirname(__file__))

        # validate the trained model
        if cost_forecast_loc_fit is None:
            self.cost_forecast_loc_fit_ = os.path.join(
                this_directory, 'saved_models', 'bps_train.joblib')
        else:
            self.cost_forecast_loc_fit_ = cost_forecast_loc_fit

        if cost_forecast_loc_pred is None:
            self.cost_forecast_loc_pred_ = os.path.join(
                this_directory, 'saved_models', 'bps_prediction.joblib')
        else:
            self.cost_forecast_loc_pred_ = cost_forecast_loc_pred

        return self

    def fit(self, X):
        """Fit all base estimators.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X)
        n_samples, n_features = X.shape[0], X.shape[1]

        # Validate target_dim_frac for random projection
        if isinstance(self.target_dim_frac, (numbers.Integral, np.integer)):
            self.target_dim_frac_ = self.target_dim_frac
        else:  # float
            self.target_dim_frac_ = int(self.target_dim_frac * n_features)

        # build flags for random projection
        self.rp_flags_, _ = build_codes(self.base_estimators, self.rp_clf_list,
                                        self.rp_ng_clf_list,
                                        self.rp_flag_global)

        # decide whether bps is needed
        # it is turned off
        if self.bps_flag:
            # load the pre-trained cost predictor to forecast the train cost
            cost_predictor = joblib.load(self.cost_forecast_loc_fit_)

            time_cost_pred = cost_forecast_meta(cost_predictor, X,
                                                self.base_estimator_names)

            # use BPS
            n_estimators_list, starts, n_jobs = balanced_scheduling(
                time_cost_pred, self.n_estimators, self.n_jobs, self.verbose)
        else:
            # use the default sklearn equal split
            n_estimators_list, starts, n_jobs = _partition_estimators(
                self.n_estimators, self.n_jobs)

        # fit the base models
        # fit the base models
        if self.verbose:
            print('Parallel Training...')
            start = time.time()

        # TODO: code cleanup. There is an existing bug for joblib on Windows:
        # https://github.com/joblib/joblib/issues/806
        # a fix is on the way: https://github.com/joblib/joblib/pull/966
        # max_nbytes can be dropped on other OS
        all_results = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=True)(
            delayed(_parallel_fit)(
                n_estimators_list[i],
                self.base_estimators[starts[i]:starts[i + 1]],
                X,
                self.n_estimators,
                self.rp_flags[starts[i]:starts[i + 1]],
                self.target_dim_frac_,
                self.jl_method,
                verbose=self.verbose)
            for i in range(n_jobs))

        if self.verbose:
            print('Balanced Scheduling Total Train Time:', time.time() - start)

        # reformat and unfold the lists. Save the trained estimators and
        # transformers
        all_results = list(map(list, zip(*all_results)))

        # overwrite estimators
        self.base_estimators = _unfold_parallel(all_results[0], n_jobs)
        self.jl_transformers_ = _unfold_parallel(all_results[1], n_jobs)

        return self

    def approximate(self, X):
        """Use the supervised regressor (random forest by default) to
        approximate unsupervised fitted outlier detectors.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples. The same feature space of the unsupervised
            outlier detector will be used.

        Returns
        -------
        self : object
            The estimator after with approximation.
        """

        # todo: X may be optional
        # todo: allow to use a list of scores for approximation, instead of
        # todo: decision_scores

        self.approx_flags, _ = build_codes(self.base_estimators,
                                           self.approx_clf_list,
                                           self.approx_ng_clf_list,
                                           self.approx_flag_global)

        n_estimators_list, starts, n_jobs = _partition_estimators(
            self.n_estimators, n_jobs=self.n_jobs)

        all_approx_results = Parallel(n_jobs=n_jobs, verbose=True)(
            delayed(_parallel_approx_estimators)(
                n_estimators_list[i],
                self.base_estimators[starts[i]:starts[i + 1]],
                X,  # if it is a PyOD model, we do not need this
                self.n_estimators,
                self.approx_flags[starts[i]:starts[i + 1]],
                self.approx_clf,
                self.jl_transformers_[starts[i]:starts[i + 1]],
                verbose=True)
            for i in range(n_jobs))

        # print('Balanced Scheduling Total Test Time:', time.time() - start)

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
        outlier_labels : numpy array of shape (n_samples, n_estimators)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.
        """
        X = check_array(X)
        n_samples, n_features = X.shape[0], X.shape[1]

        # decide whether bps is needed
        # it is turned off
        if self.bps_flag:
            # load the pre-trained cost predictor to forecast the train cost
            cost_predictor = joblib.load(self.cost_forecast_loc_pred_)

            time_cost_pred = cost_forecast_meta(cost_predictor, X,
                                                self.base_estimator_names)

            n_estimators_list, starts, n_jobs = balanced_scheduling(
                time_cost_pred, self.n_estimators, self.n_jobs, self.verbose)
        else:
            # use simple equal split by sklearn
            n_estimators_list, starts, n_jobs = _partition_estimators(
                self.n_estimators, self.n_jobs)

        # fit the base models
        if self.verbose:
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
                # self.rp_flags[starts[i]:starts[i + 1]],
                self.jl_transformers_[starts[i]:starts[i + 1]],
                self.approx_flags[starts[i]:starts[i + 1]],
                self.contamination,
                verbose=True)
            for i in range(n_jobs))

        if self.verbose:
            print('Parallel Label Predicting without Approximators '
                  'Total Time:', time.time() - start)

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
            cost_predictor = joblib.load(self.cost_forecast_loc_pred_)

            time_cost_pred = cost_forecast_meta(cost_predictor, X,
                                                self.base_estimator_names)

            n_estimators_list, starts, n_jobs = balanced_scheduling(
                time_cost_pred, self.n_estimators, self.n_jobs, self.verbose)
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
                # self.rp_flags[starts[i]:starts[i + 1]],
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

    def predict_proba(self, X):
        """Predict the probability of a sample being outlier. Two approaches
        are possible:

        1. simply use Min-max conversion to linearly transform the outlier
           scores into the range of [0,1]. The model must be
           fitted first.
        2. use unifying scores, see :cite:`kriegel2011interpreting`.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        outlier_probability : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. Return the outlier probability, ranging
            in [0,1].
        """
        X = check_array(X)
        n_samples, n_features = X.shape[0], X.shape[1]

        # decide whether bps is needed
        # it is turned off
        if self.bps_flag:
            # load the pre-trained cost predictor to forecast the train cost
            cost_predictor = joblib.load(self.cost_forecast_loc_pred_)

            time_cost_pred = cost_forecast_meta(cost_predictor, X,
                                                self.base_estimator_names)

            n_estimators_list, starts, n_jobs = balanced_scheduling(
                time_cost_pred, self.n_estimators, self.n_jobs, self.verbose)
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
            delayed(_parallel_predict_proba)(
                n_estimators_list[i],
                self.base_estimators[starts[i]:starts[i + 1]],
                self.approximators[starts[i]:starts[i + 1]],
                X,
                self.n_estimators,
                # self.rp_flags[starts[i]:starts[i + 1]],
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

    # noinspection PyMethodParameters,PyPep8
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
        # noinspection PyPep8,PyPep8
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

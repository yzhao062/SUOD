# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

import warnings
from collections import defaultdict
from abc import ABC, abstractmethod

import numpy as np
from numpy import percentile
from scipy.special import erf
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from .cost_predictor import clf_idx_mapping

from pyod.models.sklearn_base import _pprint
from pyod.utils.utility import _get_sklearn_version

import warnings
from collections import defaultdict
from abc import ABC, abstractmethod

if _get_sklearn_version() > 20:
    from inspect import signature
else:
    from sklearn.externals.funcsigs import signature


class SUOD(ABC):
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

    def __init__(self, base_estimators, n_jobs=None, rp_clf_list=None,
                 rp_ng_clf_list=None, rp_flag_global=False, objective_dim=0.5,
                 rp_method='basic'):

        assert (isinstance(base_estimators, (list)))
        if len(base_estimators) < 2:
            raise ValueError('At least 2 estimators are required')
        self.base_estimators = base_estimators
        self.n_estimators = len(base_estimators)
        self.rp_flag_global = rp_flag_global

        if n_jobs is None:
            self.n_jobs = 1
        else:
            self.n_jobs = n_jobs

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
        pass

    # todo: make sure fit then predict is equivalent to fit_predict

    def fit_predict(self, X, y=None):
        """Fit estimator and predict on X. y is optional for unsupervised
        methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples,), optional (default=None)
            The ground truth of the input samples (labels).

        Returns
        -------
        labels : numpy array of shape (n_samples,)
            Class labels for each data sample.
        """
        pass

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
        pass

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : numpy array of shape (n_samples,)
            The class probabilities of the input samples.
            Classes are ordered by lexicographic order.
        """
        pass

    def _process_decision_scores(self):
        """Internal function to calculate key attributes for outlier detection
        combination tasks.

        - threshold_: used to decide the binary label
        - labels_: binary labels of training data

        Returns
        -------
        self
        """

        self.threshold_ = percentile(self.decision_scores_,
                                     100 * (1 - self.contamination))
        self.labels_ = (self.decision_scores_ > self.threshold_).astype(
            'int').ravel()

        # calculate for predict_proba()

        self._mu = np.mean(self.decision_scores_)
        self._sigma = np.std(self.decision_scores_)

        return self

    def _detector_predict(self, X):
        """Internal function to predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.
        """

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])

        pred_score = self.decision_function(X)
        return (pred_score > self.threshold_).astype('int').ravel()

    def _detector_predict_proba(self, X, proba_method='linear'):
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

        proba_method : str, optional (default='linear')
            Probability conversion method. It must be one of
            'linear' or 'unify'.

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. Return the outlier probability, ranging
            in [0,1].
        """

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        train_scores = self.decision_scores_

        test_scores = self.decision_function(X)

        probs = np.zeros([X.shape[0], int(self._classes)])
        if proba_method == 'linear':
            scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))
            probs[:, 1] = scaler.transform(
                test_scores.reshape(-1, 1)).ravel().clip(0, 1)
            probs[:, 0] = 1 - probs[:, 1]
            return probs

        elif proba_method == 'unify':
            # turn output into probability
            pre_erf_score = (test_scores - self._mu) / (
                    self._sigma * np.sqrt(2))
            erf_score = erf(pre_erf_score)
            probs[:, 1] = erf_score.clip(0, 1).ravel()
            probs[:, 0] = 1 - probs[:, 1]
            return probs
        else:
            raise ValueError(proba_method,
                             'is not a valid probability conversion method')

    def _set_n_classes(self, y):
        """Set the number of classes if `y` is presented.

        Parameters
        ----------
        y : numpy array of shape (n_samples,)
            Ground truth.

        Returns
        -------
        self
        """

        self._classes = 2  # default as binary classification
        if y is not None:
            check_classification_targets(y)
            self._classes = len(np.unique(y))

        return self

    def _set_weights(self, weights):
        """Internal function to set estimator weights.

        Parameters
        ----------
        weights : numpy array of shape (n_estimators,)
            Estimator weights. May be used after the alignment.

        Returns
        -------
        self

        """

        if weights is None:
            self.weights = np.ones([1, self.n_base_estimators_])
        else:
            self.weights = column_or_1d(weights).reshape(1, len(weights))
            assert (self.weights.shape[1] == self.n_base_estimators_)

            # adjust probability by a factor for integrity （added to 1）
            adjust_factor = self.weights.shape[1] / np.sum(weights)
            self.weights = self.weights * adjust_factor

            print(self.weights)
        return self

    def build_codes(object):
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

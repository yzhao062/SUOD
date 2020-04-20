# Author: Yue Zhao <zhaoy@cmu.edu>
# License: MIT
import numpy as np
from scipy.special import erf
from sklearn.preprocessing import MinMaxScaler

from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.abod import ABOD
from pyod.models.mcd import MCD

# suppress warnings
import warnings

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

warnings.filterwarnings("ignore")


def _unfold_parallel(lists, n_jobs):
    """Internal function to unfold the results returned from the parallization

    Parameters
    ----------
    lists : list
        The results from the parallelization operations.

    n_jobs : optional (default=1)
        The number of jobs to run in parallel for both `fit` and
        `predict`. If -1, then the number of jobs is set to the
        number of cores.

    Returns
    -------
    result_list : list
        The list of unfolded result.
    """
    full_list = []
    for i in range(n_jobs):
        full_list.extend(lists[i])
    return full_list


def build_codes(base_estimators, clf_list, ng_clf_list, flag_global):
    """Core function for building codes for deciding whether enable random
    projection and supervised approximation.

    Parameters
    ----------
    base_estimators: list, length must be greater than 1
        A list of base estimators. Certain methods must be present, e.g.,
        `fit` and `predict`.

    clf_list : list
        The list of outlier detection models to use a certain function. The
        detector name should be consistent with PyOD.

    ng_clf_list : list
        The list of outlier detection models to NOT use a certain function. The
        detector name should be consistent with PyOD.

    flag_global : bool
        The global flag to override the code build.

    Returns
    -------

    """
    n_estimators = len(base_estimators)
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


def raw_score_to_proba(decision_scores, test_scores, method='linear'):
    """Utility function to convert raw scores to probability. The
    transformation can be either linear or using unify introduced in
    :cite:`kriegel2011interpreting`.

    Parameters
    ----------
    decision_scores : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is fitted.

    test_scores : numpy array of shape (n_samples,)
        The outlier scores of the test data to be converted.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is fitted.

    method : str, optional (default='linear')
        The transformation method, either 'linear' or 'unify'

    Returns
    -------
    outlier_probability : numpy array of shape (n_samples,)
        For each observation, tells whether or not
        it should be considered as an outlier according to the
        fitted model. Return the outlier probability, ranging
        in [0,1].
    """
    probs = np.zeros([test_scores.shape[0], 2])
    if method == 'linear':
        scaler = MinMaxScaler().fit(decision_scores.reshape(-1, 1))
        probs[:, 1] = scaler.transform(
            test_scores.reshape(-1, 1)).ravel().clip(0, 1)
        probs[:, 0] = 1 - probs[:, 1]
        return probs

    elif method == 'unify':
        # turn output into probability
        pre_erf_score = (test_scores - np.mean(decision_scores, axis=0)) / (
                np.std(decision_scores) * np.sqrt(2))
        erf_score = erf(pre_erf_score)
        probs[:, 1] = erf_score.clip(0, 1).ravel()
        probs[:, 0] = 1 - probs[:, 1]
        return probs
    else:
        raise ValueError(
            method, 'is not a valid probability conversion method')


def get_estimators(contamination):
    """Internal method to create a list of 600 random base outlier detectors.

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    Returns
    -------
    base_detectors : list
        A list of initialized random base outlier detectors.

    """
    BASE_ESTIMATORS = [
        LOF(n_neighbors=5, contamination=contamination),
        LOF(n_neighbors=10, contamination=contamination),
        LOF(n_neighbors=15, contamination=contamination),
        LOF(n_neighbors=25, contamination=contamination),
        LOF(n_neighbors=35, contamination=contamination),
        LOF(n_neighbors=45, contamination=contamination),
        LOF(n_neighbors=50, contamination=contamination),
        LOF(n_neighbors=55, contamination=contamination),
        LOF(n_neighbors=60, contamination=contamination),
        LOF(n_neighbors=65, contamination=contamination),
        LOF(n_neighbors=70, contamination=contamination),
        LOF(n_neighbors=75, contamination=contamination),
        LOF(n_neighbors=80, contamination=contamination),
        LOF(n_neighbors=85, contamination=contamination),
        LOF(n_neighbors=90, contamination=contamination),
        LOF(n_neighbors=95, contamination=contamination),
        LOF(n_neighbors=100, contamination=contamination),

        ABOD(n_neighbors=5, contamination=contamination),
        ABOD(n_neighbors=10, contamination=contamination),
        ABOD(n_neighbors=15, contamination=contamination),
        ABOD(n_neighbors=20, contamination=contamination),
        ABOD(n_neighbors=25, contamination=contamination),
        ABOD(n_neighbors=30, contamination=contamination),
        ABOD(n_neighbors=35, contamination=contamination),
        ABOD(n_neighbors=40, contamination=contamination),

        LOF(n_neighbors=5, contamination=contamination),
        LOF(n_neighbors=10, contamination=contamination),
        LOF(n_neighbors=15, contamination=contamination),
        LOF(n_neighbors=25, contamination=contamination),
        LOF(n_neighbors=35, contamination=contamination),
        LOF(n_neighbors=45, contamination=contamination),
        LOF(n_neighbors=50, contamination=contamination),
        LOF(n_neighbors=55, contamination=contamination),
        LOF(n_neighbors=60, contamination=contamination),
        LOF(n_neighbors=65, contamination=contamination),
        LOF(n_neighbors=70, contamination=contamination),
        LOF(n_neighbors=75, contamination=contamination),
        LOF(n_neighbors=80, contamination=contamination),
        LOF(n_neighbors=85, contamination=contamination),
        LOF(n_neighbors=90, contamination=contamination),
        LOF(n_neighbors=95, contamination=contamination),
        LOF(n_neighbors=100, contamination=contamination),

        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),

        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),

        KNN(n_neighbors=5, contamination=contamination),
        KNN(n_neighbors=15, contamination=contamination),
        KNN(n_neighbors=25, contamination=contamination),
        KNN(n_neighbors=35, contamination=contamination),
        KNN(n_neighbors=45, contamination=contamination),
        KNN(n_neighbors=50, contamination=contamination),
        KNN(n_neighbors=55, contamination=contamination),
        KNN(n_neighbors=65, contamination=contamination),
        KNN(n_neighbors=75, contamination=contamination),
        KNN(n_neighbors=85, contamination=contamination),
        KNN(n_neighbors=85, contamination=contamination),
        KNN(n_neighbors=85, contamination=contamination),
        KNN(n_neighbors=95, contamination=contamination),
        KNN(n_neighbors=100, contamination=contamination),

        IForest(n_estimators=50, contamination=contamination),
        IForest(n_estimators=100, contamination=contamination),
        IForest(n_estimators=150, contamination=contamination),
        IForest(n_estimators=200, contamination=contamination),
        IForest(n_estimators=50, contamination=contamination),
        IForest(n_estimators=100, contamination=contamination),
        IForest(n_estimators=150, contamination=contamination),
        IForest(n_estimators=200, contamination=contamination),

        LOF(n_neighbors=5, contamination=contamination),
        LOF(n_neighbors=10, contamination=contamination),
        LOF(n_neighbors=15, contamination=contamination),
        LOF(n_neighbors=25, contamination=contamination),
        LOF(n_neighbors=35, contamination=contamination),
        LOF(n_neighbors=45, contamination=contamination),
        LOF(n_neighbors=50, contamination=contamination),
        LOF(n_neighbors=55, contamination=contamination),
        LOF(n_neighbors=60, contamination=contamination),
        LOF(n_neighbors=65, contamination=contamination),
        LOF(n_neighbors=70, contamination=contamination),
        LOF(n_neighbors=75, contamination=contamination),
        LOF(n_neighbors=80, contamination=contamination),
        LOF(n_neighbors=85, contamination=contamination),
        LOF(n_neighbors=90, contamination=contamination),
        LOF(n_neighbors=95, contamination=contamination),
        LOF(n_neighbors=100, contamination=contamination),

        LOF(n_neighbors=5, contamination=contamination),
        LOF(n_neighbors=10, contamination=contamination),
        LOF(n_neighbors=15, contamination=contamination),
        LOF(n_neighbors=25, contamination=contamination),
        LOF(n_neighbors=35, contamination=contamination),
        LOF(n_neighbors=45, contamination=contamination),
        LOF(n_neighbors=50, contamination=contamination),
        LOF(n_neighbors=55, contamination=contamination),
        LOF(n_neighbors=60, contamination=contamination),
        LOF(n_neighbors=65, contamination=contamination),
        LOF(n_neighbors=70, contamination=contamination),
        LOF(n_neighbors=75, contamination=contamination),
        LOF(n_neighbors=80, contamination=contamination),
        LOF(n_neighbors=85, contamination=contamination),
        LOF(n_neighbors=90, contamination=contamination),
        LOF(n_neighbors=95, contamination=contamination),
        LOF(n_neighbors=100, contamination=contamination),

        LOF(n_neighbors=5, contamination=contamination),
        LOF(n_neighbors=10, contamination=contamination),
        LOF(n_neighbors=15, contamination=contamination),
        LOF(n_neighbors=25, contamination=contamination),
        LOF(n_neighbors=35, contamination=contamination),
        LOF(n_neighbors=45, contamination=contamination),
        LOF(n_neighbors=50, contamination=contamination),
        LOF(n_neighbors=55, contamination=contamination),
        LOF(n_neighbors=60, contamination=contamination),
        LOF(n_neighbors=65, contamination=contamination),
        LOF(n_neighbors=70, contamination=contamination),
        LOF(n_neighbors=75, contamination=contamination),
        LOF(n_neighbors=80, contamination=contamination),
        LOF(n_neighbors=85, contamination=contamination),
        LOF(n_neighbors=90, contamination=contamination),
        LOF(n_neighbors=95, contamination=contamination),
        LOF(n_neighbors=100, contamination=contamination),

        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),

        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),

        KNN(n_neighbors=5, contamination=contamination),
        KNN(n_neighbors=15, contamination=contamination),
        KNN(n_neighbors=25, contamination=contamination),
        KNN(n_neighbors=35, contamination=contamination),
        KNN(n_neighbors=45, contamination=contamination),
        KNN(n_neighbors=50, contamination=contamination),
        KNN(n_neighbors=55, contamination=contamination),
        KNN(n_neighbors=65, contamination=contamination),
        KNN(n_neighbors=75, contamination=contamination),
        KNN(n_neighbors=85, contamination=contamination),
        KNN(n_neighbors=85, contamination=contamination),
        KNN(n_neighbors=85, contamination=contamination),
        KNN(n_neighbors=95, contamination=contamination),
        KNN(n_neighbors=100, contamination=contamination),

        IForest(n_estimators=50, contamination=contamination),
        IForest(n_estimators=100, contamination=contamination),
        IForest(n_estimators=150, contamination=contamination),
        IForest(n_estimators=200, contamination=contamination),
        IForest(n_estimators=50, contamination=contamination),
        IForest(n_estimators=100, contamination=contamination),
        IForest(n_estimators=150, contamination=contamination),
        IForest(n_estimators=200, contamination=contamination),

        LOF(n_neighbors=5, contamination=contamination),
        LOF(n_neighbors=10, contamination=contamination),
        LOF(n_neighbors=15, contamination=contamination),
        LOF(n_neighbors=25, contamination=contamination),
        LOF(n_neighbors=35, contamination=contamination),
        LOF(n_neighbors=45, contamination=contamination),
        LOF(n_neighbors=50, contamination=contamination),
        LOF(n_neighbors=55, contamination=contamination),
        LOF(n_neighbors=60, contamination=contamination),
        LOF(n_neighbors=65, contamination=contamination),
        LOF(n_neighbors=70, contamination=contamination),
        LOF(n_neighbors=75, contamination=contamination),
        LOF(n_neighbors=80, contamination=contamination),
        LOF(n_neighbors=85, contamination=contamination),
        LOF(n_neighbors=90, contamination=contamination),
        LOF(n_neighbors=95, contamination=contamination),
        LOF(n_neighbors=100, contamination=contamination),

        LOF(n_neighbors=5, contamination=contamination),
        LOF(n_neighbors=10, contamination=contamination),
        LOF(n_neighbors=15, contamination=contamination),
        LOF(n_neighbors=25, contamination=contamination),
        LOF(n_neighbors=35, contamination=contamination),
        LOF(n_neighbors=45, contamination=contamination),
        LOF(n_neighbors=50, contamination=contamination),
        LOF(n_neighbors=55, contamination=contamination),
        LOF(n_neighbors=60, contamination=contamination),
        LOF(n_neighbors=65, contamination=contamination),
        LOF(n_neighbors=70, contamination=contamination),
        LOF(n_neighbors=75, contamination=contamination),
        LOF(n_neighbors=80, contamination=contamination),
        LOF(n_neighbors=85, contamination=contamination),
        LOF(n_neighbors=90, contamination=contamination),
        LOF(n_neighbors=95, contamination=contamination),
        LOF(n_neighbors=100, contamination=contamination),

        LOF(n_neighbors=5, contamination=contamination),
        LOF(n_neighbors=10, contamination=contamination),
        LOF(n_neighbors=15, contamination=contamination),
        LOF(n_neighbors=25, contamination=contamination),
        LOF(n_neighbors=35, contamination=contamination),
        LOF(n_neighbors=45, contamination=contamination),
        LOF(n_neighbors=50, contamination=contamination),
        LOF(n_neighbors=55, contamination=contamination),
        LOF(n_neighbors=60, contamination=contamination),
        LOF(n_neighbors=65, contamination=contamination),
        LOF(n_neighbors=70, contamination=contamination),
        LOF(n_neighbors=75, contamination=contamination),
        LOF(n_neighbors=80, contamination=contamination),
        LOF(n_neighbors=85, contamination=contamination),
        LOF(n_neighbors=90, contamination=contamination),
        LOF(n_neighbors=95, contamination=contamination),
        LOF(n_neighbors=100, contamination=contamination),

        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),

        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),

        KNN(n_neighbors=5, contamination=contamination),
        KNN(n_neighbors=15, contamination=contamination),
        KNN(n_neighbors=25, contamination=contamination),
        KNN(n_neighbors=35, contamination=contamination),
        KNN(n_neighbors=45, contamination=contamination),
        KNN(n_neighbors=50, contamination=contamination),
        KNN(n_neighbors=55, contamination=contamination),
        KNN(n_neighbors=65, contamination=contamination),
        KNN(n_neighbors=75, contamination=contamination),
        KNN(n_neighbors=85, contamination=contamination),
        KNN(n_neighbors=85, contamination=contamination),
        KNN(n_neighbors=85, contamination=contamination),
        KNN(n_neighbors=95, contamination=contamination),
        KNN(n_neighbors=100, contamination=contamination),

        IForest(n_estimators=50, contamination=contamination),
        IForest(n_estimators=100, contamination=contamination),
        IForest(n_estimators=150, contamination=contamination),
        IForest(n_estimators=200, contamination=contamination),
        IForest(n_estimators=50, contamination=contamination),
        IForest(n_estimators=100, contamination=contamination),
        IForest(n_estimators=150, contamination=contamination),
        IForest(n_estimators=200, contamination=contamination),

        LOF(n_neighbors=5, contamination=contamination),
        LOF(n_neighbors=10, contamination=contamination),
        LOF(n_neighbors=15, contamination=contamination),
        LOF(n_neighbors=25, contamination=contamination),
        LOF(n_neighbors=35, contamination=contamination),
        LOF(n_neighbors=45, contamination=contamination),
        LOF(n_neighbors=50, contamination=contamination),
        LOF(n_neighbors=55, contamination=contamination),
        LOF(n_neighbors=60, contamination=contamination),
        LOF(n_neighbors=65, contamination=contamination),
        LOF(n_neighbors=70, contamination=contamination),
        LOF(n_neighbors=75, contamination=contamination),
        LOF(n_neighbors=80, contamination=contamination),
        LOF(n_neighbors=85, contamination=contamination),
        LOF(n_neighbors=90, contamination=contamination),
        LOF(n_neighbors=95, contamination=contamination),
        LOF(n_neighbors=100, contamination=contamination),

        LOF(n_neighbors=5, contamination=contamination),
        LOF(n_neighbors=10, contamination=contamination),
        LOF(n_neighbors=15, contamination=contamination),
        LOF(n_neighbors=25, contamination=contamination),
        LOF(n_neighbors=35, contamination=contamination),
        LOF(n_neighbors=45, contamination=contamination),
        LOF(n_neighbors=50, contamination=contamination),
        LOF(n_neighbors=55, contamination=contamination),
        LOF(n_neighbors=60, contamination=contamination),
        LOF(n_neighbors=65, contamination=contamination),
        LOF(n_neighbors=70, contamination=contamination),
        LOF(n_neighbors=75, contamination=contamination),
        LOF(n_neighbors=80, contamination=contamination),
        LOF(n_neighbors=85, contamination=contamination),
        LOF(n_neighbors=90, contamination=contamination),
        LOF(n_neighbors=95, contamination=contamination),
        LOF(n_neighbors=100, contamination=contamination),

        LOF(n_neighbors=5, contamination=contamination),
        LOF(n_neighbors=10, contamination=contamination),
        LOF(n_neighbors=15, contamination=contamination),
        LOF(n_neighbors=25, contamination=contamination),
        LOF(n_neighbors=35, contamination=contamination),
        LOF(n_neighbors=45, contamination=contamination),
        LOF(n_neighbors=50, contamination=contamination),
        LOF(n_neighbors=55, contamination=contamination),
        LOF(n_neighbors=60, contamination=contamination),
        LOF(n_neighbors=65, contamination=contamination),
        LOF(n_neighbors=70, contamination=contamination),
        LOF(n_neighbors=75, contamination=contamination),
        LOF(n_neighbors=80, contamination=contamination),
        LOF(n_neighbors=85, contamination=contamination),
        LOF(n_neighbors=90, contamination=contamination),
        LOF(n_neighbors=95, contamination=contamination),
        LOF(n_neighbors=100, contamination=contamination),

        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),

        ABOD(n_neighbors=5, contamination=contamination),
        ABOD(n_neighbors=10, contamination=contamination),
        ABOD(n_neighbors=15, contamination=contamination),
        ABOD(n_neighbors=20, contamination=contamination),
        ABOD(n_neighbors=25, contamination=contamination),
        ABOD(n_neighbors=30, contamination=contamination),
        ABOD(n_neighbors=35, contamination=contamination),
        ABOD(n_neighbors=40, contamination=contamination),

        IForest(n_estimators=50, contamination=contamination),
        IForest(n_estimators=100, contamination=contamination),
        IForest(n_estimators=150, contamination=contamination),
        IForest(n_estimators=200, contamination=contamination),
        IForest(n_estimators=50, contamination=contamination),
        IForest(n_estimators=100, contamination=contamination),
        IForest(n_estimators=150, contamination=contamination),
        IForest(n_estimators=200, contamination=contamination),
        IForest(n_estimators=150, contamination=contamination),
        IForest(n_estimators=200, contamination=contamination),

        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),

        KNN(n_neighbors=5, contamination=contamination),
        KNN(n_neighbors=15, contamination=contamination),
        KNN(n_neighbors=25, contamination=contamination),
        KNN(n_neighbors=35, contamination=contamination),
        KNN(n_neighbors=45, contamination=contamination),
        KNN(n_neighbors=50, contamination=contamination),
        KNN(n_neighbors=55, contamination=contamination),
        KNN(n_neighbors=65, contamination=contamination),
        KNN(n_neighbors=75, contamination=contamination),
        KNN(n_neighbors=85, contamination=contamination),
        KNN(n_neighbors=85, contamination=contamination),
        KNN(n_neighbors=85, contamination=contamination),
        KNN(n_neighbors=95, contamination=contamination),
        KNN(n_neighbors=100, contamination=contamination),

        IForest(n_estimators=50, contamination=contamination),
        IForest(n_estimators=100, contamination=contamination),
        IForest(n_estimators=150, contamination=contamination),
        IForest(n_estimators=200, contamination=contamination),
        IForest(n_estimators=50, contamination=contamination),
        IForest(n_estimators=100, contamination=contamination),
        IForest(n_estimators=150, contamination=contamination),
        IForest(n_estimators=200, contamination=contamination),

        LOF(n_neighbors=5, contamination=contamination),
        LOF(n_neighbors=10, contamination=contamination),
        LOF(n_neighbors=15, contamination=contamination),
        LOF(n_neighbors=25, contamination=contamination),
        LOF(n_neighbors=35, contamination=contamination),
        LOF(n_neighbors=45, contamination=contamination),
        LOF(n_neighbors=50, contamination=contamination),
        LOF(n_neighbors=55, contamination=contamination),
        LOF(n_neighbors=60, contamination=contamination),
        LOF(n_neighbors=65, contamination=contamination),
        LOF(n_neighbors=70, contamination=contamination),
        LOF(n_neighbors=75, contamination=contamination),
        LOF(n_neighbors=80, contamination=contamination),
        LOF(n_neighbors=85, contamination=contamination),
        LOF(n_neighbors=90, contamination=contamination),
        LOF(n_neighbors=95, contamination=contamination),
        LOF(n_neighbors=100, contamination=contamination),

        ABOD(n_neighbors=5, contamination=contamination),
        ABOD(n_neighbors=10, contamination=contamination),
        ABOD(n_neighbors=15, contamination=contamination),
        ABOD(n_neighbors=20, contamination=contamination),
        ABOD(n_neighbors=25, contamination=contamination),
        ABOD(n_neighbors=30, contamination=contamination),
        ABOD(n_neighbors=35, contamination=contamination),
        ABOD(n_neighbors=40, contamination=contamination),

        LOF(n_neighbors=5, contamination=contamination),
        LOF(n_neighbors=10, contamination=contamination),
        LOF(n_neighbors=15, contamination=contamination),
        LOF(n_neighbors=25, contamination=contamination),
        LOF(n_neighbors=35, contamination=contamination),
        LOF(n_neighbors=45, contamination=contamination),
        LOF(n_neighbors=50, contamination=contamination),
        LOF(n_neighbors=55, contamination=contamination),
        LOF(n_neighbors=60, contamination=contamination),
        LOF(n_neighbors=65, contamination=contamination),
        LOF(n_neighbors=70, contamination=contamination),
        LOF(n_neighbors=75, contamination=contamination),
        LOF(n_neighbors=80, contamination=contamination),
        LOF(n_neighbors=85, contamination=contamination),
        LOF(n_neighbors=90, contamination=contamination),
        LOF(n_neighbors=95, contamination=contamination),
        LOF(n_neighbors=100, contamination=contamination),

        LOF(n_neighbors=5, contamination=contamination),
        LOF(n_neighbors=10, contamination=contamination),
        LOF(n_neighbors=15, contamination=contamination),
        LOF(n_neighbors=25, contamination=contamination),
        LOF(n_neighbors=35, contamination=contamination),
        LOF(n_neighbors=45, contamination=contamination),
        LOF(n_neighbors=50, contamination=contamination),
        LOF(n_neighbors=55, contamination=contamination),
        LOF(n_neighbors=60, contamination=contamination),
        LOF(n_neighbors=65, contamination=contamination),
        LOF(n_neighbors=70, contamination=contamination),
        LOF(n_neighbors=75, contamination=contamination),
        LOF(n_neighbors=80, contamination=contamination),
        LOF(n_neighbors=85, contamination=contamination),
        LOF(n_neighbors=90, contamination=contamination),
        LOF(n_neighbors=95, contamination=contamination),
        LOF(n_neighbors=100, contamination=contamination),

        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),
        HBOS(contamination=contamination),

        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),
        PCA(contamination=contamination),

        KNN(n_neighbors=5, contamination=contamination),
        KNN(n_neighbors=15, contamination=contamination),
        KNN(n_neighbors=25, contamination=contamination),
        KNN(n_neighbors=35, contamination=contamination),
        KNN(n_neighbors=45, contamination=contamination),
        KNN(n_neighbors=50, contamination=contamination),
        KNN(n_neighbors=55, contamination=contamination),
        KNN(n_neighbors=65, contamination=contamination),
        KNN(n_neighbors=75, contamination=contamination),
        KNN(n_neighbors=85, contamination=contamination),
        KNN(n_neighbors=85, contamination=contamination),
        KNN(n_neighbors=85, contamination=contamination),
        KNN(n_neighbors=95, contamination=contamination),
        KNN(n_neighbors=100, contamination=contamination),

        IForest(n_estimators=50, contamination=contamination),
        IForest(n_estimators=100, contamination=contamination),
        IForest(n_estimators=150, contamination=contamination),
        IForest(n_estimators=200, contamination=contamination),
        IForest(n_estimators=50, contamination=contamination),
        IForest(n_estimators=100, contamination=contamination),
        IForest(n_estimators=150, contamination=contamination),
        IForest(n_estimators=200, contamination=contamination),

        LOF(n_neighbors=5, contamination=contamination),
        LOF(n_neighbors=10, contamination=contamination),
        LOF(n_neighbors=15, contamination=contamination),
        LOF(n_neighbors=25, contamination=contamination),
        LOF(n_neighbors=35, contamination=contamination),
        LOF(n_neighbors=45, contamination=contamination),
        LOF(n_neighbors=50, contamination=contamination),
        LOF(n_neighbors=55, contamination=contamination),
        LOF(n_neighbors=60, contamination=contamination),
        LOF(n_neighbors=65, contamination=contamination),
        LOF(n_neighbors=70, contamination=contamination),
        LOF(n_neighbors=75, contamination=contamination),
        LOF(n_neighbors=80, contamination=contamination),
        LOF(n_neighbors=85, contamination=contamination),
        LOF(n_neighbors=90, contamination=contamination),
        LOF(n_neighbors=95, contamination=contamination),
        LOF(n_neighbors=100, contamination=contamination),

        ABOD(n_neighbors=5, contamination=contamination),
        ABOD(n_neighbors=10, contamination=contamination),
        ABOD(n_neighbors=15, contamination=contamination),
        ABOD(n_neighbors=20, contamination=contamination),
        ABOD(n_neighbors=25, contamination=contamination),
        ABOD(n_neighbors=30, contamination=contamination),
        ABOD(n_neighbors=35, contamination=contamination),
        ABOD(n_neighbors=40, contamination=contamination),
        ABOD(n_neighbors=45, contamination=contamination),

        OCSVM(contamination=contamination),
        OCSVM(contamination=contamination),
        OCSVM(contamination=contamination),
        OCSVM(contamination=contamination),
        OCSVM(contamination=contamination),
        OCSVM(contamination=contamination),
        OCSVM(contamination=contamination),
        OCSVM(contamination=contamination),
        OCSVM(contamination=contamination),
        OCSVM(contamination=contamination),

        MCD(contamination=contamination),
        MCD(contamination=contamination),
        MCD(contamination=contamination),
        MCD(contamination=contamination),
        MCD(contamination=contamination),
        MCD(contamination=contamination),
        MCD(contamination=contamination),
        MCD(contamination=contamination),
        MCD(contamination=contamination),
        MCD(contamination=contamination),

        MCD(contamination=contamination),
        MCD(contamination=contamination),
        MCD(contamination=contamination),
        MCD(contamination=contamination),
        MCD(contamination=contamination),
        MCD(contamination=contamination),
        MCD(contamination=contamination),
        MCD(contamination=contamination),
        MCD(contamination=contamination),
        MCD(contamination=contamination),

        LOF(n_neighbors=75, contamination=contamination),
        LOF(n_neighbors=80, contamination=contamination),
        LOF(n_neighbors=85, contamination=contamination),
        LOF(n_neighbors=90, contamination=contamination),
        LOF(n_neighbors=95, contamination=contamination),
        LOF(n_neighbors=100, contamination=contamination),

        ABOD(n_neighbors=5, contamination=contamination),
        ABOD(n_neighbors=10, contamination=contamination),
        ABOD(n_neighbors=15, contamination=contamination),
        ABOD(n_neighbors=20, contamination=contamination),
        ABOD(n_neighbors=25, contamination=contamination),
        ABOD(n_neighbors=30, contamination=contamination),
        ABOD(n_neighbors=35, contamination=contamination),
        ABOD(n_neighbors=40, contamination=contamination),
    ]

    return BASE_ESTIMATORS

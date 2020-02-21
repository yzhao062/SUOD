import os
import sys

import scipy as sp

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.lscp import LSCP
from pyod.utils.data import evaluate_print

from combo.models.score_comb import majority_vote, maximization, average

# suppress warnings
import warnings

warnings.filterwarnings("ignore")

# temporary solution for relative imports in case combo is not installed
# if combo is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from suod.models.base import SUOD

if __name__ == "__main__":
    # load files
    mat_file_list = [
        'cardio.mat',
        # 'satellite.mat',
        # 'satimage-2.mat',
        # 'mnist.mat',
    ]

    mat_file = mat_file_list[0]
    mat_file_name = mat_file.replace('.mat', '')
    print("\n... Processing", mat_file_name, '...')
    mat = sp.io.loadmat(os.path.join('', 'datasets', mat_file))

    X = mat['X']
    y = mat['y']

    # standardize data to be digestible for most algorithms
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.4, random_state=42)

    contamination = y.sum() / len(y)

    base_estimators = [
        LOF(n_neighbors=5, contamination=contamination),
        LOF(n_neighbors=15, contamination=contamination),
        LOF(n_neighbors=25, contamination=contamination),
        LOF(n_neighbors=35, contamination=contamination),
        LOF(n_neighbors=45, contamination=contamination),
        HBOS(contamination=contamination),
        PCA(contamination=contamination),
        OCSVM(contamination=contamination),
        KNN(n_neighbors=5, contamination=contamination),
        KNN(n_neighbors=15, contamination=contamination),
        KNN(n_neighbors=25, contamination=contamination),
        KNN(n_neighbors=35, contamination=contamination),
        KNN(n_neighbors=45, contamination=contamination),
        IForest(n_estimators=50, contamination=contamination),
        IForest(n_estimators=100, contamination=contamination),
        LOF(n_neighbors=5, contamination=contamination),
        LOF(n_neighbors=15, contamination=contamination),
        LOF(n_neighbors=25, contamination=contamination),
        LOF(n_neighbors=35, contamination=contamination),
        LOF(n_neighbors=45, contamination=contamination),
        HBOS(contamination=contamination),
        PCA(contamination=contamination),
        OCSVM(contamination=contamination),
        KNN(n_neighbors=5, contamination=contamination),
        KNN(n_neighbors=15, contamination=contamination),
        KNN(n_neighbors=25, contamination=contamination),
        KNN(n_neighbors=35, contamination=contamination),
        KNN(n_neighbors=45, contamination=contamination),
        IForest(n_estimators=50, contamination=contamination),
        IForest(n_estimators=100, contamination=contamination),
        LOF(n_neighbors=5, contamination=contamination),
        LOF(n_neighbors=15, contamination=contamination),
        LOF(n_neighbors=25, contamination=contamination),
        LOF(n_neighbors=35, contamination=contamination),
        LOF(n_neighbors=45, contamination=contamination),
        HBOS(contamination=contamination),
        PCA(contamination=contamination),
        OCSVM(contamination=contamination),
        KNN(n_neighbors=5, contamination=contamination),
        KNN(n_neighbors=15, contamination=contamination),
        KNN(n_neighbors=25, contamination=contamination),
        KNN(n_neighbors=35, contamination=contamination),
        KNN(n_neighbors=45, contamination=contamination),
        IForest(n_estimators=50, contamination=contamination),
        IForest(n_estimators=100, contamination=contamination),
        LSCP(detector_list=[LOF(contamination=contamination),
                            LOF(contamination=contamination)])
    ]

    model = SUOD(base_estimators=base_estimators, n_jobs=6, bps_flag=True,
                 contamination=contamination, approx_flag_global=True)

    model.fit(X_train)  # fit all models with X
    model.approximate(X_train)  # conduct model approximation if it is enabled
    predicted_labels = model.predict(X_test)  # predict labels
    predicted_scores = model.decision_function(X_test)  # predict scores
    predicted_probs = model.predict_proba(X_test)  # predict scores

    ###########################################################################
    # compared with other approaches
    evaluate_print('majority vote', y_test, majority_vote(predicted_labels))
    evaluate_print('average', y_test, average(predicted_scores))
    evaluate_print('maximization', y_test, maximization(predicted_scores))

    evaluate_print('average', y_test, average(predicted_probs))
    evaluate_print('maximization', y_test, maximization(predicted_probs))

    clf = LOF()
    clf.fit(X_train)
    evaluate_print('LOF', y_test, clf.decision_function(X_test))

    clf = IForest()
    clf.fit(X_train)
    evaluate_print('IForest', y_test, clf.decision_function(X_test))


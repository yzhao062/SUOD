# %%
import os
import sys
import time
import numpy as np
import scipy as sp

import joblib
from joblib import Parallel, delayed

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.lscp import LSCP
from pyod.utils.data import evaluate_print

from combo.models.score_comb import majority_vote, maximization, average

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
    
    contamination = y.sum()/len(y)

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
                 contamination=contamination, approx_flag_global=False)

    model.fit(X)
    model.approximate(X)
    predicted_labels = model.predict(X)
    predicted_scores = model.decision_function(X)

    # %%
    evaluate_print('majority vote', y, majority_vote(predicted_labels))
    evaluate_print('average', y, average(predicted_scores))
    evaluate_print('maximization', y, maximization(predicted_scores))

    clf = LOF()
    clf.fit(X)
    evaluate_print('LOF', y, clf.decision_scores_)

    clf = IForest()
    clf.fit(X)
    evaluate_print('IForest', y, clf.decision_scores_)

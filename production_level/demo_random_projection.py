# %%
import os
import time
import numpy as np
import scipy as sp

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as PCA_sklearn

from pyod.models.iforest import IForest
from pyod.models.abod import ABOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.mcd import MCD
from pyod.models.lscp import LSCP
from pyod.utils.utility import precision_n_scores
from pyod.utils.utility import generate_bagging_indices
from sklearn.metrics import roc_auc_score

from sklearn.random_projection import johnson_lindenstrauss_min_dim
from jl_projection import jl_fit_transform

import warnings

warnings.filterwarnings("ignore")

# not all the mat files are included
mat_file_list = [
    # 'annthyroid.mat',
    # 'BreastW.mat',
    'cardio.mat',
    # 'glass.mat',
    # 'http_n.mat',
    # 'ionosphere.mat',
    # 'letter.mat',
    # 'lympho.mat',
    # 'mammography.mat',
    # 'mnist.mat',
    # 'musk.mat',
    # 'optdigits.mat',
    # 'pendigits.mat',
    # 'pima.mat',
    'satellite.mat',
    'satimage-2.mat',
    # 'seismic.mat',
    # 'shuttle.mat',
    # 'smtp_n.mat',
    # 'speech.mat',
    # 'vertebral.mat',
    # 'vowels.mat',
    # 'thyroid.mat',
    # 'wbc.mat',
    # 'wine.mat'
]

n_iter = 10

for mat_file in mat_file_list:
    mat_file_name = mat_file.replace('.mat', '')
    print("\n... Processing", mat_file_name, '...')
    mat = sp.io.loadmat(os.path.join('./datasets', mat_file))

    X = mat['X']
    y = mat['y']

    X = StandardScaler().fit_transform(X)

    if X.shape[1] <= 20:
        print('skipped', mat_file_name)
        continue

    dim_new = int(X.shape[1] / 2)

    original_roc = []
    original_prn = []
    original_time = []

    basic_roc = []
    basic_prn = []
    basic_time = []

    discrete_roc = []
    discrete_prn = []
    discrete_time = []

    circulant_roc = []
    circulant_prn = []
    circulant_time = []

    toeplitz_roc = []
    toeplitz_prn = []
    toeplitz_time = []

    pca_roc = []
    pca_prn = []
    pca_time = []

    rp_roc = []
    rp_prn = []
    rp_time = []

    for j in range(n_iter):
        start = time.time()
        clf = LOF()  # change this to other detection algorithms
        clf.fit(X)
        y_train_scores = clf.decision_scores_
        original_time.append(time.time() - start)
        original_roc.append(roc_auc_score(y, y_train_scores))
        original_prn.append(precision_n_scores(y, y_train_scores))

        X_transformer, _ = jl_fit_transform(X, dim_new, "basic")
        start = time.time()
        clf.fit(X_transformer)
        y_train_scores = clf.decision_scores_
        basic_time.append(time.time() - start)
        basic_roc.append(roc_auc_score(y, y_train_scores))
        basic_prn.append(precision_n_scores(y, y_train_scores))

        X_transformer, _ = jl_fit_transform(X, dim_new, "discrete")
        start = time.time()
        clf.fit(X_transformer)
        y_train_scores = clf.decision_scores_
        discrete_time.append(time.time() - start)
        discrete_roc.append(roc_auc_score(y, y_train_scores))
        discrete_prn.append(precision_n_scores(y, y_train_scores))

        X_transformer, _ = jl_fit_transform(X, dim_new, "circulant")
        start = time.time()
        clf.fit(X_transformer)
        y_train_scores = clf.decision_scores_
        circulant_time.append(time.time() - start)
        circulant_roc.append(roc_auc_score(y, y_train_scores))
        circulant_prn.append(precision_n_scores(y, y_train_scores))

        X_transformer, _ = jl_fit_transform(X, dim_new, "toeplitz")
        start = time.time()
        clf.fit(X_transformer)
        y_train_scores = clf.decision_scores_
        toeplitz_time.append(time.time() - start)
        toeplitz_roc.append(roc_auc_score(y, y_train_scores))
        toeplitz_prn.append(precision_n_scores(y, y_train_scores))

        X_transformer = PCA_sklearn(n_components=dim_new).fit_transform(X)
        start = time.time()
        clf.fit(X_transformer)
        y_train_scores = clf.decision_scores_
        pca_time.append(time.time() - start)
        pca_roc.append(roc_auc_score(y, y_train_scores))
        pca_prn.append(precision_n_scores(y, y_train_scores))

        selected_features = generate_bagging_indices(random_state=j,
                                                     bootstrap_features=False,
                                                     n_features=int(
                                                         X.shape[1]),
                                                     min_features=dim_new,
                                                     max_features=dim_new + 1)
        assert (dim_new == len(selected_features))
        X_transformer = X[:, selected_features]
        start = time.time()
        clf.fit(X_transformer)
        y_train_scores = clf.decision_scores_
        rp_time.append(time.time() - start)
        rp_roc.append(roc_auc_score(y, y_train_scores))
        rp_prn.append(precision_n_scores(y, y_train_scores))

    print()
    print(mat_file_name)
    print('original', np.round(np.average(original_time), decimals=4),
          np.round(np.average(original_roc), decimals=4),
          np.round(np.average(original_prn), decimals=4))
    print('basic', np.round(np.average(basic_time), decimals=4),
          np.round(np.average(basic_roc), decimals=4),
          np.round(np.average(basic_prn), decimals=4))
    print('discrete', np.round(np.average(discrete_time), decimals=4),
          np.round(np.average(discrete_roc), decimals=4),
          np.round(np.average(discrete_prn), decimals=4))
    print('circulant', np.round(np.average(circulant_time), decimals=4),
          np.round(np.average(circulant_roc), decimals=4),
          np.round(np.average(circulant_prn), decimals=4))
    print('toeplitz', np.round(np.average(toeplitz_time), decimals=4),
          np.round(np.average(toeplitz_roc), decimals=4),
          np.round(np.average(toeplitz_prn), decimals=4))
    print('pca', np.round(np.average(pca_time), decimals=4),
          np.round(np.average(pca_roc), decimals=4),
          np.round(np.average(pca_prn), decimals=4))
    print('rp', np.round(np.average(rp_time), decimals=4),
          np.round(np.average(rp_roc), decimals=4),
          np.round(np.average(rp_prn), decimals=4))
    print(clf)

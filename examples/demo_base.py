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

base_estimators = [
    LOF(n_neighbors=5), LOF(n_neighbors=15),
    LOF(n_neighbors=25), LOF(n_neighbors=35),
    LOF(n_neighbors=45),
    HBOS(),
    PCA(),
    OCSVM(),
    KNN(n_neighbors=5), KNN(n_neighbors=15),
    KNN(n_neighbors=25), KNN(n_neighbors=35),
    KNN(n_neighbors=45),
    IForest(n_estimators=50),
    IForest(n_estimators=100),
    LOF(n_neighbors=5), LOF(n_neighbors=15),
    LOF(n_neighbors=25), LOF(n_neighbors=35),
    LOF(n_neighbors=45),
    HBOS(),
    PCA(),
    OCSVM(),
    KNN(n_neighbors=5), KNN(n_neighbors=15),
    KNN(n_neighbors=25), KNN(n_neighbors=35),
    KNN(n_neighbors=45),
    IForest(n_estimators=50),
    IForest(n_estimators=100),
    LOF(n_neighbors=5), LOF(n_neighbors=15),
    LOF(n_neighbors=25), LOF(n_neighbors=35),
    LOF(n_neighbors=45),
    HBOS(),
    PCA(),
    OCSVM(),
    KNN(n_neighbors=5), KNN(n_neighbors=15),
    KNN(n_neighbors=25), KNN(n_neighbors=35),
    KNN(n_neighbors=45),
    IForest(n_estimators=50),
    IForest(n_estimators=100),
    LSCP(detector_list=[LOF(), LOF()])
]
model = SUOD(base_estimators=base_estimators)

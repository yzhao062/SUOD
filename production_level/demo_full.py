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

mat_file_list = [
    # 'annthyroid.mat',
    # 'BreastW.mat',
    'cardio.mat',
]

mat_file = mat_file_list[0]
mat_file_name = mat_file.replace('.mat', '')
print("\n... Processing", mat_file_name, '...')
mat = sp.io.loadmat(os.path.join('./datasets', mat_file))

X = mat['X']
y = mat['y']

X = StandardScaler().fit_transform(X)
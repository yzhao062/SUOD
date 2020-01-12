"""Part of the code is adapted from https://github.com/PTAug/jlt-python

"""

import numpy as np
from sklearn.utils import check_array
import math


def jl_fit_transform(X, objective_dim, method="basic"):
    """Reduce the dimension of X by Johnson–Lindenstrauss process.

    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
        The input samples.

    objective_dim : int
        The expected output dimension.

    method : string, optional (default = 'basic')
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

    Returns
    -------
    X_transformed :
    jlt_transformer :

    """
    if method.lower() == "basic":
        jlt_transformer = (1 / math.sqrt(objective_dim)) \
                          * np.random.normal(0, 1, size=(objective_dim,
                                                         len(X[0])))
    elif method.lower() == "discrete":
        jlt_transformer = (1 / math.sqrt(objective_dim)) \
                          * np.random.choice([-1, 1],
                                             size=(objective_dim, len(X[0])))
    elif method.lower() == "circulant":
        from scipy.linalg import circulant
        first_row = np.random.normal(0, 1, size=(1, len(X[0])))
        jlt_transformer = ((1 / math.sqrt(objective_dim))
                           * circulant(first_row))[:objective_dim]
    elif method.lower() == "toeplitz":
        from scipy.linalg import toeplitz
        first_row = np.random.normal(0, 1, size=(1, len(X[0])))
        first_column = np.random.normal(0, 1, size=(1, objective_dim))
        jlt_transformer = ((1 / math.sqrt(objective_dim))
                           * toeplitz(first_column, first_row))
    else:
        NotImplementedError('Wrong transformation type')

    jlt_transformer = jlt_transformer.T
    #    trans_dataset = []
    #    [trans_dataset.append(np.dot(jlt, np.transpose(X[i])))
    #     for i in range(len(X))]

    return np.dot(X, jlt_transformer), jlt_transformer


def jl_transform(X, jl_transformer):
    X = check_array(X)
    jl_transformer = check_array(jl_transformer)

    # no need for transformation
    if np.array_equal(jl_transformer, np.ones([X.shape[1], X.shape[1]])):
        return X

    if X.shape[1] != jl_transformer.shape[0]:
        ValueError("X and jl_transformer have different dimensions.")
    return np.dot(X, jl_transformer)


# %%

X = np.random.random([30, 8])
X2 = np.random.random([3, 8])
X_transformed, jl_transformer = jl_fit_transform(X, 5, method='toeplitz')
X2_transformed = jl_transform(X2, jl_transformer)

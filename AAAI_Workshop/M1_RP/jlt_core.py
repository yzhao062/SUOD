"""This code is adapted from https://github.com/PTAug/jlt-python
Author: Paul-Tristan Auguin
"""

import numpy as np
import math


def jl_transform(dataset_in, objective_dim, type_transform="basic"):
    """
    This function takes the dataset_in and returns the reduced dataset. The
    output dimension is objective_dim.

    dataset_in -- original dataset, list of Numpy ndarray
    objective_dim -- objective dimension of the reduction
    type_transform -- type of the transformation matrix used.
    If "basic" (default), each component of the transformation matrix
    is taken at random in N(0,1).
    If "discrete", each component of the transformation matrix
    is taken at random in {-1,1}.
    If "circulant", he first row of the transformation matrix
    is taken at random in N(0,1), and each row is obtainedfrom the
    previous one by a one-left shift.
    If "toeplitz", the first row and column of the transformation
    matrix is taken at random in N(0,1), and each diagonal has a
    constant value taken from these first vector.
    """
    if type_transform.lower() == "basic":
        jlt = (1 / math.sqrt(objective_dim)) * np.random.normal(0, 1, size=(
        objective_dim, len(dataset_in[0])))
    elif type_transform.lower() == "discrete":
        jlt = (1 / math.sqrt(objective_dim)) * np.random.choice([-1, 1], size=(
            objective_dim, len(dataset_in[0])))
    elif type_transform.lower() == "circulant":
        from scipy.linalg import circulant
        first_row = np.random.normal(0, 1, size=(1, len(dataset_in[0])))
        jlt = ((1 / math.sqrt(objective_dim)) * circulant(first_row))[
              :objective_dim]
    elif type_transform.lower() == "toeplitz":
        from scipy.linalg import toeplitz
        first_row = np.random.normal(0, 1, size=(1, len(dataset_in[0])))
        first_column = np.random.normal(0, 1, size=(1, objective_dim))
        jlt = ((1 / math.sqrt(objective_dim)) * toeplitz(first_column,
                                                         first_row))
    else:
        print('Wrong transformation type')
        return None

    trans_dataset = []
    [trans_dataset.append(np.dot(jlt, np.transpose(dataset_in[i])))
     for i in range(len(dataset_in))]

    return np.asarray(trans_dataset)

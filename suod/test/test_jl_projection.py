# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

import unittest
from sklearn.utils.testing import assert_equal

# temporary solution for relative imports in case pyod is not installed
# if suod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from suod.models.jl_projection import jl_fit_transform, jl_transform


class TestCostPredictor(unittest.TestCase):
    def setUp(self):
        self.objective_dim = 5
        self.X = np.random.random([30, 8])
        self.X2 = np.random.random([10, 8])

    def test_fit_transform(self):
        X_transformed, jl_transformer = jl_fit_transform(self.X,
                                                         self.objective_dim,
                                                         method='basic')
        assert_equal(X_transformed.shape,
                     (self.X.shape[0], self.objective_dim))
        assert_equal(jl_transformer.shape,
                     (self.X.shape[1], self.objective_dim))

        X_transformed, jl_transformer = jl_fit_transform(self.X,
                                                         self.objective_dim,
                                                         method='discrete')
        assert_equal(X_transformed.shape,
                     (self.X.shape[0], self.objective_dim))
        assert_equal(jl_transformer.shape,
                     (self.X.shape[1], self.objective_dim))

        X_transformed, jl_transformer = jl_fit_transform(self.X,
                                                         self.objective_dim,
                                                         method='circulant')
        assert_equal(X_transformed.shape,
                     (self.X.shape[0], self.objective_dim))
        assert_equal(jl_transformer.shape,
                     (self.X.shape[1], self.objective_dim))

        X_transformed, jl_transformer = jl_fit_transform(self.X,
                                                         self.objective_dim,
                                                         method='toeplitz')
        assert_equal(X_transformed.shape,
                     (self.X.shape[0], self.objective_dim))
        assert_equal(jl_transformer.shape,
                     (self.X.shape[1], self.objective_dim))

    def test_transform(self):
        _, jl_transformer = jl_fit_transform(self.X,
                                             self.objective_dim,
                                             method='basic')
        X_transformed = jl_transform(self.X2, jl_transformer)
        assert_equal(X_transformed.shape,
                     (self.X2.shape[0], self.objective_dim))

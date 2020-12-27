# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

import unittest
from numpy.testing import assert_equal

# temporary solution for relative imports in case pyod is not installed
# if suod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from suod.utils.utility import raw_score_to_proba


class TestUtility(unittest.TestCase):
    def setUp(self):
        self.objective_dim = 5
        self.train_scores = np.random.random([100, ])
        self.test_scores = np.random.random([20, ])

    def test_raw_score_to_proba(self):
        prob1 = raw_score_to_proba(self.train_scores, self.test_scores)
        assert (prob1.max() <= 1)
        assert (prob1.min() >= 0)

        prob2 = raw_score_to_proba(self.train_scores, self.test_scores,
                                   method='unify')
        assert (prob2.max() <= 1)
        assert (prob2.min() >= 0)

    def tearDown(self):
        pass

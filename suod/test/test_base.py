# -*- coding: utf-8 -*-
import os
import sys

import unittest
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_true

import numpy as np

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from suod.models.base import SUOD
from pyod.utils.data import generate_data
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from pyod.models.hbos import HBOS
from pyod.models.lscp import LSCP


class TestBASE(unittest.TestCase):
    def setUp(self):
        self.n_train = 1000
        self.n_test = 500
        self.contamination = 0.1
        self.roc_floor = 0.6
        self.random_state = 42
        self.X_train, self.y_train, self.X_test, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=self.random_state)

        self.base_estimators = [
            LOF(n_neighbors=5, contamination=self.contamination),
            LOF(n_neighbors=15, contamination=self.contamination),
            LOF(n_neighbors=25, contamination=self.contamination),
            LOF(n_neighbors=35, contamination=self.contamination),
            LOF(n_neighbors=45, contamination=self.contamination),
            HBOS(contamination=self.contamination),
            PCA(contamination=self.contamination),
            LSCP(detector_list=[
                LOF(n_neighbors=5, contamination=self.contamination),
                LOF(n_neighbors=15, contamination=self.contamination)],
                random_state=self.random_state)
        ]

        self.cost_forecast_loc_fit_ = os.path.join('bps_train.joblib')

        self.cost_forecast_loc_pred_ = os.path.join('bps_prediction.joblib')

        self.model = SUOD(base_estimators=self.base_estimators, n_jobs=2,
                          rp_flag_global=True, bps_flag=True,
                          contamination=self.contamination,
                          approx_flag_global=True,
                          cost_forecast_loc_fit=self.cost_forecast_loc_fit_,
                          cost_forecast_loc_pred=self.cost_forecast_loc_pred_)

    def test_fit(self):
        """
        Test base class initialization

        :return:
        """
        self.model.fit(self.X_train)

    def test_approximate(self):
        self.model.fit(self.X_train)
        self.model.approximate(self.X_train)

    def test_predict(self):
        self.model.fit(self.X_train)
        self.model.approximate(self.X_train)
        self.model.predict(self.X_test)

    def test_decision_function(self):
        self.model.fit(self.X_train)
        self.model.approximate(self.X_train)
        self.model.decision_function(self.X_test)

# -*- coding: utf-8 -*-
import os
import sys

import unittest

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from suod.models.cost_predictor import build_cost_predictor


class TestCostPredictor(unittest.TestCase):
    def setUp(self):
        pass

    def test_build(self):
        build_cost_predictor(file_name="summary_train.xlsx",
                             output_file="bps_train.joblib",
                             save_to_local=False)
        build_cost_predictor(file_name="summary_prediction.xlsx",
                             output_file="bps_prediction.joblib",
                             save_to_local=False)

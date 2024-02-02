# -*- coding: utf-8 -*-
import os
import sys

import unittest

# temporary solution for relative imports in case pyod is not installed
# if suod
# is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from suod.models.base import SUOD
from pyod.utils.data import generate_data
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from pyod.models.hbos import HBOS
from pyod.models.lscp import LSCP
from joblib import dump, load


class TestModelSaveLoad(unittest.TestCase):
	def setUp(self):
		self.n_train = 1000
		self.n_test = 500
		self.contamination = 0.1
		self.roc_floor = 0.6
		self.random_state = 42
		self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
			n_train=self.n_train, n_test=self.n_test, behaviour='new',
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

		self.model = SUOD(base_estimators=self.base_estimators, n_jobs=2,
						  rp_flag_global=True, bps_flag=True,
						  contamination=self.contamination,
						  approx_flag_global=True,
						  verbose=True)

	def test_save(self):
		self.model.fit(self.X_train)  # fit all models with X
		self.model.approximate(
			self.X_train)  # conduct model approximation if it is enabled

		# save the model
		dump(self.model, 'model.joblib')
		assert (os.path.exists('model.joblib'))
		os.remove('model.joblib')

	def test_load(self):
		self.model.fit(self.X_train)  # fit all models with X
		self.model.approximate(
			self.X_train)  # conduct model approximation if it is enabled

		# save the model
		dump(self.model, 'model.joblib')
		model = load('model.joblib')

		predicted_labels = model.predict(self.X_test)  # predict labels
		predicted_scores = model.decision_function(
			self.X_test)  # predict scores
		predicted_probs = model.predict_proba(self.X_test)  # predict scores

		assert (len(predicted_labels) != 0)

	# assert (predicted_scores)
	# assert (predicted_probs)

	def tearDown(self):
		if os.path.exists('model.joblib'):
			os.remove('model.joblib')

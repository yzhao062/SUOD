SUOD: A Scalable Unsupervised Outlier Detection Framework
=========================================================

**Deployment & Documentation & Stats**

.. image:: https://img.shields.io/pypi/v/suod.svg?color=brightgreen
   :target: https://pypi.org/project/suod/
   :alt: PyPI version


.. image:: https://readthedocs.org/projects/suod/badge/?version=latest
   :target: https://suod.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


.. image:: https://img.shields.io/github/stars/yzhao062/suod.svg
   :target: https://github.com/yzhao062/suod/stargazers
   :alt: GitHub stars


.. image:: https://img.shields.io/github/forks/yzhao062/suod.svg?color=blue
   :target: https://github.com/yzhao062/suod/network
   :alt: GitHub forks


.. image:: https://pepy.tech/badge/suod
   :target: https://pepy.tech/project/suod
   :alt: Downloads


.. image:: https://pepy.tech/badge/suod/month
   :target: https://pepy.tech/project/suod
   :alt: Downloads


----


**Build Status & Coverage & Maintainability & License**


.. image:: https://travis-ci.org/yzhao062/suod.svg?branch=master
   :target: https://travis-ci.org/yzhao062/suod
   :alt: Build Status


.. image:: https://circleci.com/gh/yzhao062/SUOD.svg?style=svg
   :target: https://circleci.com/gh/yzhao062/SUOD
   :alt: Circle CI


.. image:: https://ci.appveyor.com/api/projects/status/5kp8psvntp5m1d6m/branch/master?svg=true
   :target: https://ci.appveyor.com/project/yzhao062/combo/branch/master
   :alt: Appveyor


.. image:: https://coveralls.io/repos/github/yzhao062/SUOD/badge.svg
   :target: https://coveralls.io/github/yzhao062/SUOD
   :alt: Coverage Status

.. image:: https://img.shields.io/github/license/yzhao062/suod.svg
   :target: https://github.com/yzhao062/suod/blob/master/LICENSE
   :alt: License


----


**SUOD** (**S**\calable **U**\nsupervised **O**\utlier **D**\etection) is an **acceleration framework for large-scale unsupervised heterogeneous outlier detector training and prediction**.
Outlier detection (OD) is a key data mining task for identifying abnormal objects from general samples with numerous high-stake applications including fraud detection and intrusion detection. Due to the lack of ground truth labels,
practitioners often have to build a large number of unsupervised models that are heterogeneous (i.e., different algorithms and hyperparameters) for further combination and analysis with ensemble learning, rather than relying on a single model.
However, this yields severe scalability issues on high-dimensional, large datasets.

**SUOD** is therefore proposed to address the challenge at three complementary levels: random projection (**data level**), pseudo-supervised approximation (**model level**), and balanced parallel scheduling (**evaluation level**).
As mentioned, the key focus is to **accelerate the training and prediction when a large number of anomaly detectors are presented**, while preserving the prediction capacity.
Since its inception in Jan 2019, SUOD has been successfully used in various academic researches and industry applications, include PyOD [#Zhao2019PyOD]_ and `IQVIA <https://www.iqvia.com/>`_ medical claim analysis.
It could be especially useful for outlier ensembles that rely on a large number of base estimators.


.. image:: https://raw.githubusercontent.com/yzhao062/SUOD/master/figs/system_overview.png
   :target: https://raw.githubusercontent.com/yzhao062/SUOD/master/figs/system_overview.png
   :alt: SUOD Flowchart

SUOD is featured for:

* **Unified APIs, detailed documentation, and examples** for the easy use.
* **Optimized performance with JIT and parallelization** when possible, using `numba <https://github.com/numba/numba>`_ and `joblib <https://github.com/joblib/joblib>`_.
* **Fully compatible with the models in PyOD**.
* **Customizable modules and flexible design**: each module may be turned on/off or totally replaced by custom functions.

----

**API Demo**\ :


   .. code-block:: python


       from suod.models.base import SUOD

       # initialize a set of base outlier detectors to train and predict on
       base_estimators = [
           LOF(n_neighbors=5, contamination=contamination),
           LOF(n_neighbors=15, contamination=contamination),
           LOF(n_neighbors=25, contamination=contamination),
           HBOS(contamination=contamination),
           PCA(contamination=contamination),
           OCSVM(contamination=contamination),
           KNN(n_neighbors=5, contamination=contamination),
           KNN(n_neighbors=15, contamination=contamination),
           KNN(n_neighbors=25, contamination=contamination)]

       # initialize a SUOD model with all features turned on
       model = SUOD(base_estimators=base_estimators, n_jobs=6,  # number of workers
                    rp_flag_global=True,  # global flag for random projection
                    bps_flag=True,  # global flag for balanced parallel scheduling
                    approx_flag_global=False,  # global flag for model approximation
                    contamination=contamination)

       model.fit(X_train)  # fit all models with X
       model.approximate(X_train)  # conduct model approximation if it is enabled
       predicted_labels = model.predict(X_test)  # predict labels
       predicted_scores = model.decision_function(X_test)  # predict scores
       predicted_probs = model.predict_proba(X_test)  # predict outlying probability

----

A preliminary version (`accepted at AAAI-20 Security Workshop <http://aics.site/AICS2020/>`_) can be accessed on `arxiv <https://www.andrew.cmu.edu/user/yuezhao2/papers/20-preprint-suod.pdf>`_.
The extended version (under submission at a major ML system conference) can be accessed `here <http://www.andrew.cmu.edu/user/yuezhao2/papers/20-preprint-suod.pdf>`_.


If you use SUOD in a scientific publication, we would appreciate citations to the following paper::

    @inproceedings{zhao2020suod,
      author  = {Zhao, Yue and Ding, Xueying and Yang, Jianing and Haoping Bai},
      title   = {{SUOD}: Toward Scalable Unsupervised Outlier Detection},
      journal = {Workshops at the Thirty-Fourth AAAI Conference on Artificial Intelligence},
      year    = {2020}
    }

::

    Yue Zhao, Xueying Ding, Jianing Yang, Haoping Bai, "Toward Scalable Unsupervised Outlier Detection". Workshops at the Thirty-Fourth AAAI Conference on Artificial Intelligence, 2020.


**Table of Contents**\ :


* `Installation <#installation>`_
* `API Cheatsheet & Reference <#api-cheatsheet--reference>`_
* `Examples <#examples>`_


------------


Installation
^^^^^^^^^^^^

It is recommended to use **pip** for installation. Please make sure
**the latest version** is installed, as suod is updated frequently:

.. code-block:: bash

   pip install suod            # normal install
   pip install --upgrade suod  # or update if needed
   pip install --pre suod      # or include pre-release version for new features

Alternatively, you could clone and run setup.py file:

.. code-block:: bash

   git clone https://github.com/yzhao062/suod.git
   cd suod
   pip install .


**Required Dependencies**\ :


* Python 3.5, 3.6, or 3.7
* joblib
* numpy>=1.13
* pandas (**optional for building the cost forecast model**)
* pyod
* scipy>=0.19.1
* scikit_learn>=0.19.1


**Note on Python 2**\ :
The maintenance of Python 2.7 will be stopped by January 1, 2020 (see `official announcement <https://github.com/python/devguide/pull/344>`_).
To be consistent with the Python change and suod's dependent libraries, e.g., scikit-learn,
**SUOD only supports Python 3.5+** and we encourage you to use
Python 3.5 or newer for the latest functions and bug fixes. More information can
be found at `Moving to require Python 3 <https://python3statement.org/>`_.


------------


API Cheatsheet & Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^

Full API Reference: (https://suod.readthedocs.io/en/latest/api.html).

* **fit(X, y)**\ : Fit estimator. y is optional for unsupervised methods.
* **approximate(X)**\ : Use supervised models to approximate unsupervised base detectors. Fit should be invoked first.
* **predict(X)**\ : Predict on a particular sample once the estimator is fitted.
* **predict_proba(X)**\ : Predict the probability of a sample belonging to each class once the estimator is fitted.


Examples
^^^^^^^^

All three modules can be **executed separately** and the demo codes are in /examples/module_examples/{M1_RP, M2_BPS, and M3_PSA}.
For instance, you could navigate to /M1_RP/demo_random_projection.py. Demo codes all start with "demo_*.py".

**The examples for the full framework** can be found under /examples folder; run "demo_base.py" for
a simplified example. Run "demo_full.py" for a full example.

It is noted the best performance may be achieved with multiple cores available.

------------


**More to come...**
Last updated on Oct 11th, 2020.

Feel free to star for the future update :)

----

References
----------

.. [#Johnson1984Extensions] Johnson, W.B. and Lindenstrauss, J., 1984. Extensions of Lipschitz mappings into a Hilbert space. *Contemporary mathematics*, 26(189-206), p.1.

.. [#Zhao2019PyOD] Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. *Journal of Machine Learning Research*, 20, pp.1-7.
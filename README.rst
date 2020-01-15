SUOD: An Acceleration System for Large Scale Unsupervised Anomaly Detection
===========================================================================

**Deployment & Documentation & Stats**

.. image:: https://img.shields.io/pypi/v/suod.svg?color=brightgreen
   :target: https://pypi.org/project/suod/
   :alt: PyPI version

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


.. image:: https://coveralls.io/repos/github/yzhao062/SUOD/badge.svg
   :target: https://coveralls.io/github/yzhao062/SUOD
   :alt: Coverage Status


----


**SUOD** (Toward **S**\calable **U**\nsupervised **O**\utlier **D**\etection) is an **acceleration framework for large scale unsupervised outlier detector training and prediction**.
Notably, anomaly detection is often formulated as an unsupervised problem since the ground truth is expensive to acquire.
As a result, analysts often build many diversified models and further combine them (sometimes with rule-based models)---this has become a standard process in many industries to 
offset the challenges of the data imbalance and unsupervised nature. However, **building a large number of unsupervised models are very costly or even infeasible on high-dimensional, large datasets**.

SUOD is therefore proposed to alleviate, if not fully fix, this problem.
The focus of SUOD is **to accelerate the training and prediction when a large number of anomaly detectors are presented**.


**API Demo**\ :


   .. code-block:: python


       from suod.models.base import SUOD

       # initialize a set of base outlier detectors to train and predict on
       base_estimators = [
           LOF(n_neighbors=5, contamination=contamination),
           LOF(n_neighbors=15, contamination=contamination),
           LOF(n_neighbors=25, contamination=contamination),
           LOF(n_neighbors=35, contamination=contamination),
           LOF(n_neighbors=45, contamination=contamination),
           HBOS(contamination=contamination),
           PCA(contamination=contamination),
           OCSVM(contamination=contamination),
           KNN(n_neighbors=5, contamination=contamination),
           KNN(n_neighbors=15, contamination=contamination),
           KNN(n_neighbors=25, contamination=contamination),
           KNN(n_neighbors=35, contamination=contamination)]

       # initialize a SUOD model with all features turned on
       model = SUOD(base_estimators=base_estimators,
                    n_jobs=6, bps_flag=True,
                    contamination=contamination, approx_flag_global=False)

       model.fit(X_train)  # fit all models with X
       model.approximate(X_train)  # conduct model approximation if it is enabled
       predicted_labels = model.predict(X_test)  # predict labels
       predicted_scores = model.decision_function(X_test)  # predict scores


----


If you use SUOD in a scientific publication, we would appreciate citations to the following paper::

    @inproceedings{zhao2020suod,
      author  = {Zhao, Yue and Ding, Xueying and Yang, Jianing and Haoping Bai},
      title   = {{SUOD}: Toward Scalable Unsupervised Outlier Detection},
      journal = {Workshops at the Thirty-Fourth AAAI Conference on Artificial Intelligence},
      year    = {2020}
    }

::

    Yue Zhao, Xueying Ding, Jianing Yang, Haoping Bai, "Toward Scalable Unsupervised Outlier Detection". Workshops at the Thirty-Fourth AAAI Conference on Artificial Intelligence, 2020.


A preliminary version of paper can be accessed `here <https://www.andrew.cmu.edu/user/yuezhao2/papers/20-preprint-suod.pdf>`_.
The revised and extended version will be submitted to `KDD 2020 (ADS track) <https://www.kdd.org/kdd2020/>`_.
[`Preprint <https://www.andrew.cmu.edu/user/yuezhao2/papers/20-preprint-suod.pdf>`_], [`slides <https://www.andrew.cmu.edu/user/yuezhao2/misc/10715-SUOD-Toward-Scalable-Unsupervised-Outlier-Detection.pdf>`_], [`AICS <http://aics.site/AICS2020/>`_]


**Table of Contents**\ :


* `Reproduction Instructions <#reproduction-instructions>`_
* `Installation <#installation>`_

------------

Reproduction Instructions
^^^^^^^^^^^^^^^^^^^^^^^^^

All three modules can be **executed separately** and the demo codes are in /AAAI_Workshop/{M1_RP, M2_BPS, and M3_PSA}.
For instance, you could navigate to /M1_RP/demo_random_projection.py. Demo codes all start with "demo_*.py".

**A full example may be found in demo_full.py under the root directory.**

**Examples** can be found under /examples folder; run "demo_base.py" for
a simplified example. Run "demo_full.py" for a full example.

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
* matplotlib (**optional for running examples**)
* numpy>=1.13
* numba>=0.35
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


----


**More to come...**
Last updated on Jan 14th, 2020.

Feel free to star for the future update :)

----

References
----------

.. [#Johnson1984Extensions] Johnson, W.B. and Lindenstrauss, J., 1984. Extensions of Lipschitz mappings into a Hilbert space. *Contemporary mathematics*, 26(189-206), p.1.

.. SUOD documentation master file, created by
   sphinx-quickstart on Sat Feb 15 17:15:28 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SUOD's documentation!
================================


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

**Background**: Outlier detection (OD) is a key data mining task for identifying abnormal objects from general samples with numerous high-stake applications including fraud detection and intrusion detection.
Due to the lack of ground truth labels, practitioners often have to build a large number of unsupervised models that are heterogeneous (i.e., different algorithms and hyperparameters) for further combination and analysis with ensemble learning, rather than relying on a single model.
However, **this yields severe scalability issues on high-dimensional, large datasets**.

**SUOD** (**S**\calable **U**\nsupervised **O**\utlier **D**\etection) is an **acceleration framework for large-scale unsupervised heterogeneous outlier detector training and prediction**.
It focuses on three complementary aspects to accelerate (dimensionality reduction for high-dimensional data, model approximation for complex models, and execution efficiency improvement for taskload imbalance within distributed systems), while controlling detection performance degradation.

Since its inception in Sep 2019, SUOD has been successfully used in various academic researches and industry applications with more than 700,000 downloads, including PyOD :cite:`zhao2019pyod` and `IQVIA <https://www.iqvia.com/>`_ medical claim analysis.


.. figure:: https://raw.githubusercontent.com/yzhao062/SUOD/master/figs/system_overview.png
    :alt: SUOD Flowchart

SUOD is featured for:

* **Unified APIs, detailed documentation, and examples** for the easy use.
* **Optimized performance with JIT and parallelization** when possible, using `numba <https://github.com/numba/numba>`_ and `joblib <https://github.com/joblib/joblib>`_.
* **Fully compatible with the models in PyOD**.
* **Customizable modules and flexible design**: each module may be turned on/off or totally replaced by custom functions.


Roadmap:

* Provide more choices of distributed schedulers (adapted for SUOD), e.g., batch sampling, Sparrow (SOSP'13), Pigeon (SoCC'19) etc.
* Enable the flexibility of selecting data projection methods.

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

`The corresponding paper <https://www.andrew.cmu.edu/user/yuezhao2/papers/21-mlsys-suod.pdf>`_ is published in Conference on Machine Learning Systems (MLSys).
See https://mlsys.org/ for more information.


If you use SUOD in a scientific publication, we would appreciate citations to the following paper::


    @inproceedings{zhao2021suod,
      title={SUOD: Accelerating Large-scale Unsupervised Heterogeneous Outlier Detection},
      author={Zhao, Yue and Hu, Xiyang and Cheng, Cheng and Wang, Cong and Wan, Changlin and Wang, Wen and Yang, Jianing and Bai, Haoping and Li, Zheng and Xiao, Cao and others},
      journal={Conference on Machine Learning Systems (MLSys)},
      year={2021}
    }

::

    Zhao, Y., Hu, X., Cheng, C., Wang, C., Wan, C., Wang, W., Yang, J., Bai, H., Li, Z., Xiao, C. and Wang, Y., 2021. SUOD: Accelerating Large-scale Unsupervised Heterogeneous Outlier Detection. Conference on Machine Learning Systems (MLSys).


----


.. toctree::
   :maxdepth: 2
   :caption: Contents: Getting Started

   install
   example
   model_persistence


.. toctree::
   :maxdepth: 2
   :caption: Contents: Documentation

   api_cc
   api


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   about
   faq
   whats_new


----

.. rubric:: References

.. bibliography::
   :cited:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

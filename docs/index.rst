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


.. image:: https://coveralls.io/repos/github/yzhao062/SUOD/badge.svg
   :target: https://coveralls.io/github/yzhao062/SUOD
   :alt: Coverage Status


----

**SUOD** (**S**\calable **U**\nsupervised **O**\utlier **D**\etection) is an **acceleration framework for large-scale unsupervised outlier detector training and prediction**.
Notably, anomaly detection is often formulated as an unsupervised problem since the ground truth is expensive to acquire.
To compensate for the unstable nature of unsupervised algorithms, practitioners often build a large number of models for further combination and analysis, e.g., taking the average or majority vote.
**However, this poses scalability challenges in high-dimensional, large datasets**, especially for proximity-base models operating in Euclidean space.

**SUOD** is therefore proposed to address the challenge at three complementary levels:  random projection (**data level**), pseudo-supervised approximation (**model level**), and balanced parallel scheduling (**system level**).
As mentioned, the key focus is to **accelerate the training and prediction when a large number of anomaly detectors are presented**, while preserving the prediction capacity.
Since its inception in Jan 2019, SUOD has been successfully used in various academic researches and industry applications, include PyOD :cite:`a-zhao2019pyod` and `IQVIA <https://www.iqvia.com/>`_ medical claim analysis.


SUOD is featured for:

* **Unified APIs, detailed documentation, and examples** for the easy use.
* **Optimized performance with JIT and parallelization** when possible, using `numba <https://github.com/numba/numba>`_ and `joblib <https://github.com/joblib/joblib>`_.
* **Fully compatible with the models in PyOD**.
* **Customizable modules and flexible design**: each module may be turned on/off or totally replaced by custom functions.


----


.. toctree::
   :maxdepth: 2
   :caption: Contents:


----

.. rubric:: References

.. bibliography:: zreferences.bib
   :cited:
   :labelprefix: A
   :keyprefix: a-



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

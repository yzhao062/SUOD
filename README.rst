SUOD: An Acceleration System for Large Scale Unsupervised Anomaly Detection
===========================================================================

**Deployment & Documentation & Stats**

.. image:: https://img.shields.io/pypi/v/suod.svg?color=brightgreen
   :target: https://pypi.org/project/suod/
   :alt: PyPI version

----


**SUOD** (Toward **S**\calable **U**\nsupervised **O**\utlier **D**\etection) is an **acceleration framework for large scale unsupervised outlier detector training and prediction**.
Notably, anomaly detection is often formulated as an unsupervised problem since the ground truth is expensive to acquire in practice.
As a result, analysts often build many diversified models and further combine them (sometimes with rule-based models)---this has become a standard process in many industries to 
offset the challenges of the data imbalance and unsupervised nature. However, **building a large number of unsupervised models are very costly or even infeasible on high-dimensional, large datasets**.

SUOD is therefore proposed to alleviate, if not fully fix, this problem.
The focus of SUOD is **to accelerate the training and prediction when a large number of anomaly detectors are presented**.


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


A preliminary version of paper can be accessed `here <https://www.andrew.cmu.edu/user/yuezhao2/papers/20-preprint-suod.pdf>`_. The revised and extended version will be submitted to `KDD 2020 (ADS track) <https://www.kdd.org/kdd2020/>`_

[`Preprint <https://www.andrew.cmu.edu/user/yuezhao2/papers/20-preprint-suod.pdf>`_], [`slides <https://www.andrew.cmu.edu/user/yuezhao2/misc/10715-SUOD-Toward-Scalable-Unsupervised-Outlier-Detection.pdf>`_], [`AICS <http://aics.site/AICS2020/>`_]

------------

Reproduction Instructions
^^^^^^^^^^^^^^^^^^^^^^^^^

All three modules can be **executed separately** and the demo codes are in /AAAI_Workshop/{M1_RP, M2_BPS, and M3_PSA}.
For instance, you could navigate to /M1_RP/demo_random_projection.py. Demo codes all start with "demo_*.py".

**A full example may be found in demo_full.py under the root directory.**

**Production level code will be released soon---it will support PyPI installation with full documentation and example!**

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
* pyod
* scipy>=0.19.1
* scikit_learn>=0.19.1


**Note on Python 2**\ :
The maintenance of Python 2.7 will be stopped by January 1, 2020 (see `official announcement <https://github.com/python/devguide/pull/344>`_).
To be consistent with the Python change and suod's dependent libraries, e.g., scikit-learn,
**suod only supports Python 3.5+** and we encourage you to use
Python 3.5 or newer for the latest functions and bug fixes. More information can
be found at `Moving to require Python 3 <https://python3statement.org/>`_.


----


**More to come...**
Last updated on Dec 23, 2019.

Feel free to star for the future update :)

----

References
----------

.. [#Johnson1984Extensions] Johnson, W.B. and Lindenstrauss, J., 1984. Extensions of Lipschitz mappings into a Hilbert space. *Contemporary mathematics*, 26(189-206), p.1.

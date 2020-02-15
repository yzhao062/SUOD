Overview
--------

**SUOD** (Toward **S**\calable **U**\nsupervised **O**\utlier **D**\etection): an acceleration framework for large scale unsupervised outlier detector training and prediction.

**Key question**: why are we doing this?

The short answer is that anomaly detection is often formulated as an unsupervised problem since the ground truth is expensive to acquire in practice.
Additionally, anomaly detection is an imbalanced learning task, which further complicates label acquisition.
Practically speaking, analysts are inclined to build many diversified models and further combine them (sometimes with rule-based models)---this has become a standard process in risk industry,
e.g., banks and insurance firms. However, **building a large number of unsupervised models are very costly or even infeasible on high-dimensional, large datasets**.

SUOD is therefore proposed to alleviate, if not fully fix, this problem.
The focus of SUOD is **to accelerate the training and prediction while a large number of anomaly detectors are presented**.


----

A preliminary version of paper can be accessed `here <https://www.andrew.cmu.edu/user/yuezhao2/papers/20-preprint-suod.pdf>`_. A scalable python implementation will be released shortly.
The revised and extended version will be submitted to `KDD 2020 (ADS track) <https://www.kdd.org/kdd2020/>`_

If you use SUOD in a scientific publication, we would appreciate
citations to the following paper::

    @inproceedings{zhao2020suod,
      author  = {Zhao, Yue and Ding, Xueying and Yang, Jianing and Haoping Bai},
      title   = {{SUOD}: Toward Scalable Unsupervised Outlier Detection},
      journal = {Workshops at the Thirty-Fourth AAAI Conference on Artificial Intelligence},
      year    = {2020}
    }

::

    Yue Zhao, Xueying Ding, Jianing Yang, Haoping Bai, "Toward Scalable Unsupervised Outlier Detection". Workshops at the Thirty-Fourth AAAI Conference on Artificial Intelligence, 2020.



------------

Reproduction Instructions
^^^^^^^^^^^^^^^^^^^^^^^^^

All three modules can be **executed separately** and the demo codes are in /AAAI_Workshop/{M1_RP, M2_BPS, and M3_PSA}.
For instance, you could navigate to /M1_RP/demo_random_projection.py. Demo codes all start with "demo_*.py".

**Production level code will be released soon---it will support PyPI installation with full documentation and example!**

------------

Abstract
--------

Outlier detection is a key field of machine learning for identifying abnormal data objects.
Due to the high expense of acquiring ground truth, unsupervised models are often chosen in practice.
To compensate for the unstable nature of unsupervised algorithms, practitioners from high-stakes fields like finance, health, and security, prefer to build a large number of models for further combination and analysis.
However, this poses scalability challenges in high-dimensional large datasets.
In this study, we propose a three-module acceleration framework called SUOD to expedite the training and prediction with a large number of unsupervised detection models.
SUOD's Random Projection module can generate lower subspaces for high-dimensional datasets while reserving their distance relationship.
Balanced Parallel Scheduling module can forecast the training and prediction cost of models with high confidence---so the task scheduler could assign nearly equal amount of taskload among workers for efficient parallelization.
SUOD also comes with a Pseudo-supervised Approximation module, which can approximate fitted unsupervised models by lower time complexity supervised regressors for fast prediction on unseen data.
It may be considered as an unsupervised model knowledge distillation process. Notably, all three modules are independent with great flexibility to "mix and match";
a combination of modules can be chosen based on use cases. Extensive experiments on more than 30 benchmark datasets have shown the efficacy of SUOD, and a comprehensive future development plan is also presented.


System Overview
---------------

See the basic flowchart below for clarification. **SUOD** is a three-module acceleration framework for training and predicting with a large number of unsupervised outlier detectors.
Not all the modules are needed all the time---you may consider it as a LEGO system with great flexibility! A more formal algorithm description is also provided below.

.. image:: https://raw.githubusercontent.com/yzhao062/SUOD/master/figs/basic_framework.png
   :target: https://raw.githubusercontent.com/yzhao062/SUOD/master/figs/basic_framework.png
   :alt: SUOD Flowchart


.. image:: https://raw.githubusercontent.com/yzhao062/SUOD/master/figs/algorithm-suod.png
   :target: https://raw.githubusercontent.com/yzhao062/SUOD/master/figs/algorithm-suod.png
   :alt: SUOD Algorithm Design


Module I: Random Projection
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A widely used algorithm to alleviate the curse of dimensionality on high-dimensional data is the Johnson-Lindenstraus (JL) projection [#Johnson1984Extensions]_,
although its use in outlier mining is still unexplored. JL projection is a simple compression scheme without heavy distortion on the Euclidean distances of the data.
Its built-in randomness is also useful for inducing diversity for outlier ensembles.
Despite, projection may be less useful or even detrimental for methods like Isolation Forests and HBOS that rely on subspace splitting.
Detailed proof and four JL variations (*basic*, *discrete*, *circulant*, and *toeplitz*) can be found in the paper.

Module II: Balanced Parallel Scheduling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. image:: https://raw.githubusercontent.com/yzhao062/SUOD/master/figs/flowchart-suod.png
   :target: https://raw.githubusercontent.com/yzhao062/SUOD/master/figs/flowchart-suod.png
   :alt: Flowchart of Balanced Parallel Scheduling

Module III: Pseudo-Supervised Approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. image:: https://raw.githubusercontent.com/yzhao062/SUOD/master/figs/ALL.png
   :target: https://raw.githubusercontent.com/yzhao062/SUOD/master/figs/ALL.png
   :alt: Comparison among unsupervised models and their pseudo-supervised counterparts

------------

**More to come...**
Last updated on Dec 17, 2019.

Feel free to star for the future update :)

----

References
----------

.. [#Johnson1984Extensions] Johnson, W.B. and Lindenstrauss, J., 1984. Extensions of Lipschitz mappings into a Hilbert space. *Contemporary mathematics*, 26(189-206), p.1.

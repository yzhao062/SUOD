Supplementary material for **SUOD** (Toward **S**\calable **U**\nsupervised **O**\utlier **D**\etection); **submitted, under review**.
A preliminary version of paper can be accessed `here <https://www.andrew.cmu.edu/user/yuezhao2/papers/19-preprint-suod.pdf>`_.
A scalable python implementation will be released shortly. 

::

    Yue Zhao, Xueying Ding, Jianing Yang, Haoping Bai, "Toward Scalable Unsupervised Anomaly Detection". 
    Thirty-Fourth AAAI Conference on Artificial Intelligence Workshop, 2020. 


See the basic flowchart below for clarification. **SUOD** is a three-module acceleration framework for training and inferencing with a large number of unsupervised outlier detectors. Not all the modules are needed all the time---you may consider it as a LEGO system with great flexibility!

.. image:: https://raw.githubusercontent.com/yzhao062/SUOD/master/figs/basic_framework.png
   :target: https://raw.githubusercontent.com/yzhao062/SUOD/master/figs/basic_framework.png
   :alt: SUOD Flowchart

------------

Abstract
--------

Outlier detection is a key field of machine learning for identifying abnormal data objects. Due to the high expense of acquiring ground truth, unsupervised models are often chosen in practice. To compensate for the unstable nature of unsupervised algorithms, practitioners from high-stakes fields like finance, health, and security, prefer to build a large number of models for further combination and analysis. However, this poses scalability challenges in high-dimensional large datasets. In this study, we propose a three-module acceleration framework called SUOD to expedite the training and prediction with a large number of unsupervised detection models. SUOD's Random Projection module can generate lower subspaces for high-dimensional datasets while reserving their distance relationship. Balanced Parallel Scheduling module can forecast the training and prediction cost of models with high confidence---so the task scheduler could assign nearly equal amount of taskload among workers for efficient parallelization. SUOD also comes with a Pseudo-supervised Approximation module, which can approximate fitted unsupervised models by lower time complexity supervised regressors for fast prediction on unseen data. It may be considered as an unsupervised model knowledge distillation process. Notably, all three modules are independent with great flexibility to "mix and match"; a combination of modules can be chosen based on use cases. Extensive experiments on more than 30 benchmark datasets have shown the efficacy of SUOD, and a comprehensive future development plan is also presented. 


------------

Module I: Random Projection
---------------------------

A widely used algorithm to alleviate the curse of dimensionality on high-dimensional data is the Johnson-Lindenstraus (JL) projection [#Johnson1984Extensions]_,
although its use in outlier mining is still unexplored. JL projection is a simple compression scheme without heavy distortion on the Euclidean distances of the data. 
Its built-in randomness is also useful for inducing diversity for outlier ensembles. 
Despite, projection may be less useful or even detrimental for methods like Isolation Forests and HBOS that rely on subspace splitting. 
Detailed proof and four JL variations (*basic*, *discrete*, *circulant*, and *toeplitz*) can be found in the paper.

Module II: Balanced Parallel Scheduling
---------------------------------------


.. image:: https://raw.githubusercontent.com/yzhao062/SUOD/master/figs/flowchart-suod.png
   :target: https://raw.githubusercontent.com/yzhao062/SUOD/master/figs/flowchart-suod.png
   :alt: Flowchart of Balanced Parallel Scheduling

Module III: Pseudo-Supervised Approximation
-------------------------------------------


.. image:: https://raw.githubusercontent.com/yzhao062/SUOD/master/figs/ALL.png
   :target: https://raw.githubusercontent.com/yzhao062/SUOD/master/figs/ALL.png
   :alt: Comparison among unsupervised models and their pseudo-supervised counterparts

------------

**More to come...**
Last updated on Dec 12, 2019.

Feel free to star for the future update :)

----

References
----------

.. [#Johnson1984Extensions] Johnson, W.B. and Lindenstrauss, J., 1984. Extensions of Lipschitz mappings into a Hilbert space. *Contemporary mathematics*, 26(189-206), p.1.

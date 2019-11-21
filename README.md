# SUOD (Scalable Unsupervised Outlier Detection)

### Supplementary material for "Toward Scalable Unsupervised Anomaly Detection". 
Both demo code and production-level implementation will be released soon.

------------
Yue Zhao, Xueying Ding, Jianing Yang, Haoping Bai, "Toward Scalable Unsupervised Anomaly Detection". 
*Thirty-Fourth AAAI Conference on Artificial Intelligence Workshop*, 2020. **Submitted, under review**.


## Abstract

Outlier detection is a key field of machine learning for identifying abnormal data objects, and has been widely used in many real-world applications. Due to the high expense of acquiring the ground truth in practice, unsupervised models are often chosen. To compensate for the unstable nature of unsupervised algorithms, practitioners prefer to build a large number of models for further combination and analysis, which is common in fields like security, health, and finance. However, this poses scalability challenges, especially in high-dimensional, large datasets. In this study, we propose a three-module framework called ***SUOD*** to expedite the training and inference when a large number of unsupervised detection models are presented. ***SUOD***'s Random Projection module can generate lower subspaces for high-dimensional datasets while reserving their distance relationship. Balanced Parallel Scheduling module can forecast the training and inference cost of models with high confidence---so the scheduler could assign a nearly equal amount of taskload to workers for efficient parallelization. ***SUOD*** also comes with Pseudo-supervised Approximation module, which can approximate fitted unsupervised models by supervised regressors. This approximation leads to faster inference speed and smaller storage cost; it can also be considered as a way of distilling knowledge. It is noted that all three modules are independent with great flexibility to "mix and match", so a combination of modules can be chosen given different situations. Extensive experiments on more than 30 benchmark datasets have shown the efficacy of the modules; a comprehensive future development plan is also presented. 

------------

### Module I: Random Projection

TBA

### Module II: Balanced Parallel Scheduling 

![BPS Flowchart](https://github.com/yzhao062/SUOD/blob/master/figs/flowchart-suod.png "Phase I Flowchart")
Fig 1. Flowchart of Balanced Parallel Scheduling

### Module III: Pseudo-Supervised Approximation

![All Comparision](https://github.com/yzhao062/SUOD/blob/master/figs/ALL.png "Approximation Comparison")
Fig 2. Comparison among unsupervised models and their pseudo-supervised counterparts

------------

**More to come...**
Last updated on Nov 20, 2019.

Feel free to star for the future update :)

# SUOD (Scalable Unsupervised Outlier Detection)

### Supplementary materials: datasets, demo source codes and sample outputs.

------------
Yue Zhao, Xueying Ding, Jianing Yang, Haoping Bai, "Toward Scalable Unsupervised Anomaly Detection". 
*Thirty-Fourth AAAI Conference on Artificial Intelligence Workshop*, 2020. **Submitted, under review**.


## Abstract

Outlier detection is a key field of machine learning for identifying abnormal data objects, and has been widely used in other fields such as security. Due to the high expense of acquiring the ground truth, unsupervised models are often chosen in practice. To compensate for the unstable nature of unsupervised models, practitioners prefer to build a large number of models for further combination and analysis. 
This poses scalability challenges in high-dimensional large datasets. In this study, we propose a three-module framework called ***SUOD*** to expedite the training and inference when a large number of unsupervised detection models are presented. ***SUOD***'s Random Projection module can generate random lower subspaces for high-dimensional datasets while reserving their distance relationship. 
Balanced Parallel Scheduling module can forecast the training and inference cost of models with high accuracy---so the scheduler could assign a nearly equal amount of task load to multiple workers for parallelization. ***SUOD*** also comes with Pseudo-supervised Approximation module, which can approximate fitted unsupervised models by supervised regressors. This approximation leads to faster inference speed and smaller storage cost; 
it can also be considered as a way of knowledge distillation. It is noted that these three modules are independent with great flexibility to "mix and match" so a combination of modules can be chosen for different use cases. Extensive experiments on 20 benchmark datasets have shown the efficacy of the modules; a comprehensive future development plan is also presented. 


**More to come...**
Last updated on Nov 19, 2019.
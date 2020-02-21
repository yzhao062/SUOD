API Reference
=============

The following APIs are the key ones for using SUOD:

* :func:`suod.models.base.SUOD.fit`: Fit estimator. y is optional for unsupervised methods.
* :func:`suod.models.base.SUOD.predict`: Predict on a particular sample once the estimator is fitted.


* :func:`combo.models.base.SUOD.get_params`: Get the parameters of the model.
* :func:`combo.models.base.SUOD.set_params`: Set the parameters of the model.
* Each base estimator can be accessed by calling clf[i] where i is the estimator index.

----


All Models
^^^^^^^^^^

.. toctree::
   :maxdepth: 4

   suod.models
   suod.utils



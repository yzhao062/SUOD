Model Save & Load
=================

SUOD takes a similar approach of sklearn regarding model persistence.
See `model persistence <https://scikit-learn.org/stable/modules/model_persistence.html>`_ for clarification.

In short, we recommend to use joblib or pickle for saving and loading SUOD models.
See `"examples/demo_model_save_load.py" <https://github.com/yzhao062/suod/blob/master/examples/demo_model_save_load.py>`_ for an example.
In short, it is simple as below:

.. code-block:: python

    from joblib import dump, load

    # save the fitted model
    dump(model, 'model.joblib')
    # load the model
    model = load('model.joblib')



Examples
========

All three modules can be **executed separately** and the demo codes are in /example/module_demo/:

* M1_RP: demo_random_projection.py
* M2_PSA: demo_pseudo_sup_approximation.py
* M3_BPS: demo_balanced_scheduling.py

For instance, you could navigate to /M1_RP/demo_random_projection.py. Demo codes all start with "demo_*.py".

**The examples for the full framework** can be found under /examples folder; run "demo_base.py" for
a simplified example. Run "demo_full.py" for a full example.


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

Installation
============

It is recommended to use **pip** for installation. Please make sure
**the latest version** is installed, as SUOD is updated frequently:

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


* Python 3.8+
* joblib
* numpy>=1.13
* pandas (**optional for building the cost forecast model**)
* pyod
* scipy>=0.19.1
* scikit_learn>=1.0

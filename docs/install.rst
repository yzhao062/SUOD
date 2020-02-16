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


* Python 3.5, 3.6, or 3.7
* joblib
* numpy>=1.13
* pandas (**optional for building the cost forecast model**)
* pyod
* scipy>=0.19.1
* scikit_learn>=0.19.1


**Note on Python 2**\ :
The maintenance of Python 2.7 has stopped since January 1, 2020 (see `official announcement <https://github.com/python/devguide/pull/344>`_).
To be consistent with the Python change and SUOD's dependent libraries, e.g., scikit-learn,
**SUOD only supports Python 3.5+** and we encourage you to use
Python 3.5 or newer for the latest functions and bug fixes. More information can
be found at `Moving to require Python 3 <https://python3statement.org/>`_.
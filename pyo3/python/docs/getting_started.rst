Getting started
===============


Installation
------------
The process is similar to installing other Python packages.
The Python version of boruta-fru is available as ``boruta-fru`` on PyPI.
Use your favorite package manager to install it.
The package requires ``python >= 3.12``.

For this tutorial, you will also need ``scikit-learn``, ``pandas`` and ``polars``.

Basic usage example
-------------------
You can initialize a model and use the ``fit`` and ``final_decision`` functions.
The fit function is used to perform feature selection. After running ``fit``, the ``final_decision``
function returns a decision for each feature. The model accepts a ``max_runs`` parameter during initialization.
If there are many ``Tentative`` decisions, increasing this parameter may help.
Additionally, increasing the number of ``trees`` may improve the quality of the decisions.

.. warning::
    Pandas fully supports the PyCapsule interface starting from version 3.
    We support pandas version 3 and above.

.. warning::
    The target must be a categorical series with string categories or numerical series.
	Passing NumPy arrays is not supported. If you are using NumPy arrays, convert
	them to ``pandas.DataFrame`` and ``pandas.Series``.


.. code-block:: python

    from sklearn.datasets import load_breast_cancer
    from boruta_fru import Boruta
    
    # load data
    data = load_breast_cancer(as_frame=True)
	
    # convert target to a categorical series with string categories
    y = data["target"].astype(str).astype("category")
	
    # create model instance
    boruta = Boruta(max_runs=200, trees=100)

    # fit model
    boruta.fit(data["data"], y)

    # make predictions
    boruta.final_decision()

The result of ``final_decision`` is an array containing a decision for each column in the original data frame.

- ``Confirmed`` means the feature was accepted as important for prediction.
- ``Rejected`` means the feature was determined not to be important for prediction.
- ``Tentative`` means the feature was neither rejected nor confirmed; therefore, it is unclear whether the feature should be considered important for prediction.


Polars example
--------------


Another feature of ``boruta-fru`` is that it works with libraries supporting the Arrow PyCapsule interface,
such as ``pandas``, ``polars``, ``pyarrow``, ``duckdb``, and others.

.. code-block:: python

    import polars as pl

    from sklearn.datasets import load_breast_cancer

    from boruta_fru import Boruta

    # load data	
    data = load_breast_cancer(as_frame=True)

    # create model instance
    x = pl.from_pandas(data["data"])
    y = pl.from_pandas(data["target"].astype(str).astype("category"))
    
    # create model instance
    boruta = Boruta(max_runs=200, trees=100)

    # fit model
    boruta.fit(x, y)

    # make predictions
    boruta.final_decision()


Missing values
--------------

``boruta-fru`` supports missing values. By default, in each iteration the algorithm
imputes missing entries by randomly sampling (with replacement) from the non-null
values in the same column.

To disable this behavior, set ``impute=False``.

.. code-block:: python

    import pandas as pd

    from sklearn.datasets import load_breast_cancer
    from boruta_fru import Boruta
    
    # load data
    data = load_breast_cancer(as_frame=True)
    x = data["data"]
    x.iloc[0, 0] = pd.NA
	
    # convert target to a categorical series with string categories
    y = data["target"].astype(str).astype("category")
	
    # create model instance
    boruta = Boruta(max_runs=200, trees=100)

    # fit model
    boruta.fit(data["data"], y)

    # make predictions
    boruta.final_decision()


Progress
--------
The package provides progress reporting via logging. After each iteration, a log
message is emitted containing the iteration number and the counts of
``Tentative``, ``Rejected``, and ``Confirmed`` features.To enable logging,
you just need to change log level to info.

.. code-block:: python

    import logging
    
    from sklearn.datasets import load_breast_cancer
    from boruta_fru import Boruta
    
    # load data
    data = load_breast_cancer(as_frame=True)
	
    # convert target to a categorical series with string categories
    y = data["target"].astype(str).astype("category")
	
    # create model instance
    boruta = Boruta(max_runs=200, trees=100)

    # configure logging
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    
    # fit model
    boruta.fit(data["data"], y)

    # make predictions
    boruta.final_decision()


Result as PyCapsule (optional)
------------------------------
Results can optionally be returned as an Arrow PyCapsule. This allows them to be
loaded into any data frame library supporting the Arrow PyCapsule interface.

.. code-block:: python

    import pandas as pd

    from sklearn.datasets import load_breast_cancer

    from boruta_fru import Boruta

    # load data
    data = load_breast_cancer(as_frame=True)
    X = data["data"]
    y = data["target"].astype(str).astype("category")
	
    # create model instance	
    boruta = Boruta(max_runs=200, trees=100)

    # fit model	
    boruta.fit(X, y)

    # convert results from PyCapsule	
    pd.DataFrame.from_arrow(boruta.final_decision(to_pycapsule=True))

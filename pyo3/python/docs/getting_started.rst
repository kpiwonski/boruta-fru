Getting started
===============


Installation
------------
The process is similar to installing other Python packages.
The Python version of boruta-fru is available as ``boruta-fru`` on PyPI.
Use your favorite package manager to install it.
The package requires ``python >= 3.12``.

For this tutorial, you will also need ``pandas`` and ``polars``.

Basic usage example
-------------------
You can initialize a model and use ``fit`` and ``final_decision`` functions.
``fit`` is used for performing a feature selection.
After performing ``fit``, ``final_decision`` reutrns decision for each feature.
The model takes ``max_runs`` parameter on initialization. If there are many
``Tentative`` decisions, increasing this parameter may help.
Moreover, increasing ``trees`` may be helpful to increase the quality of decisions. 

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

The result of ``final_decision`` will be an array with a decision for each column
from our original data frame.

- ``Confirmed`` means the feature was accepted as important for prediction
- ``Rejected`` means the feature was rejected being important for prediction
- ``Tentative`` means the feature was neither rejected nor confirmed. Hence, it is unknown if the feature should be considered as important for prediction.

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

import secrets
import sys

from boruta_fru.data_structures import ResultTable
from . import _rust

PVAL_TH = 0.01

class Boruta:
    """
    Boruta [1]_ model used for the feature selection.

    Parameters
    ----------
    max_runs : int
        Maximal number of the model iterations. If all features are resolved as
        confirmed or rejected, the model witll stop earlier. If there are
        tentative features, you may consider increasing this parameter.
    trees : int
        Number of trees to grow in the forest (often called ``ntree`` in other
        software). Must be greater than zero. The value should be large enough
        to provide stable results (prediction accuracy or importance). Larger
        datasets typically require more trees. Computation time grows linearly
        with the number of trees. Defaults to 500.
    tries : int | None
        Number of features to try at each split (often called ``mtry``).
        Must be greater than zero and less than or equal to the number of features.
        By default, it is set to the rounded square root of the number of features.
        Higher values increase correlation between trees. In most cases, the default
        setting is recommended.
    seed : int | None
        Seed used by the algorithm. Set to ``None`` to use a random seed.
    threads : int | None
        Number of threads to use. Must be greater than zero. If ``None``, all
        available CPU cores are used. Defaults to ``None``.

    Notes
    -----
    Boruta iteratively compares importances of attributes with importances of
    shadow attributes, created by shuffling original ones. Attributes that have
    significantly worst importance than shadow ones are being consecutively
    dropped. On the other hand, attributes that are significantly better than shadows
    are admitted to be Confirmed. Shadows are re-created in each iteration.
    Algorithm stops when only Confirmed attributes are left, or when it reaches
    max_runs importance source runs. If the second scenario occurs, some attributes
    may be left without a decision. They are claimed Tentative.

    References
    ----------
    .. [1] `Miron B. Kursa, Witold R. Rudnicki (2010).
       Feature Selection with the Boruta Package.
       Journal of Statistical Software, 36(11), p. 1-13.
       <https://doi.org/10.18637/jss.v036.i11>`_
    """
    def __init__(
        self,
        max_runs=100,
        trees=500,
        tries=None,
        seed=None,
        threads=None,
    ):
        self.max_runs = max_runs
        self.pval_th = PVAL_TH
        self.trees = trees
        self.tries = tries
        self.seed = seed
        self.threads = threads

    def fit(self, X, y):
        """
        Runs Boruta feature selection.

        Parameters
        ----------
        X : Arrow PyCapsule
            A DataFrame-like object supporting the Arrow PyCapsule interface.
            Any library supporting this interface can be used (e.g., pandas,
            polars). Columns can be boolean, numerical, or categorical. Mixed
            column types are allowed. Other data types are not supported and
            will raise an exception. ``NaN`` values are not allowed.
        y : Arrow PyCapsule
            A Series-like object supporting the Arrow PyCapsule interface.
            Any library supporting this interface can be used. For
            classification, ``y`` must be categorical, otherwise an exception
            is raised. ``NaN`` values are not allowed. The length of ``y`` must
            match the number of rows in ``X``.
        """
        y = self._validate_y(y)
        X = self._validate_x(X)
        self._res = _rust.boruta(
            X,
            y,
            self.max_runs,
            self.pval_th,
            self.trees,
            self.tries,
            self._get_seed(),
            self.threads,
        )

    def final_decision(self, to_pycapsule=False):
        """
        Final decision of the model. For each feature can be either:
        ``Confirmed``, ``Rejected``, or ``Tentative``.

        Parameters
        ----------
        to_pycapsule : bool
            If ``True``, results are returned as an Arrow PyCapsule. If ``False``, results
            are returned as NumPy arrays, similar to scikit-learn. Defaults to ``False``.
        """
        return ResultTable(self._res.final_decision(), to_pycapsule).get_table()

    def _get_seed(self):
        return self.seed if self.seed is not None else secrets.randbits(64)

    @staticmethod
    def _remove_pandas_rownames(obj):
        pd = sys.modules.get("pandas")

        # Remove pandas rownames, otherwise would be passed as __index_level_0__ to arrow
        if pd and isinstance(obj, pd.DataFrame):
            return obj.reset_index(drop=True)
        return obj

    @classmethod
    def _validate_x(cls, X):
        if not hasattr(X, "__arrow_c_stream__"):
            raise AttributeError("X must implement PyCapsule")
        X = cls._remove_pandas_rownames(X)
        return X

    @staticmethod
    def _validate_y(y):
        if not hasattr(y, "__arrow_c_stream__"):
            raise AttributeError("y must implement PyCapsule")
        return y

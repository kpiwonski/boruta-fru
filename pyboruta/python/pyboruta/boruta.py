import secrets
import sys

# from boruta.data_structures import ImportanceResultTable, ResultArray, ResultTable

from . import _rust

class Boruta:
    def __init__(
        self,
        max_runs=100,
        pval_th=0.05,
        trees=500,
        tries=None,
        seed=None,
        threads=None,
    ):
        self.max_runs = max_runs
        self.pval_th = pval_th
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

    @property
    def final_decision(self):
        return self._res.final_decision()

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

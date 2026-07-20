class ResultTable:
    IMPORTANCE_COL = "decision"

    def __init__(self, obj, to_pycapsule):
        if not hasattr(obj, "__arrow_c_stream__"):
            raise AttributeError("Object does not have arrow stream PyCapsule")

        self.obj = obj
        self.to_pycapsule = to_pycapsule

    def __arrow_c_stream__(self, requested_schema=None):
        return self.obj.__arrow_c_stream__(requested_schema)

    def get_table(self):
        if not self.to_pycapsule:
            return self._df_to_numpy()
        return self

    def _df_to_numpy(self):
        return self.obj[self.IMPORTANCE_COL].to_numpy(zero_copy_only=False)

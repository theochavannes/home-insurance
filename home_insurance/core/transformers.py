"""
This file contains transformers included in the machine learning pipeline
"""
from sklearn.base import BaseEstimator, TransformerMixin


class BinaryColumnsCleaner(BaseEstimator, TransformerMixin):
    """
    This class is used after a one hot encoder (aikit NumericalEncoder with "dummy" option).
    One hot encoding on binary categorical columns creates 2 columns
    ("Y" and "N" for example) but they contain the same information.
    This transformers delete one of the two columns created.
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        redundant_cols = []
        columns = list(X.columns)
        for col in X.columns:
            if "__N" in col:
                redundant_cols.append(col)

        set_cols_to_keep = set(columns).difference(redundant_cols)
        cols_to_keep = []
        for col in columns:
            if col in set_cols_to_keep:
                cols_to_keep.append(col)
        X = X[cols_to_keep]

        return X

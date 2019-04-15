import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from ...utils import check_has_columns, check_type



class fColumnSelector(BaseEstimator, TransformerMixin):
    """
    A transformer that selects columns from a data frame.
    a =
    [a b c]
    [1 2 3]
    [4 5 6]
    [7 8 9]

    >> fColumnSelector(columns=[a , b]).fit_transform(a)

    [a b]
    [1 2]
    [4 5]
    [7 8]
    """
    def __init__(self, columns='all', copy=True):

        self.columns = columns
        self.copy = copy

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        check_type(X, types=[pd.DataFrame])
        check_has_columns(X, self.columns)

        if self.copy == True:
            return X[self.columns].copy()
        else:
            return X[self.columns]



class fToArray(BaseEstimator, TransformerMixin):
    """
    Converts a pandas DataFrame to numpy array .

    checks for every dataframe passed to transform that it has all columns passed to fit.
    note :
        inverse transform transforms only the a numpy array representing a dataframe
        passed to fit.
        so for example:

        enc = fColumnSelector()

        OK :
            A = enc.fit_transform(X)
            X_ = enc.inverse_transform(A)

        NOT OK:
            A = enc.fit_transform(X)
            Y_ = enc.transform(Y)
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):

        check_type(X, types=[pd.DataFrame])

        self.columns_ = X.columns
        self.index_ = X.index

        return self

    def transform(self, X):

        check_type(X, types=[pd.DataFrame])
        check_has_columns(X, self.columns_)

        return X[self.columns_].values

    def inverse_transform(self, X):
        return pd.DataFrame(X , columns=self.columns_, index=self.index_)

def Workflow(order=None, **kwargs):

    if order is None:
        raise ValueError('Order must be given to allow correct flow of data.')

    for pipeline in order:
        if pipeline not in kwargs:
            raise ValueError('{} not given as key word argument.'.format(pipeline))

    pipelines = [(pipeline, kwargs[pipeline]) for pipeline in order]

    return Pipeline(pipelines)


def empty_DataFrame():
    return pd.DataFrame({})



import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from ...utils import check_type, check_has_columns


class fImputer(BaseEstimator, TransformerMixin):
    """
    sklearn style imputer for DataFrames and Series objects.
    """
    def __init__(self, strategy='mean', copy=True, na_sentinel=-1, columns=None):

        self.strategy = strategy
        self.copy = copy
        self.na_sentinel = na_sentinel
        self.columns = columns

    def fit(self, X, y=None):

        # ensuring validity of strategy
        strategy = self.strategy.lower()
        options = ['mean', 'median', 'most_frequent', 'na_sentinel']
        if strategy not in options:
            raise ValueError('strategy must be one of : ' + str(options))

        # checking X is either a Dataframe or Series
        check_type(X, types=[pd.DataFrame, pd.Series])

        # choose columns to impute
        if self.columns is None:
            self.columns_ = X.columns.tolist()
        else:
            self.columns_ = self.columns

        # choosing value/s to fillt Na with
        if strategy == 'mean':
            self.fill_ = X.mean()
        elif strategy == 'median':
            self.fill_ = X.median()
        elif strategy == 'most_frequent':
            self.fill_ = X.mode().iloc[0]
        else:
            self.fill_ = {column : self.na_sentinel for column in self.columns_}

        return self

    def transform(self, X):

        # check X has all columns to be imputed
        check_has_columns(X, self.columns_)

        # checking X is either a Dataframe or Series
        check_type(X, types=[pd.DataFrame, pd.Series])

        # getting imputed DtaFrame/Series
        result = X.fillna(self.fill_, inplace= not self.copy)

        if result is None:
            result = X

        return result




import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from ...utils import check_type, check_has_columns

def _nan_to_end(enc):
    '''
    :param enc: an fLabelEncoder instance
    :return: enc after moving nan in classes_ attribte to end of list
    '''

    # boolean saying if null present in data
    null_cmp = np.asarray(pd.isnull(enc.classes_)) | \
               np.asarray(list(map(lambda label: str(label).lower() == 'nan', enc.classes_)))
    has_null = np.any(null_cmp)

    if has_null:
        null_index = np.nonzero(null_cmp)[0][0]
        enc.classes_.append(enc.classes_.pop(null_index))

    return enc



class fOneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, sep='_', dummy_na=False, columns=None):

        self.sep = sep
        self.dummy_na = dummy_na
        self.columns = columns


    def fit(self, X, y=None):

        # encoding columns as integers {0, 1, ..., n_classes}
        self.enc_ = fOrdinalEncoder(nan_handle = 'hard', columns=self.columns, copy=True).fit(X)

        # propagating nan values to the end of the classes_ list in fLabelEncoder's
        for i, enc in enumerate(self.enc_.encoders_):
            self.enc_.encoders_[i] = _nan_to_end(enc)

        return self

    def transform(self, X):

        X_ord = self.enc_.transform(X)

        names = []
        arrays = []

        for i, column in enumerate(self.enc_.columns_):

            classes = self.enc_.encoders_[i].classes_

            if self.dummy_na:
                rows = np.eye(len(classes))
            else:
                rows = np.vstack([np.eye(len(classes)-1), np.zeros(len(classes)-1)[np.newaxis, :]])

            array = rows.take(X_ord[column].values, axis=0)

            name_list = [str(column) + self.sep + str(cls) for cls in classes]
            if not self.dummy_na:
                name_list = name_list[:-1]

            arrays.append(array)
            names.extend(name_list)

        arrays = np.hstack(arrays)
        dummies = pd.DataFrame(arrays,index=X.index, columns=names)

        return pd.concat([X.drop(self.enc_.columns_, axis=1) , dummies], axis=1)






class fOrdinalEncoder(BaseEstimator, TransformerMixin):
    '''
    param: copy - weather to return a copy or change data inplace.
    param:columns - columns to be encoded.

    param: dtype - dtype of resulted encoded Series, defaults to uint8.

    param: nan_handle -

    nan_handle is one of ['soft', 'hard', 'ignore']

    soft - nans will be encoded in transform only if nans are present during fit.
    hard - nans are assigned a label in transform even if not present during fit.
    ignore - ignores nan's all-together.
    '''
    def __init__(self, dtype=np.uint8, nan_handle='soft', columns=None, copy=True):

        self.dtype = dtype
        self.nan_handle = nan_handle
        self.columns = columns
        self.copy = copy

    def fit(self, X, y=None):

        # ensuring X is a Series object
        check_type(X, types=[pd.DataFrame])

        # choosing columns to encode
        if self.columns is None:
            self.columns_ = X.columns.tolist()
        else:
            check_has_columns(X, self.columns)
            self.columns_ = list(self.columns)

        # initiallizing one encoder for each column
        self.encoders_ = [fLabelEncoder(dtype=self.dtype, nan_handle=self.nan_handle) for _ in range(len(self.columns_))]

        # fitting each encoder with his corresponding column
        for i, column in enumerate(self.columns_):
            self.encoders_[i].fit(X[column])

        return self

    def transform(self, X):

        # checking data contains encoded columns
        check_has_columns(X, self.columns_)

        # return dataframe built from all Label Encoded column
        if self.copy:
            result = pd.concat([X[[column for column in X.columns.tolist() if column not in self.columns_]]]+\
                               [self.encoders_[i].transform(X[column]) for i, column in enumerate(self.columns_)], axis=1)
        else:
            result = X
            for i, column in enumerate(self.columns_):
                X[column] = self.encoders_[i].transform(X[column])

        return result

    def inverse_transform(self, X):

        # checking data contains encoded columns
        check_has_columns(X, self.columns_)

        # return dataframe after replacing labels with their corresponding unique classes
        if self.copy:
            result = pd.concat([X[[column for column in X.columns.tolist() if column not in self.columns_]]]+\
                               [self.encoders_[i].inverse_transform(X[column]) for i, column in enumerate(self.columns_)], axis=1)
        else:
            result = X
            for i, column in enumerate(self.columns_):
                X[column] = self.encoders_[i].inverse_transform(X[column])

        return result

class fLabelEncoder(BaseEstimator, TransformerMixin):
    """
    fLabelEncoder class with same functionality as sklearn's LabelEncoder
    only that fLabelEncoder accepts only Series type objects.

    adds some extra functionality

    param: dtype - dtype of resulted encoded Series, defaults to uint8

    param: nan_handle -
    nan_handle is one of ['soft', 'hard', 'ignore']

    soft - nans will be encoded in transform only if nans are present during fit
    hard - nans are assigned a label in transform even if not present during fit
    ignore - ignores nan's all-together

    """
    def __init__(self, dtype=np.uint8, nan_handle='soft'):
        self.dtype = dtype
        self.nan_handle = nan_handle

    def fit(self, X, y=None):

        # type checking input
        check_type(X, types=[pd.Series])

        # extracting unique labels from series
        self.classes_ = pd.unique(X).tolist()

        # boolean saying if null present in data
        null_cmp = np.asarray(pd.isnull(self.classes_)) |\
                   np.asarray(list(map(lambda label : str(label).lower() == 'nan', self.classes_)))
        has_null = np.any(null_cmp)

        # if nan_handle is hard ensure nan is present in classes_
        if self.nan_handle == 'hard' and not has_null:
            self.classes_.append(np.nan)

        if self.nan_handle == 'ignore' and has_null:
            null_index = np.nonzero(null_cmp)[0][0]
            del self.classes_[null_index]

        return self

    def transform(self, X):

        # type checking input
        check_type(X, types=[pd.Series])

        if self.nan_handle == 'ignore':
            dtype = np.float32
        else :
            dtype = self.dtype

        # replacing unique classes with corresponding labels
        result = X.replace(to_replace=self.classes_, value=range(len(self.classes_))).astype(dtype)

        return result

    def inverse_transform(self, X):

        # type checking input
        check_type(X, types=[pd.Series])

        # replacing labels with corresponding unique classes
        result = X.replace(to_replace=range(len(self.classes_)), value=self.classes_)

        return result

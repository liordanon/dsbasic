import numpy as np
import pandas as pd

def check_has_columns(X, columns):
    """
    :param X: pandas Dataframe (#Rows, #Features)
    :param columns: list of strings specifying columns/features to check for in X
    raises ValueError if one of the columns or more are missing from X.
    """
    data_columns = X.columns.tolist()
    missing_columns = []

    for column in columns:
        if column not in data_columns:
            missing_columns.append(column)

    if len(missing_columns) > 0:
        raise ValueError('columns ' + str(missing_columns) + ' are missing from data.')

def check_type( X , types = [pd.DataFrame]):
    """
    :param X: object
    :param types: iterable of types/classes
    raises TypeError X is not of one of the types in types.
    """
    if not any(isinstance(X, Type) for Type in types):
        raise TypeError('Data of type ' + str(type(X)) + ' should be one of ' + str(types) + '.\n')

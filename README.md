# dsbasic
 
 Open source library for basic data science tasks.
 
implements in part:

1.  sklearn API compatible transformers that act on pandas DataFrames.
2.  a Visualizer object with some nice visualization functions and api.
3.  helpers in writing clean and understandable data pipelines.

## Imputation

the dsbasic.frame.preprocessing.impute module implements the fImputer transformer
used to impute dataframe at selected columns.

fImputer(strategy='mean', copy=True, na_sentinel=-1, columns=None)

* columns - list of columns names to impute
* strategy - a string from  'mean', 'median', 'most_frequent', 'na_sentinel' where each one specifies which method of imputation is to be used.
* copy - whether the returned frame should be a copy or not.
* na_sentinel - if strategy = 'na_sentinel' fills all columns Na's with the na_sentinel variable.

fImputer accepts both pandas.DataFrame and pandas.Series objects.

Example 

```
from dsbasic.frame.preprocessing.impute import fImputer
from sklearn.pipeline import make_pipeline

numeric = ['n1', 'n2', 'n3']
categorical = ['c1', 'c2', 'c3']

imputer = make_pipeline(
	fImputer(strategy='median', columns = numeric, copy=True),
	fImputer(strategy='most_frequent', columns = categorical, copy=True)
)

X = pandas.read_csv(...)
Y = pandas.read_csv(...)
X_imputed = imputer.fit_transform(X)
Y_imputed = imputer.transform(Y)
```
	


## Categorical Variable Encoding

the dsbasic.frame.preprocessing.categorical module implements useful transformers to 
deal with categorical features.
specifically the fOrdinalEncoder, fOneHotEncoder, fLabelEncoder

**fLabelEncoder(dtype=np.uint8, nan_handle='soft' )**

assigns a natural number to each unique label of the pandas series.
* dtype - dtype of ordinal oncoded columns.
* nan_handle - nan_handle is one of ['soft', 'hard', 'ignore']

soft - nans will be encoded in transform only if nans are present during fit.
hard - nans are assigned a label in transform even if not present during fit.
ignore - ignores nan's all-together.

note : if nan_handle is set to 'ignore' dtype argument is ignored and is set to float32

**fLabelEncoder accepts only a pandas.Series object.
to encode several columns see fOrdinalEncoder**

Example :
```
from dsbasic.frame.preprocessing.categorical import fLabelEncoder 
from sklearn.pipeline import make_pipeline

labels = pandas.Series(['a', 'b', 'a', 'c', numpy.nan, 'a'])

y1 = fLabelEncoder(nan_handle='ignore').fit_transform(labels)
y2 = fLabelEncoder(nan_handle='soft').fit_transform(labels)
y3 = fLabelEncoder(nan_handle='hard').fit_transform(labels)

print('y1\n{}\n\n{}\n\n{}'.format(y1, y2, y3))
```

output :
```
0 0.0 
1 1.0 
2 0.0 
3 2.0 
4 NaN 
5 0.0 
dtype: float32 

0 0 
1 1 
2 0 
3 2 
4 3 
5 0 
dtype: uint8 

0 0 
1 1 
2 0 
3 2 
4 3 
5 0 
dtype: uint8
```
**fOrdinalEncoder(dtype=np.uint8, nan_handle='soft', columns=None, copy=True)**

Label encodes each column in "columns" using fLabelEncoder

* dtype - dtype of ordinal oncoded columns.
* nan_handle - nan_handle is one of ['soft', 'hard', 'ignore']

soft - nans will be encoded in transform only if nans are present during fit.
hard - nans are assigned a label in transform even if not present during fit.
ignore - ignores nan's all-together.
* columns - list of strings describing the columns to be encoded.
* copy - whether the returned frame should be a copy or not.

**fOneHotEncoder(sep='_', dummy_na=False, columns=None)**

One hot encodes selected columns of a dataframe and discards the 
original columns (pandas get_dummies style). 

* sep - new one hot encoded column names are set to be column_name + sep + label_name
* dummy_na - whether to one hot encode Na's.
* columns -  list of strings describing the columns to be encoded.

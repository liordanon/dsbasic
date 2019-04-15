import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from .preprocessing.categorical import fLabelEncoder


class col(object):
    def __init__(self, column):
        self.col = column

    def get_array(self, frame):
        return frame[self.col].values



class Visualizer(object):

    def __init__(self, colors = None):

        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'] if colors is None else colors
        self.fig = None
        self.ax = None
        self.i = 0

    def fit(self, X, y=None):
        self.X_ = X

        return self

    def subplots(self, dims=(1, 1)):
        self.fig, self.ax = plt.subplots(nrows = dims[0], ncols=dims[1])
        if tuple(dims) == (1, 1):
            self.ax = [self.ax]
        self.i = 0

    def PolyVis(self, x=None, y=None, deg=1):

        points = self.X_[[x, y]].dropna()
        X, yT = points[x].values.astype(np.float32), points[y].values.astype(np.float32)

        polynomial = np.polyfit(X, yT, deg=deg)
        y_ = np.polyval(polynomial, X)


        self.ax[self.i].set_ylabel(y)
        self.ax[self.i].set_xlabel(x)
        self.ax[self.i].scatter(X, yT, color='r')
        self.ax[self.i].plot(X, y_, color='b')
        self.i += 1

    def PolyResHist(self, x=None, y=None, deg=1, bins=10):

        points = self.X_[[x, y]].dropna()
        X, yT = points[x].values.astype(np.float32), points[y].values.astype(np.float32)

        polynomial = np.polyfit(X, yT, deg=deg)
        y_ = np.polyval(polynomial, X)

        res = yT - y_

        self.ax[self.i].set_xlabel(x)
        self.ax[self.i].set_ylabel(y + '[RES]')
        self.ax[self.i].hist(res, bins=bins)
        self.i += 1


    def PlanarVis(self, features = None, label=None, scaler=None):

        if features is None:
            features = self.X_.columns.tolist()
        else:
            features = list(features)

        if scaler is None:
            data = self.X_[features ].dropna().values
        else:
            data = scaler.fit_transform(self.X_[features].dropna().values)

        if isinstance(label, str):
            label = self.X_[label]

        data_2d = PCA(n_components=2).fit_transform(data)

        enc = fLabelEncoder()
        indices = enc.fit_transform(label)

        self.ax[self.i].set_ylabel('Component 1')
        self.ax[self.i].set_xlabel('Component 2')
        self.ax[self.i].scatter(data_2d[:, 0], data_2d[:, 1], c= np.array(self.colors).take(indices))
        self.i += 1

    def fromFunnction(self, func, *args, **kwargs):

        Ax_ = self.ax[self.i]
        func_axis = eval('Ax_.{}'.format(func.__name__))
        args = tuple([item  if not isinstance(item, col) else item.get_array(self.X_) for item in args])
        kwargs = {key : value  if not isinstance(value, col) else value.get_array(self.X_) for key, value in kwargs}
        func_axis(*args, **kwargs)
        self.i += 1

    def show(self):
        plt.show()














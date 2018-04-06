from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.validation import check_is_fitted
import numpy as np
import pandas as pd

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, labels=None):
        if isinstance(labels, str):
            self.labels = [labels]
        elif isinstance(labels, list):
            self.labels = labels
        else:
            self.labels = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.labels].values


class CategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='most_frequent'):
        self.strategy = strategy

    def fit(self, X, y=None):
        if self.strategy is 'most_frequent':
            self.statistics_ = pd.DataFrame(X).mode().iloc[0].values
        elif self.strategy is 'empty_string':
            self.statistics_ = np.empty((1, X.shape[1]), str)
        else:
            allowed_strategies = ['most_frequent', 'empty_string']
            raise ValueError("Can only use these strategies: {0} got strategy={1}".format(allowed_strategies, self.strategy))
        return self

    def transform(self, X):
        check_is_fitted(self, 'statistics_')
        if X.shape[1] != self.statistics_.shape[0]:
            raise ValueError("X has {0} features per sample, expected {0}".format(X.shape[1], self.statistics_.shape[0]))
        return pd.DataFrame(X).fillna(pd.Series(self.statistics_)).values


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.label_encoders_ = [LabelEncoder().fit(column) for column in X.T]
        self.one_hot_encoder_ = OneHotEncoder(handle_unknown='ignore').fit(self.label(X))
        return self

    def label(self, X):
        check_is_fitted(self, 'label_encoders_')
        return np.hstack([self.label_encoders_[i].transform(column).reshape(-1, 1) for i, column in enumerate(X.T)])

    def transform(self, X):
        check_is_fitted(self, 'one_hot_encoder_')
        return self.one_hot_encoder_.transform(self.label(X))


class ConstantScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scale=1.0):
        self.scale = scale

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * self.scale

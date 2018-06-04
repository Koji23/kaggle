from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, labels=None):
        self.labels = labels

    def fit(self, X, y=None):
        if isinstance(self.labels, str):
            self.labels_ = [self.labels]
        elif isinstance(self.labels, list):
            self.labels_ = self.labels
        else:
            self.labels_ = []
        return self

    def transform(self, X):
        check_is_fitted(self, 'labels_')
        return X[self.labels_].values


class ConstantScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scale=1.0):
        self.scale = scale

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * self.scale

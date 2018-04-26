from sklearn.base import BaseEstimator, TransformerMixin


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


class ConstantScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scale=1.0):
        self.scale = scale

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * self.scale

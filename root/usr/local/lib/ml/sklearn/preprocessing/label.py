from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.validation import check_is_fitted
import numpy as np


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

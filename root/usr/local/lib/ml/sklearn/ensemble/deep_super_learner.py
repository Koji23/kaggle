from ml.utils import tensor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state, resample, shuffle
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import matplotlib.pyplot as plt
import numpy as np


class DeepSuperClassifier(BaseEstimator, ClassifierMixin):
    """ An example classifier which implements a 1-NN algorithm.
    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store parameters.
    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        The input passed during :meth:`fit`
    y_ : array, shape = [n_samples]
        The labels passed during :meth:`fit`
    """
    def __init__(self, classifiers=[], random_state=None):
        self.classifiers = classifiers
        self.random_state = random_state

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        self.classes_, y = np.unique(y, return_inverse=True)
        random_state = check_random_state(self.random_state)

        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of int of shape = [n_samples]
            The label for each sample is the label of the class with highest predicted probability.
        """
        predictions = self.predict_proba(X)
        return self.classes_[np.argmax(predictions, axis=1)]

    def predict_proba(self, X):
        """Return probability estimates for the test data X.
        Parameters
        ----------
        X : array-like, shape (n_query, n_features)
            Test samples.
        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered by lexicographic order.
        """
        # input validation
        check_is_fitted(self, ['classes_'])
        X = check_array(X)
        return None

    def plot(self):
        # TODO
        pass

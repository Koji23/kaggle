from math import sqrt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_consistent_length, check_random_state
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook as tqdm

# estimators
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

default_estimators = [
    ExtraTreesClassifier(),
    LogisticRegression(),
    KNeighborsClassifier(),
    RandomForestClassifier(),
    XGBClassifier()
]


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
    def __init__(self, estimators=default_estimators, n_iter=10, n_jobs=-1, n_splits=3, random_state=None):
        self.estimators = estimators
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.n_splits = n_splits
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
        # input validation
        X, y = check_X_y(X, y)
        random_state = check_random_state(self.random_state)

        # convert class names to integer values
        self.label_encoder_ = LabelEncoder().fit(y)
        y = self.label_encoder_.transform(y)

        # helper variables for array sizes
        n_estimators = len(self.estimators)
        n_instances = X.shape[0]
        n_classes = self.label_encoder_.classes_.shape[0]
        X = np.hstack((np.zeros(n_instances, n_classes), X))

        # define super learner
        self.losses_ = []
        self.super_learner_ = MLPClassifier((n_classes, ))

        # fit estimators
        # TODO: maybe use sklearn.model_selection.RepeatedStratifiedKFold?
        loss = 0
        for epoch in tqdm(range(self.n_iter), desc='Epoch'):
            folds = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=random_state)
            predictions = np.zeros((n_instances, n_estimators, n_classes))

            # TODO: use parallel jobs or maybe VotingClassifier, which does all this for me
            for train_index, test_index in folds.split(X, y):
                X_train, y_train = X[train_index], y[train_index]
                X_test, y_test = X[test_index], y[test_index]
                for i, estimator in enumerate(self.estimators):
                    predictions[test_index,i] = estimator.fit(X_train, y_train).predict_proba(X_test)

            final_predictions = predictions.reshape(n_instances, -1)
            X[:,:n_classes] = self.super_learner_.fit(final_predictions, y).predict_proba(final_predictions)
            self.losses_.append(self.super_learner_.loss_)
            # TODO: test and store accuracy of all submodels as well as ensemble at this epoch for plotting

            # stop training if we are no long improving
            # TODO: instead of loss, should we just stop when the predictions haven't changed?
            if self._converged(loss, self.super_learner_.loss_):
                break
            else:
                loss = self.super_learner_.loss_
        self.epochs_ = epoch

        return self

    def _converged(self, curr_loss, prev_loss, tol=1e-3):
        return prev_loss - curr_loss < tol

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
        return self.label_encoder_.inverse_transform(predictions)

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
        check_is_fitted(self, ['epochs_', 'label_encoder_', 'losses_', 'super_learner_'])
        X = check_array(X)

        # helper variables for array sizes
        n_estimators = len(self.estimators)
        n_instances = X.shape[0]
        n_classes = self.label_encoder_.classes_.shape[0]
        predictions = np.zeros((n_instances, n_estimators, n_classes))
        X = np.hstack((np.zeros(n_instances, n_classes), X))

        # cycle through estimators until convergence
        # TODO: convergence here means that the predictions haven't changed
        for epoch in tqdm(range(self.epochs_), desc='Epoch'):
            for i, estimator in enumerate(self.estimators):
                predictions[:,i] = estimator.predict_proba(X)
            final_predictions = predictions.reshape(n_instances, -1)
            X[:,:n_classes] = self.super_learner_.predict_proba(final_predictions)

        # return predictions
        return X[:,:n_classes]

    def plot(self):
        # TODO
        pass

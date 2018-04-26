from ...torch.optim.bs_scheduler import ExponentialBS
from ...utils import one_hot_encode, tensor
from ...torch.optim import LearningRateFinder
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import check_random_state, resample, shuffle
from tqdm import tqdm_notebook as tqdm


class TorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, network, batch_size=32, max_epochs=250, n_early_stoppage=5):
        self.network = network
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.n_early_stoppage = n_early_stoppage

    def fit(self, X, y, X_validation=None, y_validation=None, random_state=None):
        """
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
        # check that X and y have correct shape
        X, y = check_X_y(X, y)
        random_state = check_random_state(random_state)
        self.X_ = X
        self.y_ = y
        self.classes_ = unique_labels(y)
        n_classes = len(self.classes_)


        # check validation shape
        if X_validation is None or y_validation is None:
            X_validation, y_validation = resample(X, y, n_samples=1000, random_state=random_state)
        X_validation, y_validation = check_X_y(X_validation, y_validation)
        X_validation, y_validation = tensor(X_validation), tensor(one_hot_encode(y_validation, n_classes))

        # store training and validation losses at every epoch
        self.training_loss_ = []
        self.validation_loss_ = []
        self.index_min_ = None
        self.state_min_ = None

        # set model hyperparameters
        self.network.train()
        self.learning_rate = LearningRateFinder(self.network).fit(X, y, batch_size=self.batch_size, random_state=random_state)
        loss_fn = nn.MSELoss(size_average=False)
        bs_scheduler = ExponentialBS(self.batch_size, max_bs=1024, gamma=2, T=4)
        lr_scheduler = self.learning_rate.scheduler(gamma=0.90)
        optimizer = lr_scheduler.optimizer

        for epoch in tqdm(range(self.max_epochs), desc='Epoch'):
            X, y = shuffle(X, y, random_state=random_state)
            epoch_training_loss = 0.0
            batch_size = bs_scheduler.get_bs()

            for i in range(0, X.shape[0], batch_size):
                batch_x = tensor(X[i:i+self.batch_size], True)
                batch_y = tensor(one_hot_encode(y[i:i+self.batch_size], n_classes))
                loss = loss_fn(self.network(batch_x), batch_y)
                epoch_training_loss += loss.data[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # score model against the training set
            self.training_loss_.append(epoch_training_loss / X.shape[0])

            # score model against a validation set
            # (a decrease in validation accuracy implies over-fitting)
            self.network.eval()
            epoch_validation_loss = loss_fn(self.network(X_validation), y_validation).data[0]
            self.validation_loss_.append(epoch_validation_loss / X_validation.shape[0])
            self.index_min_ = np.argmin(self.validation_loss_)
            self.network.train()

            # save the best model seen
            if self.index_min_ == len(self.validation_loss_) - 1:
                self.state_min_ = self.network.state_dict()

            # stop early if validation loss is higher than minimum loss for n_early_stoppage epochs in a row
            if self.index_min_ < len(self.validation_loss_) - self.n_early_stoppage:
                break

            bs_scheduler.step()
            lr_scheduler.step()


        # return the best model seen
        self.network.load_state_dict(self.state_min_)
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
        # input validation
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)

        # return labels of predicted classes
        self.network.eval()
        values, predictions = self.network(tensor(X)).max(1)
        return predictions.data.numpy()

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
        check_is_fitted(self, ['X_', 'y_'])
        X = tensor(check_array(X))

        # are predictions already probabilities?
        modules = list(self.network.modules())
        has_softmax = isinstance(modules[-1], nn.Softmax)

        # return predicted probabilities of classes
        if has_softmax:
            predictions = self.network(X)
        else:
            predictions = nn.functional.softmax(self.network(X), 1)
        return predictions.data.numpy()

    def plot(self):
        check_is_fitted(self, ['training_loss_', 'validation_loss_'])
        plt.figure(figsize=(12,8))
        plt.title('Training and Test Loss by Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Log Loss')
        plt.plot(np.log(self.training_loss_), label='Training Loss')
        plt.plot(np.log(self.validation_loss_), label='Validation Loss')
        plt.axvline(x=self.index_min_, color='k', linestyle='--')
        plt.legend(loc='upper right')
        plt.show()

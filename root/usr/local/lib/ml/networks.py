import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import resample, shuffle
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm_notebook as tqdm


def variable(X, requires_grad=False):
    return Variable(torch.from_numpy(X).float(), requires_grad=requires_grad)


def one_hot_encode(labels, n_labels):
    one_hot = np.zeros((labels.shape[0], n_labels))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot


class SGDR(_LRScheduler):
    """Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max=2, T_mult=2, eta_min=0, gamma=1.0, last_epoch=-1):
        self.T_max = T_max
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.gamma = gamma
        self.restart_epoch = 0
        self.restart_gamma = 1.0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.restart_gamma * (self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.restart_epoch / self.T_max)) / 2)
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        super().step(epoch)
        if self.restart_epoch == self.T_max:
            self.restart_epoch = 0
            self.restart_gamma *= self.gamma
            self.T_max *= self.T_mult
        else:
            self.restart_epoch += 1


class LearningRateFinder(object):
    """Find the optimal minimum and maximum learning rates for a model using a specific optimizer.

    It has been proposed in `Cyclical Learning Rates for Training Neural Networks`_.

    Args:
        model (Module): torch.nn module
        loss_fn: torch.nn loss function instance
        optimizer (Optimizer): optimizer class

    .. _Cyclical Learning Rates for Training Neural Networks:
        http://arxiv.org/abs/1506.01186
    """
    def __init__(self, model, loss_fn=nn.MSELoss(), optimizer_class=torch.optim.Adam, optimizer_kwargs={}):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

    def fit(self, X, y, batch_size=64, min_lr=-5, max_lr=0, n_lr=301, smoothing=5, random_state=None):
        X, y = check_X_y(X, y)
        n_classes = len(unique_labels(y))
        model = copy.deepcopy(self.model)
        optimizer = self.optimizer_class(model.parameters(), lr=0, **self.optimizer_kwargs)
        self.learning_rates_ = np.logspace(min_lr, max_lr, num=n_lr)
        self.losses_ = []

        for learning_rate in self.learning_rates_:
            # update learning rates
            for group in optimizer.param_groups:
                group['lr'] = learning_rate

            # random training batches
            batch_x, batch_y = resample(X, y, random_state=random_state)
            batch_x, batch_y = variable(batch_x, True), variable(one_hot_encode(batch_y, n_classes))

            # calculate training loss at the current learning rate
            loss = self.loss_fn(model(batch_x), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save loss
            self.losses_.append(loss.data[0])

        # smooth out random variations
        len_losses = len(self.losses_)
        self.losses_ = [np.median(self.losses_[max(0,i-smoothing):min(len_losses-1,i+smoothing)]) for i in range(len_losses)]
        self.d_losses_ = [self.losses_[min(len_losses-1,i+smoothing)] - self.losses_[max(0,i-smoothing)] for i in range(len_losses)]
        return self

    @property
    def max(self):
        """The maximum learning rate is the point where accuracy no longer increases."""
        check_is_fitted(self, ['d_losses_', 'learning_rates_'])
        mindex = np.argmin(self.d_losses_)
        minslope = self.d_losses_[mindex] / 8
        maxdex = mindex + np.where(np.array(self.d_losses_[mindex:]) > minslope)[0][0]
        return self.learning_rates_[maxdex]

    @property
    def min(self):
        """The minimum learning rate is the point where accuracy begins to increase quickly."""
        check_is_fitted(self, ['d_losses_', 'learning_rates_'])
        return self.learning_rates_[np.argmin(self.d_losses_)]

    def scheduler(self, **kwargs):
        check_is_fitted(self, ['d_losses_', 'learning_rates_'])
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.max, **self.optimizer_kwargs)
        return SGDR(optimizer, eta_min=self.min, **kwargs)

    def plot(self, type='losses'):
        check_is_fitted(self, ['d_losses_', 'learning_rates_'])
        plt.figure(figsize=(12,8))
        if type is 'losses':
            plt.title('Training Loss by Learning Rate')
            plt.xlabel('Log Learning Rate')
            plt.ylabel('Training Loss')
            plt.plot(np.log10(self.learning_rates_), self.losses_)
            plt.axvline(x=np.log10(self.min), color='k', linestyle='--', label='Minimum LR')
            plt.axvline(x=np.log10(self.max), color='c', linestyle='--', label='Maximum LR')
            plt.legend(loc='upper right')
        elif type is 'd_losses':
            plt.title('d(Training Loss) by Learning Rate')
            plt.xlabel('Log Learning Rate')
            plt.ylabel('d(Training Loss)')
            plt.plot(np.log10(self.learning_rates_), self.d_losses_)
            plt.axvline(x=np.log10(self.min), color='k', linestyle='--', label='Minimum LR')
            plt.axvline(x=np.log10(self.max), color='c', linestyle='--', label='Maximum LR')
            plt.legend(loc='upper right')
        plt.show()


class NetworkClassifier(BaseEstimator, ClassifierMixin):
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
        self.X_ = X
        self.y_ = y
        self.classes_ = unique_labels(y)
        n_classes = len(self.classes_)


        # check validation shape
        if X_validation is None or y_validation is None:
            X_validation, y_validation = resample(X, y, n_samples=1000, random_state=random_state)
        X_validation, y_validation = check_X_y(X_validation, y_validation)
        X_validation, y_validation = variable(X_validation), variable(one_hot_encode(y_validation, n_classes))

        # store training and validation losses at every epoch
        self.training_loss_ = []
        self.validation_loss_ = []
        self.index_min_ = None
        self.state_min_ = None

        # set model hyperparameters
        self.network.train()
        self.learning_rate = LearningRateFinder(self.network).fit(X, y, batch_size=self.batch_size, random_state=random_state)
        loss_fn = nn.MSELoss(size_average=False)
        lr_scheduler = self.learning_rate.scheduler(gamma=0.90)
        optimizer = lr_scheduler.optimizer

        for epoch in tqdm(range(self.max_epochs), desc='Epoch'):
            X, y = shuffle(X, y, random_state=random_state)
            epoch_training_loss = 0.0

            # TODO: formalize this idea with a class (subclass of _BSScheduler, after _LRScheduler)
            # TODO: needs to be concerned with OOM errors?
            batch_size = min(1024, self.batch_size * 2**int(epoch / 4))

            for i in range(0, X.shape[0], batch_size):
                batch_x = variable(X[i:i+self.batch_size], True)
                batch_y = variable(one_hot_encode(y[i:i+self.batch_size], n_classes))
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
        values, predictions = self.network(variable(X)).max(1)
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
        X = check_array(X)

        # are predictions already probabilities?
        modules = list(self.network.modules())
        has_softmax = isinstance(modules[-1], nn.Softmax)

        # return predicted probabilities of classes
        if has_softmax:
            predictions = self.network(variable(X))
        else:
            predictions = nn.functional.softmax(self.network(variable(X)), 1)
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


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.shape)
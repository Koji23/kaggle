from .lr_scheduler import SGDR
from ml.utils import tensor
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils import check_random_state, resample
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


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
    def __init__(self, model, n_iter=301, loss_fn=nn.MSELoss(), optimizer_class=torch.optim.Adam, optimizer_kwargs={}, random_state=None):
        self.model = model
        self.n_iter = n_iter
        self.loss_fn = loss_fn
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.random_state = random_state

    def fit(self, X, y, batch_size=64, min_lr=-5, max_lr=0, smoothing=5):
        # input validation
        X, y = check_X_y(X, y)
        random_state = check_random_state(self.random_state)

        # set model hyperparameters
        self.learning_rates_ = np.logspace(min_lr, max_lr, num=self.n_iter)
        self.losses_ = []
        model = copy.deepcopy(self.model)
        optimizer = self.optimizer_class(model.parameters(), lr=0, **self.optimizer_kwargs)

        # convert class names to one-hot encoded integer values
        y = LabelBinarizer().fit_transform(y)

        for learning_rate in self.learning_rates_:
            # update learning rates
            for group in optimizer.param_groups:
                group['lr'] = learning_rate

            # random training batches
            batch_x, batch_y = resample(X, y, n_samples=batch_size, random_state=random_state)
            batch_x, batch_y = tensor(batch_x, True), tensor(batch_y)

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

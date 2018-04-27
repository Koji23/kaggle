from torch.optim.lr_scheduler import _LRScheduler
import math


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

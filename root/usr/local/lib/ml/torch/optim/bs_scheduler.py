class _BSScheduler(object):
    def __init__(self, base_bs=32, last_epoch=0):
        self.base_bs = base_bs
        self.last_epoch = last_epoch

    def get_bs(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch


class ExponentialBS(_BSScheduler):
    """Set the batch size to the initial bs increased by gamma every T epochs.

    Args:
        gamma (float): Multiplicative factor of batch size increase. Default: 2
        T (int): Number of epochs to wait before increasing batch size. Default: 4.
        last_epoch (int): The index of last epoch. Default: 0.
    """

    def __init__(self, base_bs=32, max_bs=1024, gamma=2, T=4, last_epoch=0):
        self.max_bs = max_bs
        self.gamma = gamma
        self.T = T
        super().__init__(base_bs, last_epoch)

    def get_bs(self):
        return int(min(self.max_bs, self.base_bs * self.gamma ** int(self.last_epoch / self.T)))

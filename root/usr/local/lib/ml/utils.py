import numpy as np
import torch


def tensor(X, requires_grad=False):
    return torch.from_numpy(X).float().requires_grad_(requires_grad)


def one_hot_encode(labels, n_labels):
    one_hot = np.zeros((labels.shape[0], n_labels))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot

import torch


def tensor(X, requires_grad=False):
    return torch.from_numpy(X).float().requires_grad_(requires_grad)

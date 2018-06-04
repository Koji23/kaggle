import torch


def tensor(X, requires_grad=False, dtype=torch.float32):
    return torch.from_numpy(X).to(dtype).requires_grad_(requires_grad)

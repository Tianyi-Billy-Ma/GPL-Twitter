import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def normalize(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    rowsum = x.sum(1)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    x = r_mat_inv.dot(x)
    if isinstance(x, np.ndarray):
        return x
    else:
        return x.todense()


def sim(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class BGRL_EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = (
            1
            - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        )
        self.step += 1
        return old * beta + (1 - beta) * new

import torch
import torch.nn as nn


class OverSampling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, y):
        occ = torch.eye(int(y.max() + 1), int(y.max() + 1)).to(y.device)[y].sum(axis=0)
        dominant_class = torch.argmax(occ)
        n_occ = int(occ[dominant_class].item())
        xs, ys = [], []
        for i in range(len(occ)):
            if i != dominant_class:
                # calculate the amount of synthetic data to generate
                N = (n_occ - occ[i]) * 100 / occ[i]
                candidates = X[y == i]
                selection = torch.randint(
                    0, candidates.shape[0], (int(N),), device=X.device
                )
                xs.append(candidates[selection])
                ys.append(torch.ones(int(N)) * i)
        xs = torch.cat(xs).to(X.device)
        ys = torch.cat(ys).to(y.device)
        X = torch.cat((X, xs))
        y = torch.cat((y, ys))
        return X, y

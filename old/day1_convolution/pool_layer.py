import torch
from torch import nn
from d2l import torch as d2l

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'mean':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = torch.arange(16.0).reshape(4, 4)
print(pool2d(X, (2, 2)))
print(pool2d(X, (2, 2), 'mean'))
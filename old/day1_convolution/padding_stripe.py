import torch
from torch import nn

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Exclude the first two dimensions that do not interest us: examples and channels
    return Y.reshape(Y.shape[2:])

X = torch.rand(size=(8, 8))
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1), bias=False)
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2, bias=False)
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4), bias=False)
print(comp_conv2d(conv2d, X).shape)
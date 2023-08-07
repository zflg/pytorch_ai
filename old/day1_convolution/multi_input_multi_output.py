import torch
from d2l import torch as d2l

# 在通道0处合并
def conv2d_multi_in(X, K):
    # First, iterate through the 0th dimension (channel dimension) of `X` and
    # `K`. Then, add them together
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

X = torch.stack((torch.arange(9.0).reshape(3, 3), torch.arange(1.0, 10.0).reshape(3, 3)), dim=0);
K = torch.stack((torch.arange(4.0).reshape(2, 2), torch.arange(1.0, 5.0).reshape(2, 2)), dim=0);

print(conv2d_multi_in(X, K))

# 在通道维度上分别计算再合并
def conv2d_multi_in_out(X, K):
    # Iterate through the 0th dimension of `K`, and each time, perform
    # cross-correlation operations with input `X`. All of the results are
    # stacked together
    return torch.stack([conv2d_multi_in(X, k) for k in K], dim=0)

multi_K = torch.stack((K, K + 1, K + 2), dim=0)

print(conv2d_multi_in_out(X, multi_K))

K = torch.randn(2, 3, 1, 1)
X = torch.randn(3, 3, 3)
def conv2d_multi_in_out_1x1(X, K):
    out_dim, in_dim = K.shape[0], K.shape[1]
    h, w = X.shape[1:]
    X = X.reshape(in_dim, h * w)
    K = K.reshape(out_dim, in_dim)
    # Matrix multiplication in the fully-connected layer
    return torch.matmul(K, X).reshape(out_dim, h, w)

Y1 = conv2d_multi_in_out_1x1(X, K)
Y2 = conv2d_multi_in_out(X, K)
print(Y1)
print(Y2)
print(float(torch.abs(Y1 - Y2).sum()))

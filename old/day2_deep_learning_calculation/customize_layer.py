import torch
import torch.nn.functional as F
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


layer = CenteredLayer()
print(layer(torch.arange(6.0)))


class MyLinear(nn.Module):
    def __init__(self, in_num, out_num):
        super().__init__()
        self.weight = nn.Parameter(torch.normal(0, 1, size=(in_num, out_num)))
        self.bias = nn.Parameter(torch.zeros(out_num))
    def forward(self, X):
        return F.relu(torch.matmul(X, self.weight.data) + self.bias.data)

linear = MyLinear(8, 4)
print(linear.weight)
print(linear(torch.rand(2, 8)))

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = MyLinear(20, 64)
        self.output = MyLinear(64, 10)
    def forward(self, X):
        return self.output(F.relu(self.hidden(X)))
mlp = MLP()
X = torch.randn(2, 20)
Y = mlp(X)
print(Y)
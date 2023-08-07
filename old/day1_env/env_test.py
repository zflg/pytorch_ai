import torch
from torch import nn
from d2l import torch as d2l

x = torch.tensor([2, 3])
y = torch.arange(12).reshape(3, 4)
z = torch.zeros(size=(3, 3))

print(x, y, z)
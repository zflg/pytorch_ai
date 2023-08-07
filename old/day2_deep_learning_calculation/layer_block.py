import torch
from torch import nn

# net其实是多个层的组合
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.arange(8.0).reshape(2, 4)
Y = net(X)
print(Y)

# 在每一层获取所有参数
# for i in range(len(net)):
#     print(net[i].state_dict())


def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # Nested here
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
print(Y)



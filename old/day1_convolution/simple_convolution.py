import torch
from torch import nn

def conv2d(X, K):
    h, w = K.shape
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

X = torch.arange(9.0).reshape(3, 3)
K = torch.arange(4.0).reshape(2, 2)
print(X, K, conv2d(X, K))

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return conv2d(x, self.weight) + self.bias

a1 = torch.ones(6, 8)
a1[:, 2:6] = 0
a2 = torch.tensor([[1.0, -1.0]])
b1 = conv2d(a1, a2)
b2 = conv2d(a1.t(), a2)

X = a1.reshape(1, 1, 6, 8)
Y = b1.reshape(1, 1, 6, 7)
lr = 0.3


net = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
trainer = torch.optim.SGD(net.parameters(), lr)
# loss = nn.MSELoss()
loss = nn.HuberLoss()
for i in range(100):
    # l = (net(X) - Y) ** 2
    l = loss(net(X), Y) 
    net.zero_grad()
    # l.mean().backward()
    l.backward()
    trainer.step()
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.10f}, weight {net.weight.data}, grad {net.weight.grad}')

print(net.weight.data)
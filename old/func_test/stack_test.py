import torch

t1 = torch.arange(1, 10, 1).reshape(3, 3)
t2 = torch.arange(10, 100, 10).reshape(3, 3)

print(t1, t2)

print('dim0 = ', torch.stack((t1, t2), dim=0))
print('dim1 = ', torch.stack((t1, t2), dim=1))
print('dim2 = ', torch.stack((t1, t2), dim=2))

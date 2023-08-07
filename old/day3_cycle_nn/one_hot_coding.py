import torch


x = torch.arange(6).reshape(2, 3)
print("one hot before: ", x)
y = torch.nn.functional.one_hot(x, 8)
print("one hot post: ", y)

# y = torch.nn.functional.one_hot(x, 5) one_hot不能编码范围大于编码维度
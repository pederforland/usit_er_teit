import torch

t = torch.Tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
cdf = torch.cumsum(t, dim=1)
print(cdf)
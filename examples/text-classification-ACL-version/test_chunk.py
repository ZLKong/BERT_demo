import torch
b = torch.Tensor([[4,5,7], [3,9,8], [9,6,7]])
d1 = torch.chunk(b,3,dim=1)
d = torch.chunk(b,3,dim=0)
print('d1',d1)
print('d',d)

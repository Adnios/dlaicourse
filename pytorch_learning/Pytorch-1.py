import torch
import numpy as np

t = torch.Tensor()
type(t)

print(t.dtype)
print(t.device)
print(t.layout)

device = torch.device('cuda:0')
device

data = np.array([1, 2, 3])
type(data)

torch.Tensor(data)

torch.tensor(data)

torch.as_tensor(data)

torch.from_numpy(data)

torch.eye(2)

torch.zeros(2, 2)

torch.ones(2, 2)

torch.rand(2, 2)

o1 = torch.Tensor(data)
o2 = torch.tensor(data)
o3 = torch.as_tensor(data)
o4 = torch.from_numpy(data)

print(o1)
print(o2)
print(o3)
print(o4)

print(o1.dtype)
print(o2.dtype)
print(o3.dtype)
print(o4.dtype)

torch.get_default_dtype()

torch.tensor(np.array([1, 2, 3]))
torch.tensor(np.array([1., 2., 3.]))
torch.tensor(np.array([1, 2, 3]), dtype=torch.float64)

# Reshaping
t = torch.tensor([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
], dtype=torch.float32)
t.size()


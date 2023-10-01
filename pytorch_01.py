import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

my_tensor = torch.tensor([
				[1, 2, 3],
				[4, 5, 6]
], device=device)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.shape)

# other common initialization methods

x = torch.empty(size=(3, 3))  # we do know what the value of tensor x
print(x)  # return random numbers

x = torch.zeros((3, 3))
print(x)

x = torch.rand(3, 3)
print(x)

x = torch.arange(start=0, end=5, step=2)
print(x)

x = torch.linspace(start=0.1, end=1, steps=10)
print(x)

x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
print(x)

x = torch.diag(torch.ones(4))
print(x)

# convert tensor to (int, float, double)

tensor = torch.arange(4)
print(tensor)
print(tensor.bool())
tensor = torch.randint(low=0, high=20, size=(3, 3))
print(tensor.bool())

np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
print(np_array)

### Pytorch Math and Comparison Operations

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

z1 = torch.empty(len(x))
torch.add(x, y, out=z1)
print(z1)

z2 = x - y
print(z2)

z = x.pow(2)
z = x ** 2
print(z)

x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)
x3 = x1.mm(x2)
print(x3)

z = x * y
print(z)

z = torch.dot(x, y)
print(z)

batch = 32
n = 10
m = 20
p = 30

# tensor1 = torch.rand((n,m)) # return random tensor
tensor1 = torch.rand((batch, n, m))  # Batch Matrix Mutliplication

x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

sum_x = torch.sum(x, dim=0)
print(sum_x)
value, index = torch.max(x, dim=0)
print(value, " ", index)

z = torch.argmax(x1, dim=0)
print(z)

# Indexing

batch_size = 10
features = 25

x = torch.rand((batch_size, features))

print(x)
print(x[0].shape)
print(x[:, 0].shape)
print(x[2, 0:10])  # select row 3, and columns [0,1,2, ...,9]

x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])

x = torch.rand((3, 5))
# row = 1, col = 4 -- row =0, col = 0
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x)
print(x[rows, cols])

x = torch.arange(10)
print(torch.where(x > 5, x, x * 2))
print(torch.tensor([9]).unique())
print(x.ndimension())
print(x.numel())  # count elements in x

# Reshaping

x = torch.arange(9)
x_3x3 = x.reshape(3, 3)
print(x_3x3)
x_3x3 = x.view(3, 3)  # require a variables that continuing save in memory
print(x_3x3)

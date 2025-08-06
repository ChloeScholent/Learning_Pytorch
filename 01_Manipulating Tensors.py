import torch  
import pandas as pd 
import numpy as  np 
import matplotlib.pyplot as plt


#Tensors operations 
# #(addition, subtraction, multiplication (element-wise), division, matrix multiplication)
#3 ways to do the same operation
#Addition: adds 10
tensor = torch.tensor([1, 2, 3])
tensor_2 = torch.tensor([4, 5, 6])
print(tensor + 10)
print(torch.add(tensor, 10))
print(tensor.add(10))

#multiplication: multiply by 10 (scalar multiplication)
print(tensor*10)
print(torch.mul(tensor, 10))

#Subtraction 10
print(tensor-10)
print(torch.sub(tensor, 10))

#Division by 10
print(tensor/10)
print(torch.div(tensor, 10))


#MATRIX MULIPLICATION (dot product of rows and columns)
matrix_1 = torch.tensor([[3, 4, 2]])
matrix_2 = torch.tensor ([[13,9,7,15], [8,7,4,6], [6,4,0,3]])

print(torch.matmul(matrix_1, matrix_2))
print(torch.mm(matrix_1, matrix_2))

#Switching dimension of a tensor

print(matrix_2.T)
x = torch.mm(matrix_2, matrix_2.T)
print(x)

x = x.type(torch.float32)

#Tensor aggregation (min, max, mean, sum etc)

print(x.min(), x.max(), x.mean(), x.sum(), x.median())

#Find position in the tensor for the max and the min

print(x.argmin(), x.argmax())


#Reshapping, stacking, squeezing, unsqueezing tensors
#1. to a defined shape
#view: returns a view of an input of certain shape but keep same memory as the original tensor
#2. stacking tensors on top of each other (vstack) or side by side (hstack)
#3.squeeze = remove 1 dimension
#4.unsqueeze = add 1 dimension to target tensor
#5. Permute : return a view of the input with dimensions permuted in a certain way

#change the shape
y = torch.arange(1., 11.)
print(y, y.shape)

y_reshaped = y.reshape(5,2) #needs to be compatible regarding the number of elements in the tensor
print(y_reshaped)

#change the view
z = x.view(1,9)
print(x, z)

#z has x in memory, so, changing z will also change x, but can have different shapes

z[:, 0] = 5

print(x, z)

#stacking tensors
hstacked = torch.hstack([x, x, x])
vstacked = torch.vstack([x, x, x])

print(f'hstacked={hstacked} \n vstacked={vstacked}')


#indexing on a tensor

indexed_tensor = torch.arange(1,10).reshape(1,3,3)


#Pytorch and Numpy
#array to tensor
array = np.arange(1.0, 8.0)

tensor = torch.from_numpy(array)

print(array, tensor)
#np default dtype is float64 while pytorch dtype is float32. pytorch reflects np dtype unsless specified



#tensor to array

tensor_2 = torch.ones(2,2)
numpy_tensor = tensor_2.numpy()

print(tensor_2, numpy_tensor)


print(indexed_tensor)

print(indexed_tensor[0, 2, 2])

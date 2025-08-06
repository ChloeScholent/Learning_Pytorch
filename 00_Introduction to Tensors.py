import torch  
import pandas as pd 
import numpy as  np 
import matplotlib.pyplot as plt

#print(torch.__version__)



### Introduction to Tensors
#https://docs.pytorch.org/docs/stable/tensors.html

## Creating Tensors (a class)


#Scalar#
print("A little about SCALAR")
scalar = torch.tensor(7)
print(scalar)

#Get tensor back as python int
print(scalar.item())

#Dimension and shape of the tensor
print(scalar.ndim)
print(scalar.shape)


#Vector#
print("A little about VECTOR")
vector = torch.tensor([5,2])

print(vector)
print(vector[0])
print(vector[1])
print(vector.ndim)
print(vector.shape)

#MATRIX#
print("A little about MATRIX")
MATRIX = torch.tensor([[5,9], [7,6]])

print(MATRIX)
print(MATRIX[0])
print(MATRIX[1])
print(MATRIX.ndim)
print(MATRIX.shape)


#VECTOR#
print("A little about TENSOR")

TENSOR = torch.tensor([[[1,2,3],
                        [4,5,6],
                        [7,8,9]],[[1,2,2], [3,4,2], [5,6,2]]])

print(TENSOR)
print(TENSOR.ndim, TENSOR.shape)
# tensor = 2 blocks of 3X3 for the example provided

#To create a random tensor with any provided shape
random_tensor = torch.rand(2,3,3)
print(random_tensor)

random_image_size_tensor = torch.rand(size= (224, 224, 3)) #height, width, colour channel RGB
print(random_image_size_tensor.shape, random_image_size_tensor.ndim)


# Zeros and ones

zeros = torch.zeros(2,3,3)
print(zeros)

ones = torch.ones(2,2)
print(ones)

#Possibility to multiply tensors but they need to be the same size

print(zeros*random_tensor)
print


#Create a range of tensors and tensors-like

print(torch.arange(0, 10))
#adding a third argument = adding the step

step = torch.arange(0, 999, 35)
print(step)

tensor_like = torch.rand_like(zeros)
print(tensor_like)
#works with torch.ones_like and torch.zeros_like
#torch.rand_like need a float input, so it won't work with tensors defined with int
#zeros and ones tensors are considered float; so it works with them

print(vector.dtype, zeros.dtype, ones.dtype, TENSOR.dtype)

#Tensor datatypes (new arguments for tensor creation)
#Datatype is one of the 3 big errors that can be run into in Pytorch:
#1. Tensor not right datatype
#2. Tensor not right shape
#3. Tensor not on the right device

datatypes= torch.tensor([3.0, 6.0, 9.0], dtype=None, # what datatype is the tensor
                                            device=None, #cuda, cpu; on what the tensor runs
                                            requires_grad=False) #whether or not to track gradients with this tensors operations

print(datatypes.dtype)

datatypes = datatypes.type(torch.float16)
print(datatypes.dtype)


#Getting information from tensors
a = random_tensor.dtype
b = random_tensor.shape
c = random_tensor.device
print(f'My tensor has a {a} datatype, a shape of {b} and runs on {c}')







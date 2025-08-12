import torch  
import pandas as pd 
import numpy as  np 
import matplotlib.pyplot as plt


#Reproduciility

# using random seed (pseudo random generation)

random_tensor_A = torch.rand(3,3)
random_tensor_B = torch.rand(3,3)

print(random_tensor_A == random_tensor_B)

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(4,5)
torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(4,5)

print(random_tensor_C == random_tensor_D)

#torch.manual_seed(RANDOM_SEED) needs to be added everytime a new tensor is created, otherwise it will not work.abs
#This allows different randdoms seeds to be used

#Check if GPU is available for Pytorch
print(torch.cuda.is_available())

#Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

#Count nb of gpu
print(torch.cuda.device_count())

#Putting tensors and models on GPU
tensor_CPU = torch.rand(5,7)
print(tensor_CPU.device)

#Move to GPU
tensor_to_GPU = tensor_CPU.to(device)
print(tensor_to_GPU)

#Move back to CPU (for Numpy for instance)
tensor_back_to_CPU = tensor_to_GPU.cpu()
print(tensor_back_to_CPU.device)

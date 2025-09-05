#TODO
'''
Dataset
Dataset split
data loader (batch)
Faire le model
faire loss/accuracy/optimizer
Train le model
Eval le model
print confusion matrix et classification report
'''

import torch
from torch import nn
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Device:', device)


#DATASET
print('Loading dataset...')
mnist_dataset = MNIST(root="data/", download=True, transform= transforms.ToTensor())
input_size = 28*28
num_classes = 10
batch_size = 128

train_data, test_data = random_split(mnist_dataset, [40000, 20000])

train_loader = DataLoader(train_data, batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size, shuffle=False)

print(train_loader, test_loader)
print('Dataset loaded successfully ! \n')
#MODEL
class MNISTModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.layer_size = [input_size, 512, 250, 128, 64, num_classes]

        self.layers.append(nn.Flatten())
        for i in range(len(self.layer_size)-2):
            self.layers.append(nn.Linear(self.layer_size[i], self.layer_size[i+1]))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(self.layer_size[-2], self.layer_size[-1]))        

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

print('Creating model...')
Mnist_Model = MNISTModel(input_size=input_size, num_classes=num_classes).to(device)
print("\n")
print("Here are the model's parameters: \n", Mnist_Model)


#LOSS & ACCURACY & OPTIMIZER

loss_fn = nn.CrossEntropyLoss() #Already does softmax, not needed in the model
optimizer = torch.optim.Adam(params=Mnist_Model.parameters(), lr = 0.001)

# def accuracy_fn(outputs, labels):
#     preds = torch.argmax(outputs, dim=1)
#     return (torch.sum(preds == labels)/ len(preds))*100 
# 
# torch.tensor(torch.sum(preds == labels).item()/ len(preds)*100)

def accuracy_fn(outputs, labels):
    pred = torch.argmax(outputs, dim=1)
    return (torch.sum(pred == labels).item()/ len(pred))*100

#TRAIN

epochs = 20
#torch.manual_seed(12)

print('Training...')

for epoch in range(epochs):
    for input_tensors, output_tensors in train_loader:
        inputs = input_tensors.to(device)
        labels = output_tensors.to(device)

        Mnist_Model.train()
        
        outputs = Mnist_Model(inputs)
        loss = loss_fn(outputs, labels)
        accuracy = accuracy_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#TESTING
    Mnist_Model.eval()
    with torch.inference_mode():
        for test_inputs, test_labels in test_loader:
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)


            test_outputs = Mnist_Model(test_inputs)
            test_loss = loss_fn(test_outputs, test_labels)
            test_accuracy =  accuracy_fn(test_outputs, test_labels)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {accuracy:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_accuracy:.2f}%")

print('Training completed !')

#Classification report & Confusion Matrix

Mnist_Model.eval()

all_preds = []
all_labels = []

with torch.inference_mode():
    for test_inputs, test_labels in test_loader:
        test_inputs = test_inputs.to(device)
        test_labels = test_labels.to(device)

        test_outputs = Mnist_Model(test_inputs)
        preds = torch.argmax(test_outputs, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(test_labels.cpu().numpy())

# Concatenate all predictions
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Print confusion matrix & classification report
print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))
print("\nClassification Report:\n", classification_report(all_labels, all_preds))



#Saving the model
MODEL_PATH = Path("Models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "MNIST_Classification.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#save the model state dictionary
print(f'Saving model to {MODEL_SAVE_PATH}')
torch.save(obj=Mnist_Model.state_dict(), f=MODEL_SAVE_PATH)






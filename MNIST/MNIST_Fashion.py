###TODO
"""
Load dataset
Split train/test subsets
Create model
Optim, loss (acc)
train
eval
confusion matrix/classification report
save the model
"""

import torch
from torch import nn
import torchvision
from torchvision.datasets import FashionMNIST
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Device:', device)
print('\n')
#DATASET

print('Loading FashionMNIST dataset...')
print('\n')


# image_tensor, label = FashionMNIST_dataset[0]
# print(image_tensor, label)

# image, label = FashionMNIST_dataset[0]
# plt.imshow(image[0], cmap = 'gray')
# plt.show()
# print('Label,', label)

input_size = 28*28
num_classes = 10
batch_size = 128

train_data = FashionMNIST(root="data/", train=True, download=True, transform=transforms.ToTensor())
test_data = FashionMNIST(root="data/", train=False, download=True, transform=transforms.ToTensor())


train_loader = DataLoader(train_data, batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size, shuffle=False)

print(f'train_loader: {train_loader} \ntest_loader: {test_loader}')
print('\n')
print('Dataset loaded successfully !')


#MODEL
print('Creation of the model...')

class FashionMNISTCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
                    )

        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64*24*24, out_features=num_classes)
        )

    def forward(self, x):
        out = self.cnn_layers(x)
        out = self.linear_layers(out)
        return out

Fashion_Model_CNN = FashionMNISTCNN(num_classes).to(device)

print(Fashion_Model_CNN)
print('\nModel created successfully !')
print('\n')

#TRAINING

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=Fashion_Model_CNN.parameters(), lr=0.001)

def accuracy_fn(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    acc = (torch.sum(preds == labels).item()/len(preds))*100
    return acc

epochs = 31

print('Training...')
print('\n')

for epoch in range(epochs):
    losses = []
    test_losses = []
    accs = []
    test_accs = []
    for train_input, train_labels in train_loader:
        train_input = train_input.to(device)
        train_labels = train_labels.to(device)

        Fashion_Model_CNN.train()

        outputs = Fashion_Model_CNN(train_input)
        loss = loss_fn(outputs, train_labels) 
        losses.append(loss.item())
        acc = accuracy_fn(outputs, train_labels)
        accs.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses = sum(losses)/len(losses)
    train_acc = sum(accs)/len(accs)
    writer.add_scalar('Accuracy/Train', train_acc, epoch)
    writer.add_scalar('Loss/Train', losses, epoch)
#EVALUATION
    Fashion_Model_CNN.eval()
    with torch.inference_mode():
        for test_inputs, test_labels in test_loader:
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)

            test_outputs = Fashion_Model_CNN(test_inputs)
            test_loss = loss_fn(test_outputs, test_labels)
            test_losses.append(test_loss.item())
            test_acc = accuracy_fn(test_outputs, test_labels)
            test_accs.append(test_acc)
        test_accs = sum(test_accs)/len(test_accs)
        test_loss = sum(test_losses)/len(test_losses)
    writer.add_scalar('Loss/Test', test_loss, epoch)
    writer.add_scalar('Accuracy/Test', test_accs, epoch)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch} | Loss: {losses:.5f}, Accuracy: {sum(accs)/len(accs):.2f}% | Test loss: {(sum(test_losses)/len(test_losses)):.5f}, Test acc: {test_accs:.2f}%')



print('\n')
print('Training completed')

writer.flush()
writer.close()

#Confusion matrix and classification report

Fashion_Model_CNN.eval()

all_preds = []
all_labels = []

with torch.inference_mode():
    for test_inputs, test_labels in test_loader:
        test_inputs = test_inputs.to(device)
        test_labels = test_labels.to(device)

        test_outputs = Fashion_Model_CNN(test_inputs)
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

MODEL_NAME = "Fashion_MNIST_CNN_Classification.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#save the model state dictionary
print(f'Saving model to {MODEL_SAVE_PATH}')
torch.save(obj=Fashion_Model_CNN.state_dict(), f=MODEL_SAVE_PATH)



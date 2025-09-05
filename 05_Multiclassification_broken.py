#https://www.kaggle.com/code/geekysaint/solving-mnist-using-pytorch


import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"

mnist_dataset = MNIST(root="data/", download=True, transform= transforms.ToTensor())

# image, label = dataset[10]
# plt.imshow(image, cmap = 'gray')
# plt.show()
# print('Label,', label)

# image_tensor, label = mnist_dataset[0]
# print(image_tensor, label)

input_size = 28*28
num_classes = 10

train_data, test_data = random_split(mnist_dataset, [40000, 20000])
#print(len(train_data), len(test_data))
# train_data = train_data.to(device)
# test_data = test_data.to(device)

batch_size =128

train_loader = DataLoader(train_data, batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size, shuffle=False)


class MNISTModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.layer_size = [input_size, 512, 250, 250, 250, 128, 64, num_classes]

        for i in range(len(self.layer_size)-2):
            self.layers.append(nn.Linear(self.layer_size[i], self.layer_size[i+1]))
            #self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(self.layer_size[-2], self.layer_size[-1]))
        

    def forward(self, x):
        x = x.reshape(-1, 784)
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images) #generate predictions
        loss = F.cross_entropy(out, labels) #Calculate loss
        return loss
        
model_0 = MNISTModel(input_size, num_classes).to(device)


for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = model_0(images)
    break

for test_images, test_labels in test_loader:
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)
    break
    
probs = F.softmax(outputs, dim=1)
max_probs, preds = torch.max(probs, dim = 1)


def accuracy_fn(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item()/ len(preds)*100)

loss_fn = F.cross_entropy
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.01)
epochs = 100

torch.manual_seed(12)
# train_time_start_on_gpu = timer()

#TRAIN
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    train_loss = 0
    for batch, (X,y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)
        
        model_0.train()

        #FORWARD PASS
        outputs = model_0(X)

        #LOSS
        loss = loss_fn(outputs, y)
        train_loss += loss
        acc = accuracy_fn(outputs, y)

        #OPTIM
        optimizer.zero_grad()

        #Backward

        loss.backward()

        optimizer.step()

        # if batch % 400 == 0:
        #     print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

    train_loss /= len(train_loader)
    
    #Testing
    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)

            test_logits = model_0(X)

            test_loss = loss_fn(test_logits, y)
            test_acc = accuracy_fn(test_logits, y)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")






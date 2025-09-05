import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
import requests
from pathlib import Path

n_samples = 1000

X, y = make_circles(n_samples, noise=0.03, random_state=12)

# Make DataFrame of circle data
import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0],
    "X2": X[:, 1],
    "label": y
})
print(circles.head(10))

print(circles.label.value_counts())

plt.scatter(x=X[:,0], y=X[:,1], c=y, cmap = plt.cm.RdYlBu)
#plt.show()

#Trying to classify what circle a dot is part of with a toy dataset

#Check input and output shapes first of all !

print(X.shape, y.shape)

#Turn data into tensors from numpy array

print(type(X), type(y))

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

print(type(X), type(y))

#Create a random split of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)


#Building the model

#Device agnostic code

device = "cuda" if torch.cuda.is_available() else "cpu"


class CircleModelV0(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layer_size = [input_size,10,10,num_classes]

        for i in range(len(self.layer_size)-2):
            self.layers.append(nn.Linear(self.layer_size[i], self.layer_size[i+1]))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(self.layer_size[-2], self.layer_size[-1]))

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

        
model_0 = CircleModelV0(2, 1).to(device)
print(model_0)
#Loss function
# loss_fn = nn.BCELoss() # BCELoss = no sigmoid built-in
loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss = sigmoid built-in
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

#Calculate accuracy of the model

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct/len(y_pred)) *100
    return acc

# 'LOGITS' corresponds to the raw outputs of the model before training
# View the frist 5 outputs of the forward pass on the test data
y_logits = model_0(X_test.to(device))[:5]
print(y_logits)

y_pred_probs = torch.sigmoid(y_logits)

# #Can be rounded for predictions
y_preds = torch.round(y_pred_probs)

#full line
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

#Check for equality
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

#get rid of extra dimension
print(y_preds.squeeze())

#Training and testing loop


torch.manual_seed = 12

epochs = 1000

#Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

#TRAIN
for epoch in range(epochs):
    model_0.train()

    #FORWARD PASS
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    #LOSS
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    #OPTIM
    optimizer.zero_grad()

    #Backward

    loss.backward()

    optimizer.step()

    #Testing

    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")


# Download helper functions from Learn PyTorch repo (if not already downloaded)
# if Path("helper_functions.py").is_file():
#   print("helper_functions.py already exists, skipping download")
# else:
#   print("Downloading helper_functions.py")
#   request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
#   with open("helper_functions.py", "wb") as f:
#     f.write(request.content)

# from helper_functions import plot_predictions, plot_decision_boundary

# Plot decision boundaries for training and test sets
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plot_decision_boundary(model_0, X_train, y_train)
# plt.subplot(1, 2, 2)
# plt.title("Test")
# plot_decision_boundary(model_0, X_test, y_test)
#plt.show()


MODEL_PATH = Path("Models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Pytorch Classification.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#save the model state dictionary
print(f'Saving model to {MODEL_SAVE_PATH}')
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)
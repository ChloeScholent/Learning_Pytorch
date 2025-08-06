import torch
from torch import nn #nn contains all of Pytorch building blocks for neural networks
import matplotlib.pyplot as plt

#Preparinf and loading data

#create some 'known' data using linear regression formula (straight line with known parameters)

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

#Linear regression formula: y= a+ b*X
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(X[:10], y[:10])

##Splitting data into training and test sets
train_split = int(0.8*len(X))
print(train_split)

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))


#How to better visualize the datasets

def plot_predictions(train_data=X_train, 
                        train_labels=y_train,
                        test_data=X_test, 
                        test_labels=y_test, 
                        predictions=None):
    """
    Plots training data, test data and compares predictions
    """
    plt.figure(figsize=(10,7))

    #Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    #Plot training data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    #Are there predictions ?
    if predictions is not None:
        #Plot the predictions if they exist
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    #Show the legend
    plt.legend(prop={"size": 14})
    plt.show()


#plot_predictions()


##Building our first Pytorch model










from sklearn.datasets import make_regression
from skorch import NeuralNetRegressor
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F
from torch import optim

# This class could be any name.
# nn.Module is needed to enable grid searching of parameters
# with skorch later.
from skorch.callbacks import EarlyStopping


class MyNeuralNet(nn.Module):
    def __init__(self, num_units=10, nonlin=F.relu):
        super(MyNeuralNet, self).__init__()
        self.num_units = num_units
        self.nonlin = nonlin
        self.name = "This is Victor's model before hidden layer is added"
        self.dense0 = nn.Linear(4, num_units)
        self.nonlin = nonlin
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = F.relu(self.dense1(X))
        X = self.output(X)
        return X


def buildModel(x, y):
    # Trains the Neural Network with fixed hyperparameters
    # The Neural Net is initialized with fixed hyperparameters
    myNetwork = MyNeuralNet(num_units=10)

    # Define learning rate, max_epochs and momentum
    # separately from the network.
    params = {
        'lr': 0.001,  # Learning rate
        'max_epochs': 1000,  # Maximum number of epochs
        'optimizer': optim.SGD,
        'optimizer__momentum': 0.9,
    }

    net_regr = NeuralNetRegressor(
        myNetwork,
        **params,
        callbacks=[EarlyStopping(patience=60)],
    )
    model = net_regr.fit(x, y)
    return model


def evaluateModel(model, X_test, y_test, scalerY):
    print(model)
    y_pred_scaled = model.predict(X_test)
    y_pred = scalerY.inverse_transform(y_pred_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("RMSE: " + str(rmse))


# Prep the data.
# This is a toy dataset for regression, 1000 data points with 20 features each
import torch
import pandas as pd

df = pd.read_csv('C:/datasets/petrol_consumption.csv')
X = df.copy()
del X['Petrol_Consumption']
y = df['Petrol_Consumption']

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler

scalerX = StandardScaler()
scaledXTrain = scalerX.fit_transform(X_train)
scaledXTest = scalerX.transform(X_test)

scalerY = StandardScaler()
scaledYTrain = scalerY.fit_transform(np.array(y_train).reshape(-1, 1))

# Must convert the data to PyTorch tensors
X_train_tensor = torch.tensor(scaledXTrain, dtype=torch.float32)
X_test_tensor = torch.tensor(scaledXTest, dtype=torch.float32)
y_train_tensor = torch.tensor(list(scaledYTrain), dtype=torch.float32)
y_test_tensor = torch.tensor(list(y_test), dtype=torch.float32)

# Build the model.
model = buildModel(X_train_tensor, y_train_tensor)
print(model.get_params())
# Evaluate the model.
evaluateModel(model, X_test_tensor, y_test_tensor, scalerY)

import matplotlib.pyplot as plt


def drawLossPlot(net):
    plt.plot(net.history[:, 'train_loss'], color='blue', label='train')
    plt.plot(net.history[:, 'valid_loss'], color='orange', label='val')
    plt.legend()
    plt.show()


drawLossPlot(model)

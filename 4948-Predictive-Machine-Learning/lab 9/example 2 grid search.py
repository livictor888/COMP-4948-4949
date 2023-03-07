from sklearn.datasets             import make_regression
from sklearn.pipeline import Pipeline
from   skorch                     import NeuralNetRegressor
import torch.nn as nn
import numpy    as np
from sklearn.model_selection import train_test_split, GridSearchCV
from   sklearn.metrics          import mean_squared_error
import torch.nn.functional as   F

# This class could be any name.
# nn.Module is needed to enable grid searching of parameters
# with skorch later.
from torch import optim

class MyNeuralNet(nn.Module):
    def __init__(self, num_neurons):
        super(MyNeuralNet, self).__init__()

        self.num_units = num_neurons
        self.dense0 = nn.Linear(4, num_neurons)
        self.dense1 = nn.Linear(num_neurons, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, X, **kwargs):
        X = F.relu(self.dense0(X))
        X = F.relu(self.dense1(X))
        X = self.output(X)
        return X

def buildModel(x, y):
    nn = NeuralNetRegressor(MyNeuralNet, verbose=1, train_split=False)
    # Trains the Neural Network with fixed hyperparameters
    pipeline = Pipeline([ ('nn', nn)])

    params = {
      'nn__max_epochs': [30,50,60],
      'nn__lr': [0.01, 0.015, 0.007],
      'nn__module__num_neurons': [15,20,25],
      'nn__optimizer': [optim.Adam, optim.SGD, optim.RMSprop]}

    # The grid search module is instantiated
    gs = GridSearchCV(pipeline, param_grid=params, refit=True, cv=3,
                    scoring='neg_mean_squared_error', verbose=1)

    return gs.fit(x, y)

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

X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
scalerX      = StandardScaler()
scaledXTrain = scalerX.fit_transform(X_train)
scaledXTest  = scalerX.transform(X_test)

scalerY      = StandardScaler()
scaledYTrain = scalerY.fit_transform(np.array(y_train).reshape(-1,1))

# Must convert the data to PyTorch tensors
X_train_tensor = torch.tensor(scaledXTrain, dtype=torch.float32)
X_test_tensor  = torch.tensor(scaledXTest, dtype=torch.float32)
y_train_tensor = torch.tensor(list(scaledYTrain), dtype=torch.float32)
y_test_tensor = torch.tensor(list(y_test), dtype=torch.float32)

# Build the model.
model   = buildModel(X_train_tensor, y_train_tensor)

print("Best parameters:")
print(model.best_params_)

# Evaluate the model.
evaluateModel(model.best_estimator_, X_test_tensor, y_test_tensor, scalerY)


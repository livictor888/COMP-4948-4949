

# Exercise 2
import torch
import pandas as pd
from sklearn.datasets           import make_classification
from   torch                      import optim
from   skorch                     import NeuralNetClassifier
import torch.nn as nn
import numpy    as np
from   sklearn.model_selection  import train_test_split
from   sklearn.metrics          import classification_report

df = pd.read_csv('C:/datasets/bill_authentication.csv')
X = df.copy()
del X['Class']
y = df['Class']

# This class could be any name.
# nn.Module is needed to enable grid searching of parameters
# with skorch later.
class MyNeuralNet(nn.Module):
    # Define network objects.
    # Defaults are set for number of neurons and the
    # dropout rate.
    def __init__(self, num_neurons=10, dropout=0.1):
        super(MyNeuralNet, self).__init__()
        # 1st hidden layer.
        # nn. Linear(n,m) is a module that creates single layer
        # feed forward network with n inputs and m output.
        self.dense0         = nn.Linear(20, num_neurons)
        self.activationFunc = nn.ReLU()

        # Drop samples to help prevent overfitting.
        self.dropout        = nn.Dropout(dropout)

        # 2nd hidden layer.
        self.dense1         = nn.Linear(num_neurons, num_neurons)

        # Output layer.
        self.output         = nn.Linear(num_neurons, 2)

        # Softmax activation function allows for multiclass predictions.
        # In this case the prediction is binary.
        self.softmax        = nn.Softmax(dim=-1)

    # Move data through the different network objects.
    def forward(self, x):
        # Pass data from 1st hidden layer to activation function
        # before sending to next layer.
        X = self.activationFunc(self.dense0(x))
        X = self.dropout(X)
        X = self.activationFunc(self.dense1(X))
        X = self.softmax(self.output(X))
        return X

def buildModel(x, y):
  # Trains the Neural Network with fixed hyperparameters
  # The Neural Net is initialized with fixed hyperparameters
  myNetwork = MyNeuralNet(num_neurons=10, dropout=0.1)
  nn = NeuralNetClassifier(myNetwork, max_epochs=10,
                           lr=0.01, batch_size=12,
                           optimizer=optim.RMSprop)
  model = nn.fit(x, y)
  return model

def evaluateModel(model, X_test, y_test):
    print(model)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

# Prep the data.
X, y = make_classification(1000, 20, n_informative=10, random_state=0)
X    = X.astype(np.float32)
y    = y.astype(np.int64)

X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
scalerX      = StandardScaler()
scaledXTrain = scalerX.fit_transform(X_train)
scaledXTest  = scalerX.transform(X_test)

# Must convert the data to PyTorch tensors
X_train_tensor = torch.tensor(scaledXTrain, dtype=torch.float32)
X_test_tensor  = torch.tensor(scaledXTest, dtype=torch.float32)
y_train_tensor = torch.tensor(list(y_train), dtype=torch.long)

# Build the model.
model   = buildModel(X_train_tensor, y_train_tensor)

# Evaluate the model.
evaluateModel(model, X_test_tensor, y_test)

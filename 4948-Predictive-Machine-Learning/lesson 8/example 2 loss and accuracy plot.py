from sklearn.datasets           import make_classification
from   torch                      import optim
from   skorch                     import NeuralNetClassifier
import torch.nn as nn
import numpy    as np
from   sklearn.model_selection  import train_test_split
from   sklearn.metrics          import classification_report

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

from skorch.callbacks           import EpochScoring
def buildModel(X_train, y_train):
    num_neurons = 25   # hidden layers
    net = NeuralNetClassifier(MyNeuralNet( num_neurons), max_epochs=200,
        lr=0.001, batch_size=100, optimizer=optim.RMSprop,
        callbacks=[EpochScoring(scoring='accuracy',
        name='victor_acc', on_train=True)] )
    # Pipeline execution
    model = net.fit(X_train, y_train)
    return model, net


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

# Build the model.
model, net = buildModel(X_train, y_train)


# Evaluate the model.
evaluateModel(model, X_test, y_test)

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 30})

def drawLossPlot(net):
    plt.plot(net.history[:, 'train_loss'], color='blue', label='train')
    plt.plot(net.history[:, 'valid_loss'], color='orange', label='val')
    plt.legend()
    plt.show()

def drawAccuracyPlot(net):
    plt.plot(net.history[:, 'victor_acc'], color='blue', label='train')
    plt.plot(net.history[:, 'valid_acc'], color='orange', label='val')
    plt.legend()
    plt.show()

drawLossPlot(net)
drawAccuracyPlot(net)

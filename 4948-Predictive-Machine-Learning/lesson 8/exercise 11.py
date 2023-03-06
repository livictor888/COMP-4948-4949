

# Exercise 11
# Builds a PyTorch neural network and does not perform a grid search. Includes early stopping.
import pandas as pd
from skorch.callbacks           import EarlyStopping

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def getCustomerSegmentationData():
    df = pd.read_csv('C:/datasets/CustomerSegmentation.csv')
    df = pd.get_dummies(df, columns=[
        'Gender','Ever_Married',
        'Graduated','Profession','Spending_Score', 'Var_1'])
    df['Segmentation'] = df['Segmentation'].replace({'A': 0, 'B':1, 'C':2, 'D':3})
    print(df['Segmentation'].value_counts())
    X = df.copy()
    del X['Segmentation']
    y = df['Segmentation']
    return X, y

X, y = getCustomerSegmentationData()

import sklearn
import torch.nn as nn
import torch
from sklearn.datasets import load_iris
from   torch                    import optim
from   skorch                   import NeuralNetClassifier
from sklearn.preprocessing      import StandardScaler
from   sklearn.metrics          import classification_report
from skorch.callbacks           import EpochScoring

# Split and scale the data.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test  = torch.tensor(X_test_scaled, dtype=torch.float32)

# # Must have y in tensor format with long data type.
# y_train = torch.tensor(y_train, dtype=torch.long)
# y_test = torch.tensor(y_test, dtype=torch.long)

class Net(nn.Module):
    def __init__(self, num_features, num_neurons, output_dim):
        super(Net, self).__init__()
        # 1st hidden layer.
        # nn. Linear(n,m) is a module that creates single layer of a
        # feed forward network with n inputs and m output.
        self.dense0 = nn.Linear(num_features, num_neurons)
        self.activationFunc = nn.ReLU()

        # Drop samples to help prevent overfitting.
        DROPOUT = 0.1
        self.dropout = nn.Dropout(DROPOUT)

        # 2nd hidden layer.
        self.dense1 = nn.Linear(num_neurons, output_dim)

        # Output layer.
        self.output = nn.Linear(output_dim, 4)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Pass data from 1st hidden layer to activation function
        # before sending to next layer.
        X = self.activationFunc(self.dense0(x))
        X = self.dropout(X)
        X = self.activationFunc(self.dense1(X))
        X = self.softmax(self.output(X))
        return X

def evaluateModel(model, X_test, y_test):
    print(model)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

def buildModel(X_train, y_train):
    input_dim   = 28    # how many Variables are in the dataset
    num_neurons = 25   # hidden layers
    output_dim  = 4    # number of classes

    net = NeuralNetClassifier(Net(
        input_dim, num_neurons, output_dim), max_epochs=1000,
        lr=0.001, batch_size=100, optimizer=optim.RMSprop,
        callbacks=[EpochScoring(scoring='accuracy', name='train_acc', on_train=True), EarlyStopping(patience=100)])
    model = net.fit(X_train, y_train)
    return model, net

model, net = buildModel(X_train, y_train)
evaluateModel(model, X_test, y_test)
print("Done")

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 30})
def drawLossPlot(net):
    plt.plot(net.history[:, 'train_loss'], color='blue', label='train')
    plt.plot(net.history[:, 'valid_loss'], color='orange', label='val')
    plt.legend()
    plt.show()

def drawAccuracyPlot(net):
    plt.plot(net.history[:, 'train_acc'], color='blue', label='train')
    plt.plot(net.history[:, 'valid_acc'], color='orange', label='val')
    plt.legend()
    plt.show()

drawLossPlot(net)
drawAccuracyPlot(net)
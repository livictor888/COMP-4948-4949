
# Example 7
# performs a RandomGridSearchCV for a PyTorch Neural Network.
import pandas as pd
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

class Net(nn.Module):
    NUM_FEATURES = 28
    OUTPUT_DIM   = 4

    def __init__(self, num_neurons):
        super(Net, self).__init__()
        # 1st hidden layer.
        # nn. Linear(n,m) is a module that creates single layer of a
        # feed forward network with n inputs and m output.
        self.dense0 = nn.Linear(self.NUM_FEATURES, num_neurons)
        self.activationFunc = nn.ReLU()

        # Drop samples to help prevent overfitting.
        DROPOUT = 0.1
        self.dropout = nn.Dropout(DROPOUT)

        # 2nd hidden layer.
        self.dense1 = nn.Linear(num_neurons, self.OUTPUT_DIM)

        # Output layer.
        self.output = nn.Linear(self.OUTPUT_DIM, self.OUTPUT_DIM)
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
    cm = pd.crosstab(y_test, y_pred,
                     rownames=['Actual'],
                     colnames=['Predicted'])
    print("Confusion matrix")
    print(cm)
    print(report)

from sklearn.pipeline           import Pipeline
from sklearn.model_selection    import RandomizedSearchCV
def buildModel(X_train, y_train):
    input_dim   = 28    # how many Variables are in the dataset
    num_neurons = 25   # hidden layers

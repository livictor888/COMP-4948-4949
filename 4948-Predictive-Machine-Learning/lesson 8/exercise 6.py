
# Exercise 6
# Grid search a PyTorch network with that predicts ‘Diagnosed’ using the fluDiagnosis.csv file.
from sklearn.datasets           import make_classification
from   torch                      import optim
from   skorch                     import NeuralNetClassifier
import torch.nn as nn
import numpy    as np
import pandas as pd
from   sklearn.model_selection  import train_test_split
from   sklearn.metrics          import classification_report

df = pd.read_csv('C:/datasets/fluDiagnosis.csv')
X = df.copy()
del X['Diagnosed']
y = df['Diagnosed']

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
        print("Dense layer type:")
        print(self.dense0.weight.dtype)

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
        print("X type: ")
        print(x.dtype)

        # Pass data from 1st hidden layer to activation function
        # before sending to next layer.
        X = self.activationFunc(self.dense0(x))
        X = self.dropout(X)
        X = self.activationFunc(self.dense1(X))
        X = self.softmax(self.output(X))
        return X

from sklearn.pipeline           import Pipeline
from sklearn.preprocessing      import StandardScaler
from sklearn.model_selection    import GridSearchCV
def buildModel(x, y):
    # Through a grid search, the optimal hyperparameters are found
    # A pipeline is used in order to scale and train the neural net
    # The grid search module from scikit-learn wraps the pipeline

    # The Neural Net is instantiated, none hyperparameter is provided
    nn = NeuralNetClassifier(MyNeuralNet, verbose=0, train_split=False)
    # The pipeline is instantiated, it wraps scaling and training phase
    pipeline = Pipeline([('scale', StandardScaler()), ('nn', nn)])

    # The parameters for the grid search are defined
    # Must use prefix "nn__" when setting hyperparamters for the training phase
    # Must use prefix "nn__module__" when setting hyperparameters for the Neural Net
    params = {
        'nn__max_epochs': [10, 20],
        'nn__lr': [0.1, 0.01],
        'nn__module__num_neurons': [5, 10],
        'nn__module__dropout': [0.1, 0.5],
        'nn__optimizer': [optim.Adam, optim.SGD, optim.RMSprop]}

    # The grid search module is instantiated
    gs = GridSearchCV(pipeline, params, refit=True, cv=3,
                      scoring='balanced_accuracy', verbose=1)

    return gs.fit(x, y)

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
model  = buildModel(X_train, y_train)

print("Victor's flu diagnosis network best parameters:")
print(model.best_params_)

# Evaluate the model.
evaluateModel(model.best_estimator_, X_test, y_test)
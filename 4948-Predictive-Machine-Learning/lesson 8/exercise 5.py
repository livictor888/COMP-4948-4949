
# Example 4
# How to fix 'RuntimeError: mat1 and mat2 must have the same dtype'
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
        self.dense0         = nn.Linear(3, num_neurons)
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
        x = x.to(torch.float32) # this line fixes the runtime error
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
    # pipeline = Pipeline([('scale', StandardScaler()), ('nn', nn)])
    pipeline = Pipeline([('nn', nn)])

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
    print("Victor's network model for college admissions")
    print(model)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

# # Prep the data.
# X, y = make_classification(1000, 20, n_informative=10, random_state=0)
# X    = X.astype(np.float32)
# y    = y.astype(np.int64)
# X_train, X_test, y_train, y_test =\
#     train_test_split(X, y, test_size=0.2)

######################### Data Prep Start
# Setup data.
import pandas as pd
import numpy as np
import torch

candidates = {'gmat': [780,750,690,710,680,730,690,720,
 740,690,610,690,710,680,770,610,580,650,540,590,620,
 600,550,550,570,670,660,580,650,660,640,620,660,660,
 680,650,670,580,590,690],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,
 3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,
 3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,
 3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,
 1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,
 5,1,2,1,4,5],
              'admitted': [1,1,1,1,1,1,0,1,1,0,0,1,
 1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,
 0,0,1]}

df = pd.DataFrame(candidates,columns= ['gmat', 'gpa',
                  'work_experience','admitted'])
y = np.array(df['admitted'])
X = df.copy()
del X['admitted']
X = X

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# define standard scaler
from sklearn.preprocessing     import StandardScaler
scaler = StandardScaler()

# transform data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test  = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long) # whole number needed
y_test  = torch.tensor(y_test, dtype=torch.long)  # for classification.
######################### End

# Build the model.
model  = buildModel(X_train, y_train)

print("Best parameters:")
print(model.best_params_)

# Evaluate the model.
evaluateModel(model.best_estimator_, X_test, y_test)
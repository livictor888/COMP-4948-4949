
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import warnings

warnings.filterwarnings(action='once')

PATH     = "C:\\datasets\\"
CSV_DATA = "housing.data"
df  = pd.read_csv(PATH + CSV_DATA,  header=None)


# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

print(df.head())
print(df.tail())
print(df.describe())

dataset = df.values

# split into input (X) and output (Y) variables
X = dataset[:, 0:13]
y = dataset[:, 13]

trainX, temp_X, trainY, temp_y = train_test_split(X, y, train_size=0.7)
valX, testX, valY, testY = train_test_split(temp_X, temp_y, train_size=0.5)

# Scale X and Y.
scX = StandardScaler()
scalerX = scX.fit(trainX)
trainX_scaled = scalerX.transform(trainX)
valX_scaled = scalerX.transform(valX)
testX_scaled = scalerX.transform(testX)

scY = StandardScaler()
trainY_scaled = scY.fit_transform(np.array(trainY).reshape(-1, 1))
testY_scaled = scY.transform(np.array(testY).reshape(-1, 1))
valY_scaled = scY.transform(np.array(valY).reshape(-1, 1))

# Build basic multilayer perceptron.
model1 = MLPRegressor(
    # 3 hidden layers with 150 neurons, 100, and 50.
    hidden_layer_sizes=(150, 100, 50),
    max_iter=50,  # epochs
    activation='relu',
    solver='adam',  # optimizer
    verbose=1)
model1.fit(trainX_scaled, trainY_scaled)


def showLosses(model):
    plt.plot(model.loss_curve_)
    plt.title("Loss Curve")
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()


def evaluateModel(model, testX_scaled, testY_scaled, scY):
    showLosses(model)
    scaledPredictions = model.predict(testX_scaled)
    y_pred = scY.inverse_transform(
        np.array(scaledPredictions).reshape(-1, 1))
    mse = metrics.mean_squared_error(testY_scaled, y_pred)
    rmse = np.sqrt(mse)
    print("RMSE: " + str(rmse))


evaluateModel(model1, valX_scaled, valY_scaled, scY)

# here is the new part.
# param_grid = {
#     'hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30)],
#     'max_iter':            [50, 100],
#     'activation':          ['tanh', 'relu'],
#     'solver':              ['sgd', 'adam'],
#     'alpha':               [0.0001, 0.05],
#     'learning_rate':       ['constant','adaptive'],
# }


# here is the new part.
param_grid = {
    'hidden_layer_sizes': [(150, 100, 50), (120, 80, 40), (100, 50, 30)],
    'max_iter': [50, 100],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.0001, 0.001, 0.005, 0.01, 0.1, 0.2]
}
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# n_jobs=-1 means use all processors.
# Run print(metrics.get_scorer_names()) for scoring choices.
model2 = MLPRegressor()
# gridModel = GridSearchCV(model2, param_grid, n_jobs=-1, cv=10,scoring='neg_mean_squared_error')

gridModel = RandomizedSearchCV(model2, param_distributions=param_grid, n_jobs=-1, scoring='neg_mean_squared_error')

gridModel.fit(trainX_scaled, trainY_scaled)

print("Best parameters")
print(gridModel.best_params_)
evaluateModel(gridModel.best_estimator_, valX_scaled, valY_scaled, scY)

# Evaluate both models with test (unseen) data.
print("\n*** Base model with test data: ")
evaluateModel(model1, testX_scaled, testY_scaled, scY)
print(model1.get_params())
print("\n*** Grid searched model with test data: ")
evaluateModel(gridModel.best_estimator_, testX_scaled, testY_scaled, scY)
print(gridModel.get_params())

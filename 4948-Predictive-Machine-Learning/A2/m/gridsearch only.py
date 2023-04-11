import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import warnings
from pathlib import Path
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn import metrics
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor


# Load and prepare dataset
PATH = Path("Tesla.csv")
df = pd.read_csv(PATH)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Define target variable and features
X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
y = df['Adj Close']

# Split the data
"""Remove random state param when happy with RMSE score"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""
Create NN Model
"""
from keras.optimizers import Adam, RMSprop


def create_nn_model(optimizer='adam', neurons=10, lr=0.001, activation='relu', initializer='he_normal'):
    model = Sequential()
    model.add(Dense(neurons, activation=activation, kernel_initializer=initializer, input_dim=5))
    model.add(Dense(neurons, activation=activation, kernel_initializer=initializer))
    model.add(Dense(1, activation='linear'))

    if optimizer == 'adam':
        opt = Adam(learning_rate=lr)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=lr)

    model.compile(optimizer=opt, loss='mean_squared_error')
    return model


# Define the grid search parameters
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'neurons': [10, 20, 30],
    'lr': [0.001, 0.01, 0.1],
    'activation': ['relu', 'tanh'],
    'initializer': ['he_normal', 'he_uniform']
}




"""
Perform Grid Search
"""
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


# Create a KerasRegressor model
nn_model = KerasRegressor(build_fn=create_nn_model, epochs=50, batch_size=16, verbose=0, optimizer='adam', neurons=10, lr=0.001, activation='relu', initializer='he_normal')


# Define the search space
search_space = {
    'optimizer': Categorical(['adam', 'rmsprop']),
    'neurons': Integer(10, 100),
    'activation': Categorical(['relu', 'tanh']),
    'lr': Real(1e-4, 1e-2, prior='log-uniform'),
}


# Perform the Grid search
grid_search = GridSearchCV(estimator=nn_model, param_grid=param_grid, n_jobs=-1, cv=3, scoring='neg_mean_squared_error')
grid_search_result = grid_search.fit(X_train_scaled, y_train)


# Summarize the results
print(f"Best: {grid_search_result.best_score_} using {grid_search_result.best_params_}")
means = grid_search_result.cv_results_['mean_test_score']
stds = grid_search_result.cv_results_['std_test_score']
params = grid_search_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f"{mean:.4f} ({stdev:.4f}) with: {param}")
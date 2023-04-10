import matplotlib.pylab as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import ElasticNet
import statsmodels.api as sm
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor

from sklearn.neural_network import MLPRegressor
from imblearn.over_sampling import SMOTE
from keras.optimizers import Adam, SGD
from pathlib import Path
import pandas as pd
from sklearn import metrics
import numpy as np
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DataConversionWarning)


def plot_loss_and_metrics(model_name, y_true, y_pred):
    print(f"====Model {model_name} ====")
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    plt.plot(y_true, label='True Values')
    plt.plot(y_pred, label='Predicted Values')
    plt.title('Loss Function')
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.legend()
    plt.show()




    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f'R^2 Score: {r2:.4f}')


# from pandas_profiling import ProfileReport
# "C:\Users\mahan\Desktop\Winter2023\Predictive-Machine- 4948\DataSets\car_purchasing.csv"
################################# DATA LOADING & CLEANING #################################
PATH = Path("/Users/mahan/Desktop/Winter2023/Predictive-Machine- 4948/DataSets/car_purchasing.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
df = pd.read_csv(PATH, encoding="ISO-8859-1", sep=',')
df = df.drop_duplicates()
df.drop(columns=['customer name', 'customer e-mail', 'country'], inplace=True)
df = pd.get_dummies(df, columns=['gender'])
print("Null", df.isnull().sum())
print(df.info())
print(df.head())
sns.pairplot(df)
plt.show()
sns.heatmap(df.corr(), annot=True)
plt.show()
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
cols = ['credit card debt', 'age', 'annual Salary', 'net worth']
for i in range(4):
    sns.scatterplot(x='car purchase amount', y=cols[i], data=df, ax=axes[i % 2, i // 2])
plt.show()

num_cols = ['age', 'annual Salary', 'credit card debt', 'net worth']
plt.figure(figsize=(15, 15))
for i in enumerate(num_cols):
    plt.subplot(2, 2, i[0] + 1)
    sns.scatterplot(x=df[i[1]], y=df['car purchase amount'])
plt.figure(figsize=(15, 15))
plt.show()
for i in enumerate(num_cols):
    plt.subplot(2, 2, i[0] + 1)
    sns.boxplot(df[i[1]])
plt.show()


def remove_outlier(df, col_name):
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    df_out = df.loc[(df[col_name] > Q1 - 1.5 * IQR) & (df[col_name] < Q3 + 1.5 * IQR)]
    return df_out


df = remove_outlier(df, 'age')
df = remove_outlier(df, 'credit card debt')
df = remove_outlier(df, 'net worth')
df = remove_outlier(df, 'annual Salary')

results = {}

# Split data into train and test sets
X = df.drop('car purchase amount', axis=1)
columns = X.columns
print(columns)
y = df['car purchase amount']
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
# X = sm.add_constant(X) # double check this is needed
y = scaler.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

################################## 3 FEATURE SELECTION TECHNIQUES #################################


#### Random Forest Feature Importance ####

# Use the random grid to search for best hyperparameters
rf = RandomForestRegressor()
random_grid = \
    {'bootstrap': [True],
     'max_depth': [4, 6, None],
     'max_features': ['auto'],
     'min_samples_leaf': [15],
     'min_samples_split': [15],
     'n_estimators': [400, 800, 1600]}

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, n_jobs=-1)
# Fit the random search model
rf_random.fit(X_train, y_train)
print("Best parameters to use for random forest")
print(rf_random.best_params_)
# Best parameters to use for random forest{'n_estimators': 1600, 'min_samples_split': 15, 'min_samples_leaf': 15, 'max_features': 'auto', 'max_depth': 6, 'bootstrap': True}
print("----------------------------------------")

# We create the new rf with the best random forest params suggested above
rf = RandomForestRegressor(n_estimators=1600, min_samples_split=15, min_samples_leaf=15, max_features='auto',
                           max_depth=6, bootstrap=True)
rf.fit(X_train, y_train)

# Find feature importance's
importances = list(rf.feature_importances_)


# Present features and importance scores.
def showFeatureImportances(importances, feature_list):
    dfImportance = pd.DataFrame()
    for i in range(0, len(importances)):
        dfImportance = dfImportance.append({"importance": importances[i],
                                            "feature": feature_list[i]},
                                           ignore_index=True)

    dfImportance = dfImportance.sort_values(by=['importance'],
                                            ascending=False)
    print(dfImportance)


showFeatureImportances(importances, columns)
"""   importance           feature
0    0.405492               age
1    0.393707     annual Salary
3    0.200542         net worth
2    0.000147  credit card debt
5    0.000062          gender_1
4    0.000049          gender_0"""

#### RFE  ####
from sklearn.linear_model import LinearRegression

# Create a logistic regression estimator
estimator = LinearRegression()

# Create the RFE object and specify the number of features to select
rfe = RFE(estimator, n_features_to_select=3, step=1)

# Fit the RFE object to the data
rfe.fit(X, y)

# Print the selected features
print("RFE Selected Features:")
for i, col in enumerate(columns):
    if rfe.support_[i]:
        print(col)

#### FFS  ####


# Initialize a logistic regression model. performs forward feature selection on the lung cancer
logreg = LinearRegression()

# Perform forward feature selection with 5-fold cross-validation.
sfs = SequentialFeatureSelector(logreg, direction="forward", n_features_to_select=3, cv=5)
sfs.fit(X_train, y_train)

# Print the selected feature indices and names.
print("FFS Selected Features:", list(df.drop(columns=['car purchase amount']).columns[sfs.get_support()]))

#
# ################################## MODELS #################################
#

# Split data into train and test sets
X = df[['age', 'annual Salary', 'net worth']]
y = df['car purchase amount']
X = sm.add_constant(X)  # double check this is needed

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# X=scaler.fit_transform(X)
# y=scaler.fit_transform(y.values.reshape(-1,1))

# Make predictions and evaluate with the RMSE.
model = sm.OLS(y_train, X_train_scaled).fit()
predictions = model.predict(X_test_scaled)
plot_loss_and_metrics("MinMaxScaled OLS Model (['age', 'annual Salary', 'net worth'])",y_test, predictions)
print("ABOVE")
results['Model 1 OLS'] = {
    'R-squared': r2_score(y_test, predictions),
    'MSE': mean_squared_error(y_test, predictions),
    'MAE': mean_absolute_error(y_test, predictions)
}
print(model.summary())
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Split data into train and test sets
X = df[['age', 'annual Salary', 'net worth']]
y = df['car purchase amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_dim=3))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

print(model.summary())
model.compile(optimizer='adam', loss='mean_squared_error')  # ,metrics=['mean_absolute_error'])
history = model.fit(X_train, y_train, batch_size=16, epochs=50, validation_split=0.2)
y_pred = model.predict(X_test)

results['Model 2 adam opt'] = {
    'R-squared': r2_score(y_test, y_pred),
    'MSE': mean_squared_error(y_test, y_pred),
    'MAE': mean_absolute_error(y_test, y_pred)
}

from sklearn.metrics import r2_score

ss = r2_score(y_test, y_pred)
print("r2", ss)
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss During Training or Validation')
plt.ylabel('Training & Validation Losses')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

loss = model.evaluate(X_test, y_test)
print("Multi-Layer Perceptron (MLP) loss: ", loss)

# Split the data.
X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                    y, test_size=0.3, random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_temp,
                                                y_temp, test_size=0.5, random_state=0)

# set best parameters
num_layers = 2
num_neurons = 16
activation_func = 'relu'
learning_rate = 0.01
kernel_initializer = 'uniform'

# create model with best parameters
model = Sequential()
model.add(
    Dense(num_neurons, input_dim=X_train.shape[1], activation=activation_func, kernel_initializer=kernel_initializer))
for i in range(num_layers - 1):
    model.add(Dense(num_neurons, activation=activation_func, kernel_initializer=kernel_initializer))
model.add(Dense(1, activation='linear'))
opt = Adam(learning_rate=learning_rate)
model.compile(loss='mse', optimizer=opt, metrics=['mae'])
from sklearn.metrics import mean_squared_error

# Fit the model.
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# simple early stopping
# patience:  # of epochs observed where no improvement before exiting.
# mode:      Could be max, min, or auto.
# min_delta: Amount of change needed to be considered an improvement.
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0.000001, patience=200)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1,
                     save_best_only=True)

# fit model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4000, verbose=0,
                    callbacks=[es, mc])

# load the saved model
model = load_model('best_model.h5')

# train model on full training set
# history = model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_data=(X_val, y_val))
predictions = model.predict(X_test)

results['Model 3 gridchosen param'] = {
    'R-squared': r2_score(y_test, predictions),
    'MSE': mean_squared_error(y_test, predictions),
    'MAE': mean_absolute_error(y_test, predictions)
}

print("RMSE: " + str(np.sqrt(mean_squared_error(y_test, predictions))))  # evaluate model on validation set
val_loss = model.evaluate(X_val, y_val)[0]
print('Validation loss:', val_loss)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss During Training or Validation (best param)')
plt.ylabel('Training & Validation Losses')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

# Print results
for model, metrics in results.items():
    print(model)
    print(metrics)
    print(f"R-squared: {metrics['R-squared']:.2f}")
    print(f"MSE: {metrics['MSE']:.2f}")
    print(f"MAE: {metrics['MAE']:.2f}")
    print()


def getUnfitModels():
    models = list()
    models.append(ElasticNet())
    models.append(SVR(gamma='scale'))
    models.append(DecisionTreeRegressor())
    models.append(AdaBoostRegressor())
    models.append(RandomForestRegressor(n_estimators=200))
    models.append(ExtraTreesRegressor(n_estimators=200))
    return models


def evaluateModel(y_test, predictions, model):
    mse = mean_squared_error(y_test, predictions)
    rmse = round(np.sqrt(mse), 3)
    print(" RMSE:" + str(rmse) + " " + model.__class__.__name__)


def fitBaseModels(X_train, y_train, X_test, models):
    dfPredictions = pd.DataFrame()

    # Fit base model and store its predictions in dataframe.
    for i in range(0, len(models)):
        models[i].fit(X_train, y_train)
        predictions = models[i].predict(X_test)
        colName = str(i)
        # Add base model predictions to column of data frame.
        dfPredictions[colName] = predictions
    return dfPredictions, models


def fitStackedModel(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


# Split data into train, test and validation sets.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.70)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50)

# Get base models.
unfitModels = getUnfitModels()

# Fit base and stacked models.
dfPredictions, models = fitBaseModels(X_train, y_train, X_val, unfitModels)
stackedModel = fitStackedModel(dfPredictions, y_val)

# Evaluate base models with validation data.
print("\n** Evaluate Base Models **")
dfValidationPredictions = pd.DataFrame()
for i in range(0, len(models)):
    predictions = models[i].predict(X_test)
    colName = str(i)
    dfValidationPredictions[colName] = predictions
    evaluateModel(y_test, predictions, models[i])

# Evaluate stacked model with validation data.
stackedPredictions = stackedModel.predict(dfValidationPredictions)
print("\n** Evaluate Stacked Model **")
evaluateModel(y_test, stackedPredictions, stackedModel)

# Create and fit model.
model = MLPRegressor()
model.fit(X_train, y_train)
print(model.get_params())  # Show model parameters.

# Evaluate model.
predicted_y = model.predict(X_test)
mse = mean_squared_error(y_test, predicted_y)
rmse = round(np.sqrt(mse), 3)
print(" RMSE:" + str(rmse))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss During Training or Validation (best param)')
plt.ylabel('Training & Validation Losses')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()


def showLosses(model):
    plt.plot(model.loss_curve_)
    plt.title("Loss Curve")
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()


showLosses(model)

"""GRID SEARCH FOR BEST PARAM"""

"""Initial result for me: Best Param: {'num_layers': 2, 'num_neurons': 16, 'activation_func': 'relu', 'learning_rate': 0.01, 'kernel_initializer': 'uniform', 'train_loss': 45131624.0, 'val_loss': 33958648.0}
"""

#
# def grid_search_nn(X_train, y_train, X_val, y_val, hidden_layers, neurons, activations, learning_rates, kernel_initializers):
#     best_loss = float('inf')
#     best_params = None
#     results = []
#     for layer_size in hidden_layers:
#         for num_neurons in neurons:
#             for activation_func in activations:
#                 for learning_rate in learning_rates:
#                     for kernel_initializer in kernel_initializers:
#                         model = Sequential()
#                         model.add(Dense(num_neurons, input_dim=X_train.shape[1], activation=activation_func, kernel_initializer=kernel_initializer))
#                         for i in range(layer_size - 1):
#                             model.add(Dense(num_neurons, activation=activation_func, kernel_initializer=kernel_initializer))
#                         model.add(Dense(1, activation='linear'))
#                         opt = Adam(learning_rate=learning_rate)
#                         model.compile(loss='mse', optimizer=opt, metrics=['mae'])
#                         history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), verbose=0)
#                         train_loss = history.history['loss'][-1]
#                         val_loss = history.history['val_loss'][-1]
#                         results.append({
#                             'num_layers': layer_size,
#                             'num_neurons': num_neurons,
#                             'activation_func': activation_func,
#                             'learning_rate': learning_rate,
#                             'kernel_initializer': kernel_initializer,
#                             'train_loss': train_loss,
#                             'val_loss': val_loss
#                         })
#                         if val_loss < best_loss:
#                             best_loss = val_loss
#                             best_params = {
#                                 'num_layers': layer_size,
#                                 'num_neurons': num_neurons,
#                                 'activation_func': activation_func,
#                                 'learning_rate': learning_rate,
#                                 'kernel_initializer': kernel_initializer,
#                                 'train_loss': train_loss,
#                                 'val_loss': val_loss
#                             }
#     return results, best_params
#
#
# import numpy as np
# from sklearn.model_selection import train_test_split
#
# # Load your dataset and split it into training, validation, and test sets
# X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
#
# # Define the hyperparameters to search over
# hidden_layers = [1, 2, 3]
# neurons = [8, 16, 32]
# activations = ['relu', 'sigmoid']
# learning_rates = [0.001, 0.01, 0.1]
# kernel_initializers = ['uniform', 'normal', 'glorot_uniform']
# import datetime
#
# # Record the start time
# start_time = datetime.datetime.now()
# # Call the grid_search_nn function
# results, bestParam = grid_search_nn(X_train, y_train, X_val, y_val, hidden_layers, neurons, activations, learning_rates, kernel_initializers)
#
# # Print the results
# for r in results:
#     print(r)
# print("Best Param:", bestParam)
# # Record the end time
# end_time = datetime.datetime.now()
#
# # Calculate the elapsed time
# elapsed_time = end_time - start_time
#
# # Print the elapsed time
# print("Elapsed time: ", elapsed_time)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)\
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


#
# import itertools
# import numpy as np
# from tensorflow import keras
#
#
# def grid_search(X_train, y_train, X_val, y_val, X_test, y_test,
#                 layer_sizes=[(10,), (20,), (30,)],
#                 learning_rates=[0.001, 0.01, 0.1],
#                 num_layers=[1, 2, 3],
#                 activations=['relu', 'sigmoid'],
#                 kernel_initializers=['glorot_uniform', 'he_uniform']):
#     # Create a list of all possible parameter combinations
#     param_combinations = list(itertools.product(layer_sizes, learning_rates,
#                                                 num_layers, activations, kernel_initializers))
#
#     # Initialize variables to keep track of results
#     network_stats = []
#     best_rmse = np.inf
#     best_model = None
#
#     # Loop over all parameter combinations and train models
#     for params in param_combinations:
#
#         # Unpack the parameter combination
#         layer_size, learning_rate, num_layers, activation, kernel_initializer = params
#
#         # Create the model with the current parameter combination
#         model = keras.Sequential()
#         model.add(keras.layers.Dense(layer_size[0], input_dim=X_train.shape[1],
#                                      kernel_initializer=kernel_initializer,
#                                      activation=activation))
#         for i in range(num_layers - 1):
#             model.add(keras.layers.Dense(layer_size[0], kernel_initializer=kernel_initializer,
#                                          activation=activation))
#         model.add(keras.layers.Dense(1, kernel_initializer=kernel_initializer))
#
#         # Compile the model with mean squared error loss and the specified learning rate
#         optimizer = keras.optimizers.Adam(lr=learning_rate)
#         model.compile(loss='mean_squared_error', optimizer=optimizer)
#
#         # Train the model
#         history = model.fit(X_train, y_train, epochs=100, batch_size=10,
#                             verbose=0, validation_data=(X_val, y_val))
#
#         # Evaluate the model on the test set and save the RMSE
#         rmse = np.sqrt(model.evaluate(X_test, y_test, verbose=0))
#
#         # Keep track of the best model and its parameters
#         if rmse < best_rmse:
#             best_rmse = rmse
#             best_model = model
#
#         # Save the RMSE and parameters for this model
#         network_stats.append({"rmse": rmse, "layer_size": layer_size,
#                               "learning_rate": learning_rate, "num_layers": num_layers,
#                               "activation": activation, "kernel_initializer": kernel_initializer})
#
#     # Sort the results by RMSE in descending order
#     network_stats = sorted(network_stats, key=lambda x: x['rmse'])
#
#     # Print the results
#     for result in network_stats:
#         print(result)
#
#     # Return the best model and its RMSE
#     print("Best model RMSE:", best_rmse)
#     return best_model


# from sklearn.metrics import classification_report
# def showClassificationReport(y_test, yhats):
#     # Convert continous predictions to
#     # 0 or 1.
#     for i in range(0, len(yhats)):
#         if(yhats[i]>0.5):
#             predictions.append(1)
#         else:
#             predictions.append(0)
#     print(classification_report(y_test, predictions))
# showClassificationReport(y_test, yhats)

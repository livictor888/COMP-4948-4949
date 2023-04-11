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





warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def plot_loss_and_metrics(model_name, y_true, y_pred):
    print(f"====Model {model_name} ====")
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f'R^2 Score: {r2:.4f}')
    print(f"==== # ====")

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

# Create a dictionary to store the results of each model
results = {}

# # Linear Regression model
# lr_model = LinearRegression()
# lr_model.fit(X_train_scaled, y_train)
# lr_predictions = lr_model.predict(X_test_scaled)
# results['Linear Regression'] = {
#     'R-squared': r2_score(y_test, lr_predictions),
#     'RMSE': np.sqrt(mean_squared_error(y_test, lr_predictions)),
#     'MSE': mean_squared_error(y_test, lr_predictions),
#     'MAE': mean_absolute_error(y_test, lr_predictions)
# }

# Make predictions and evaluate with the RMSE.
model = sm.OLS(y_train, X_train_scaled).fit()

# OLS_model = model
predictions = model.predict(X_test_scaled)
# plot_loss_and_metrics("MinMaxScaled OLS Model (['age', 'annual Salary', 'net worth'])",y_test, predictions)
results['Model 1 OLS'] = {
    'R-squared': r2_score(y_test, predictions),
    'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
    'MSE': mean_squared_error(y_test, predictions),
    'MAE': mean_absolute_error(y_test, predictions)
}
print(model.summary())
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Neural Network model
nn_model = Sequential()
nn_model.add(Dense(10, activation='relu', input_dim=5))
nn_model.add(Dense(10, activation='relu'))
nn_model.add(Dense(10, activation='relu'))
nn_model.add(Dense(1, activation='linear'))

nn_model.compile(optimizer='adam', loss='mean_squared_error')
history = nn_model.fit(X_train_scaled, y_train, batch_size=16, epochs=50, validation_split=0.2)
nn_predictions = nn_model.predict(X_test_scaled)

results['Neural Network'] = {
    'R-squared': r2_score(y_test, nn_predictions),
    'RMSE': np.sqrt(mean_squared_error(y_test, nn_predictions)),
    'MSE': mean_squared_error(y_test, nn_predictions),
    'MAE': mean_absolute_error(y_test, nn_predictions)
}


# Print the results of each model
for model_name, metrics in results.items():
    print(model_name)
    print(metrics)
    print("\n")


"""
Generate a summary and plot variable graphs
"""
from pandas_profiling import ProfileReport

# Generate the profiling report
profile = ProfileReport(df, title="Tesla Dataset Profiling Report", explorative=True)

# Save the report as an HTML file
profile.to_file("output.html")

# Plot histogram of 'Adj Close'
plt.figure(figsize=(10, 5))
sns.histplot(df['Adj Close'], bins=50, kde=True)
plt.title('Histogram of Adj Close')
plt.xlabel('Adj Close')
plt.ylabel('Frequency')
plt.show()

# Plot box plot of 'Adj Close'
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['Adj Close'])
plt.title('Box Plot of Adj Close')
plt.xlabel('Adj Close')
plt.show()

# Plot trends for Open, High, Low, Close, and Adj Close
plt.figure(figsize=(15, 8))
plt.plot(df.index, df['Open'], label='Open')
plt.plot(df.index, df['High'], label='High')
plt.plot(df.index, df['Low'], label='Low')
plt.plot(df.index, df['Close'], label='Close')
plt.plot(df.index, df['Adj Close'], label='Adj Close')

# Customize plot
plt.title('Trends in Opening, Closing, High, Low, and Adjusted Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()

# Plot histogram of 'Volume'
plt.figure(figsize=(10, 5))
sns.histplot(df['Volume'], bins=50, kde=True)
plt.title('Histogram of Trading Volume')
plt.xlabel('Volume')
plt.ylabel('Frequency')
plt.show()

# Plot box plot of 'Volume'
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['Volume'])
plt.title('Box Plot of Trading Volume')
plt.xlabel('Volume')
plt.show()

# Create a scatter plot of trading volume vs adjusted closing price
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df['Volume'], y=df['Adj Close'])
plt.title('Scatter Plot: Trading Volume vs Adjusted Closing Price')
plt.xlabel('Trading Volume')
plt.ylabel('Adjusted Closing Price')
plt.show()



#
#
# """
# Create NN Model
# """
# from keras.optimizers import Adam, RMSprop
#
#
# def create_nn_model(optimizer='adam', neurons=10, lr=0.001, activation='relu', initializer='he_normal'):
#     model = Sequential()
#     model.add(Dense(neurons, activation=activation, kernel_initializer=initializer, input_dim=5))
#     model.add(Dense(neurons, activation=activation, kernel_initializer=initializer))
#     model.add(Dense(1, activation='linear'))
#
#     if optimizer == 'adam':
#         opt = Adam(learning_rate=lr)
#     elif optimizer == 'rmsprop':
#         opt = RMSprop(learning_rate=lr)
#
#     model.compile(optimizer=opt, loss='mean_squared_error')
#     return model
#
#
# # Define the grid search parameters
# param_grid = {
#     'optimizer': ['adam', 'rmsprop'],
#     'neurons': [10, 20, 30],
#     'lr': [0.001, 0.01, 0.1],
#     'activation': ['relu', 'tanh'],
#     'initializer': ['he_normal', 'he_uniform']
# }
#
#
#
#
# """
# Perform Grid Search
# """
# from skopt import BayesSearchCV
# from skopt.space import Real, Categorical, Integer
# from keras.optimizers import Adam
# from scikeras.wrappers import KerasRegressor
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import GridSearchCV
#
#
# # Create a KerasRegressor model
# nn_model = KerasRegressor(build_fn=create_nn_model, epochs=50, batch_size=16, verbose=0, optimizer='adam', neurons=10, lr=0.001, activation='relu', initializer='he_normal')
#
#
# # Define the search space
# search_space = {
#     'optimizer': Categorical(['adam', 'rmsprop']),
#     'neurons': Integer(10, 100),
#     'activation': Categorical(['relu', 'tanh']),
#     'lr': Real(1e-4, 1e-2, prior='log-uniform'),
# }
#
#
# # Perform the Grid search
# grid_search = GridSearchCV(estimator=nn_model, param_grid=param_grid, n_jobs=-1, cv=3, scoring='neg_mean_squared_error')
# grid_search_result = grid_search.fit(X_train_scaled, y_train)
#
#
# # Summarize the results
# print(f"Best: {grid_search_result.best_score_} using {grid_search_result.best_params_}")
# means = grid_search_result.cv_results_['mean_test_score']
# stds = grid_search_result.cv_results_['std_test_score']
# params = grid_search_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print(f"{mean:.4f} ({stdev:.4f}) with: {param}")
#
# """
# Optimized parameters
# Best: -1.6165573675727283 using
# {'activation': 'relu', 'initializer': 'he_uniform', 'lr': 0.1, 'neurons': 30, 'optimizer': 'rmsprop'}
#
# Best: -2.35016944334606 using
# {'activation': 'relu', 'initializer': 'he_normal', 'lr': 0.01, 'neurons': 20, 'optimizer': 'adam'}
#
# Best: -2.0810359138867462 using
# {'activation': 'relu', 'initializer': 'he_normal', 'lr': 0.1, 'neurons': 10, 'optimizer': 'rmsprop'}
#
# """
#
# """
# Create an Optimized Neural Network Model
# """
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam, RMSprop
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import matplotlib.pyplot as plt
#
# # Define the optimized parameters obtained from the grid search
# optimizer = 'rmsprop'
# neurons = 10
# lr = 0.1
# activation = 'relu'
# initializer = 'he_normal'
#
# # Create the optimized NN model
# def create_nn_model(optimizer, neurons, lr, activation, initializer):
#     model = Sequential()
#     model.add(Dense(neurons, activation=activation, kernel_initializer=initializer, input_dim=5))
#     model.add(Dense(neurons, activation=activation, kernel_initializer=initializer))
#     model.add(Dense(1, activation='linear'))
#
#     if optimizer == 'adam':
#         opt = Adam(learning_rate=lr)
#     elif optimizer == 'rmsprop':
#         opt = RMSprop(learning_rate=lr)
#
#     model.compile(optimizer=opt, loss='mean_squared_error')
#     return model
#
# optimized_model = create_nn_model(optimizer, neurons, lr, activation, initializer)
#
# # Train the optimized model
# history = optimized_model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.2)
#
# # Predict on the test set
# y_pred = optimized_model.predict(X_test_scaled)
#
# # Calculate the mean squared error
# mse = mean_squared_error(y_test, y_pred)
#
# # Calculate the mean absolute error
# mae = mean_absolute_error(y_test, y_pred)
#
# # Calculate the R-squared score
# r2score = r2_score(y_test, y_pred)
#
# # Calculate the root mean squared error
# rmse = mean_squared_error(y_test, y_pred, squared=False)
#
# # Print the metrics
# print(f"R-squared: {r2score:.4f}")
# print(f"Mean Squared Error: {mse:.4f}")
# print(f"Root Mean Squared Error: {rmse:.4f}")
# print(f"Mean Absolute Error: {mae:.4f}")
#
# # Plot the model loss during training
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['train', 'validation'], loc='upper right')
# plt.show()
#
#
#

# Pandas is used for data manipulation
import pandas as pd
from   sklearn.metrics import mean_squared_error

# Read in data and display first 5 rows
features = pd.read_csv('C:\\datasets\\temperatures.csv')

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(features)

# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)

# Display the first 5 rows of the last 12 columns.
print(features.head(5))

# Use numpy to convert to arrays
import numpy as np

# Labels are the values we want to predict
labels = np.array(features['actual'])

# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('actual', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels =\
    train_test_split(features, labels, test_size = 0.25, random_state = 42)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')

# Print out the mean square error.
mse = mean_squared_error(test_labels, predictions)
print('RMSE:', np.sqrt(mse))




#############################################################
# This revised example shows a more realistic range of hyperparameters.
#############################################################
# Pandas is used for data manipulation
import pandas as pd
from   sklearn.metrics import mean_squared_error

# Read in data and display first 5 rows
features = pd.read_csv('C:\\datasets\\temperatures.csv')

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(features)

# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)

# Display the first 5 rows of the last 12 columns.
print(features.head(5))

# Use numpy to convert to arrays
import numpy as np

# Labels are the values we want to predict
labels = np.array(features['actual'])

# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('actual', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels =\
    train_test_split(features, labels, test_size = 0.25, random_state = 42)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees
# rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf = RandomForestRegressor(n_estimators = 1600, min_samples_split=10, min_samples_leaf=2, max_features=1.0, max_depth=None, bootstrap=True)

# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')

# Print out the mean square error.
mse = mean_squared_error(test_labels, predictions)
print('RMSE:', np.sqrt(mse))

# Get numerical feature importances
importances = list(rf.feature_importances_)

# Present features and importance scores.
def showFeatureImportances(importances, feature_list):
    dfImportance = pd.DataFrame()
    for i in range(0, len(importances)):
        dfImportance = pd.concat([dfImportance, pd.DataFrame({"importance":[importances[i]],
                                                               "feature":[feature_list[i]] })], ignore_index = True)
    dfImportance = dfImportance.sort_values(by=['importance'],
                                            ascending=False)
    print(dfImportance)
showFeatureImportances(importances, feature_list)

from sklearn.model_selection import RandomizedSearchCV

random_grid =\
{'bootstrap': [True],
 'max_depth': [4,6, None],
 'max_features': [1.0],
 'min_samples_leaf': [15],
 'min_samples_split': [15],
 'n_estimators': [ 400, 800, 1600]}

print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_features, train_labels)

print("Best parameters")
print(rf_random.best_params_)

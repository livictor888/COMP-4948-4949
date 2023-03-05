#!/usr/bin/env python
# coding: utf-8

# In[4]:


from IPython.display import display, HTML

display(HTML("<style>pre { white-space: pre !important; }</style>"))

# In[20]:


from pandas_datareader import data as pdr

import datetime

import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm

import pandas as pd
import numpy as np

# Show all columns.

pd.set_option('display.max_columns', None)

pd.set_option('display.width', 1000)

DATAPATH = './jena_climate_2009_2016.csv'

data = pd.read_csv(DATAPATH, sep=',')

# Make dates actual dates
data['Date Time'] = pd.to_datetime(data['Date Time'])

for col in data.iloc[:, 2:].columns:
    if data[col].dtypes == object:
        data[col] = data[col].str.replace(',', '.').astype('float')

data.head()


# In[18]:


# Compute the average considering only the positive values
def positive_average(num):
    return num[num > -200].mean()


# # Aggregate data
# daily_data = data.drop('Time', axis=1).groupby('Date').apply(positive_average)

# # Drop columns with more than 8 NaN
# daily_data = daily_data.iloc[:,(daily_data.isna().sum() <= 8).values]

# # Remove rows containing NaN values
# daily_data = daily_data.dropna()

# # Aggregate data by week
# weekly_data = daily_data.resample('W').mean()

# Plot the weekly concentration of each gas
def plot_data():
    weekly_data = data.resample('W', on='Date Time').mean()
    plt.figure(figsize=(17, 8))
    plt.plot(weekly_data['T (degC)'])
    plt.xlabel('Time')
    plt.ylabel(col)
    plt.grid(False)
    plt.show()


plot_data()


def plot_data():
    weekly_data = data.resample('M', on='Date Time').mean()
    plt.figure(figsize=(17, 8))
    plt.plot(weekly_data['T (degC)'])
    plt.xlabel('Time')
    plt.ylabel(col)
    plt.grid(False)
    plt.show()


plot_data()

# In[147]:


from pandas_datareader import data as pdr
import yfinance as yfin  # Work around until
# pandas_datareader is fixed.
import datetime
import matplotlib.pyplot as plt
import pandas as pd

# Do not show warning.
pd.options.mode.chained_assignment = None  # default='warn'
dfPredictions = pd.DataFrame()
##################################################################
# CONFIGURATION SECTION
NUM_DAYS = 1200
NUM_TIME_STEPS = 2
TEST_DAYS = 10
##################################################################

daily_data = data.resample('D', on='Date Time').mean()


# Creates time shifted columns for as many time steps needed.
def backShiftColumns(df, originalColName, numTimeSteps):
    dfNew = df[[originalColName]]

    for i in range(1, numTimeSteps + 1):
        newColName = originalColName + 't-' + str(i)
        dfNew[newColName] = dfNew[originalColName].shift(periods=i)
    return dfNew


def prepareStockDf(columns):
    df = daily_data
    print(df.head())
    # Create data frame with back shift columns for all features of interest.
    mergedDf = pd.DataFrame()
    for i in range(0, len(columns)):
        backShiftedDf = backShiftColumns(df, columns[i], NUM_TIME_STEPS)
        if (i == 0):
            mergedDf = backShiftedDf
        else:
            mergedDf = mergedDf.merge(backShiftedDf, left_index=True,
                                      right_index=True)

    newColumns = list(mergedDf.keys())

    # Append stock symbol to column names.
    for i in range(0, len(newColumns)):
        mergedDf.rename(columns={newColumns[i]: "BackShift" + "_" + newColumns[i]}, inplace=True)

    return mergedDf


columns = list(daily_data.columns.values)[0::]
# columns = ['T (degC)']
mergedDf1 = prepareStockDf(columns)
# aaplDf   = data['T (degC)']
mergedDf = daily_data.merge(mergedDf1, left_index=True, right_index=True)
mergedDf = mergedDf.dropna()
print(mergedDf)

# import seaborn as sns

# corr = mergedDf.corr()
# plt.figure(figsize = (4,4))
# ax = sns.heatmap(corr[['MSFT_Open']],
#             linewidth=0.5, vmin=-1,
#             vmax=1, cmap="YlGnBu")
# plt.show()

# xfeatures = ['MSFT_Ct-2', 'GOOGL_Ct-1']

xfeatures = list(mergedDf.columns.values)[::]
print(xfeatures)
xfeatures.remove('T (degC)')
xfeatures.remove('BackShift_T (degC)')
xfeatures.remove('p (mbar)')
xfeatures.remove('BackShift_p (mbar)')
xfeatures.remove('Tpot (K)')
xfeatures.remove('BackShift_Tpot (K)')
xfeatures.remove('Tdew (degC)')
xfeatures.remove('BackShift_Tdew (degC)')
xfeatures.remove('rh (%)')
xfeatures.remove('BackShift_rh (%)')
xfeatures.remove('VPmax (mbar)')
xfeatures.remove('BackShift_VPmax (mbar)')
xfeatures.remove('VPdef (mbar)')
xfeatures.remove('BackShift_VPdef (mbar)')
xfeatures.remove('sh (g/kg)')
xfeatures.remove('BackShift_sh (g/kg)')
xfeatures.remove('H2OC (mmol/mol)')
xfeatures.remove('BackShift_H2OC (mmol/mol)')
xfeatures.remove('rho (g/m**3)')
xfeatures.remove('BackShift_rho (g/m**3)')
xfeatures.remove('wv (m/s)')
xfeatures.remove('BackShift_wv (m/s)')
xfeatures.remove('max. wv (m/s)')
xfeatures.remove('BackShift_max. wv (m/s)')
xfeatures.remove('wd (deg)')
xfeatures.remove('BackShift_wd (deg)')
xfeatures.remove('VPact (mbar)')
# xfeatures.remove('max. wv (m/s)')
xfeatures.remove('BackShift_p (mbar)t-2')
# xfeatures.remove('BackShift_T (degC)t-2')
xfeatures.remove('BackShift_Tdew (degC)t-2')
xfeatures.remove('BackShift_rh (%)t-1')
xfeatures.remove('BackShift_rh (%)t-2')
xfeatures.remove('BackShift_VPmax (mbar)t-1')
xfeatures.remove('BackShift_VPmax (mbar)t-2')
xfeatures.remove('BackShift_VPact (mbar)t-2')
xfeatures.remove('BackShift_VPdef (mbar)t-1')
xfeatures.remove('BackShift_VPdef (mbar)t-2')
xfeatures.remove('BackShift_sh (g/kg)t-1')
xfeatures.remove('BackShift_sh (g/kg)t-2')
xfeatures.remove('BackShift_H2OC (mmol/mol)t-1')
xfeatures.remove('BackShift_H2OC (mmol/mol)t-2')
xfeatures.remove('BackShift_rho (g/m**3)t-2')
xfeatures.remove('BackShift_wv (m/s)t-1')
xfeatures.remove('BackShift_max. wv (m/s)t-1')
xfeatures.remove('BackShift_max. wv (m/s)t-2')
xfeatures.remove('BackShift_wd (deg)t-1')
xfeatures.remove('BackShift_T (degC)t-2')
xfeatures.remove('BackShift_Tpot (K)t-2')
xfeatures.remove('BackShift_wv (m/s)t-2')
print(xfeatures)
X = mergedDf[xfeatures]
y = mergedDf[['T (degC)']]

# Add intercept for OLS regression.
import statsmodels.api as sm

X = sm.add_constant(X)

# Split into test and train sets. The test data must be
# the latest data range.
lenData = len(X)
X_train = X[0:lenData - TEST_DAYS]
y_train = y[0:lenData - TEST_DAYS]
X_test = X[lenData - TEST_DAYS:]
y_test = y[lenData - TEST_DAYS:]

# Model and make predictions.
modelOLS = sm.OLS(y_train, X_train).fit()
print(modelOLS.summary())
predictions = modelOLS.predict(X_test)
dfPredictions['OLS'] = predictions
print('********')
print(dfPredictions)
# Show RMSE and plot the data.
from sklearn import metrics
import numpy as np

print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))

plt.figure(figsize=(17, 8))
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.show()

# In[148]:


import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yfin  # Work around until
# pandas_datareader is fixed.
import datetime
import pmdarima as pm

TOTAL_DAYS = 10

arimaDf = daily_data.copy()

# Build feature set with backshifted closing prices.
arimaDf['Tdew (degC)_t-1'] = arimaDf['Tdew (degC)'].shift(1)
arimaDf = arimaDf.dropna()
dfX = arimaDf[['T (degC)', 'Tdew (degC)_t-1']]
size = len(dfX) - TOTAL_DAYS
train, test = dfX[0:size], dfX[size:]

# Create training set and copy of the training set.
train.tail(TOTAL_DAYS)
history = train.copy()
predictions = []

# Iterate to make predictions for the evaluation set.
for i in range(0, len(test)):
    lenOpen = len(history[['Tdew (degC)_t-1']])
    print("\n\nModel " + str(i))
    #     print(history.shape)

    model = pm.auto_arima(history[['T (degC)']],
                          exogenous=history[['Tdew (degC)_t-1']],
                          start_p=1, start_q=1,
                          test='adf',  # Use adftest to find optimal 'd'
                          max_p=3, max_q=3,  # Set maximum p and q.
                          d=None,  # Let model determine 'd'.
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True)
    fc, confint = model.predict(n_periods=1,
                                exogenous=np.array(
                                    history.iloc[lenOpen - 1]['Tdew (degC)_t-1']).reshape(1, -1),
                                return_conf_int=True)
    predictions.append(fc)
    open = test.iloc[i]['T (degC)']
    close_t_1 = test.iloc[i]['Tdew (degC)_t-1']
    history = history.append({"T (degC)": open, "Tdew (degC)_t-1": close_t_1},
                             ignore_index=True)

# dfPredictions['ARIMA'] = predictions
plt.plot(test.index, test['T (degC)'], marker='o',
         label='Actual', color='blue')
plt.plot(test.index, predictions, marker='o',
         label='Predicted', color='orange')
plt.legend()
plt.xticks(rotation=70)
plt.show()

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(test['T (degC)'], predictions))
print('Test RMSE: %.3f' % rmse)

# In[151]:


# print(model)
# arimaModel = model
# print(arimaModel)
tempList = []
for each in predictions:
    tempList.append(each.values)

dfPredictions['ARIMA'] = np.array(tempList)
print(dfPredictions)

# In[153]:


# Pandas is used for data manipulation
import pandas as pd
from sklearn.metrics import mean_squared_error

# Read in data and display first 5 rows
features = daily_data.copy()
features.dropna()
features.reset_index(drop=True, inplace=True)

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(features.describe())

# Use numpy to convert to arrays
import numpy as np

# Labels are the values we want to predict
labels = np.array(features['T (degC)'])
print(np.isnan(features))

# Remove the labels from the features
# axis 1 refers to the columns
features = features.drop('T (degC)', axis=1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                            random_state=42)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators=1000, random_state=42)

print('test')
# print(np.isnan(train_features).any())
# print(np.isnan(train_labels).any())


# print(test_labels)
train_features = X_train
train_labels = y_train
test_features = X_test
test_labels = y_test['T (degC)']

# train_features = np.nan_to_num(train_features)
# train_labels = np.nan_to_num(train_labels)
# test_features = np.nan_to_num(test_features)
# test_labels = np.nan_to_num(test_labels)

print('test')
# print(test_labels)
# print(np.isnan(train_features).any())
# print(np.isnan(train_labels).any())

# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
print('test')
print(predictions)
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

print('dfPred')
print(dfPredictions)
dfPredictions['RF'] = predictions

# In[155]:


print(dfPredictions)
print(test_labels)

# In[164]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

stackedModel = LinearRegression()
stackedModel.fit(dfPredictions, test_labels)

stackedPredictions = stackedModel.predict(dfPredictions)


def evaluateModel(y_test, predictions, model):
    mse = mean_squared_error(y_test, predictions)
    rmse = round(np.sqrt(mse), 3)
    print(" RMSE:" + str(rmse) + " " + model.__class__.__name__)


print("\n** Evaluate Stacked Model **")
evaluateModel(test_labels, stackedPredictions, stackedModel)

# In[ ]:





"""MAYBE HAVE TO BIN THE TEMPERATURE"""
from pathlib import Path

from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import xgboost as xgb
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)

PATH = r"C:/datasets/"
CSV_DATA = "powerconsumption.csv"
df = pd.read_csv(PATH + CSV_DATA, sep=',')

print("ORIGINAL DF--------------------")
print(df)
print("ORIGINAL DF--------------------")

"""Set datetime as the index instead of just a number 0 1 2 3 4..."""
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.set_index('Datetime')
# Remove PowerConsumption_Zone2 and PowerConsumption_Zone3 columns
df.drop(['PowerConsumption_Zone1', 'PowerConsumption_Zone3'], axis=1, inplace=True)


"""
print(df.head())
print(df.info)
df.hist(figsize=(12, 12))
plt.show()
"""


# print(df.isnull().values.any())  # False, there are no NUll values in the dataset
# print(df.duplicated().values.any())  # False, there are no duplicated rows

"""FROM KEGGLE"""
def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df


df = create_features(df)
print(df.head())

"""FROM PATS SAMPLE CODE"""
# Create back-shifted columns for an attribute.
def addBackShiftedColumns(df, colName, timeLags):
    for i in range(1, timeLags + 1):
        newColName = colName + "_t-" + str(i)
        df[newColName] = df[colName].shift(i)
    return df


# Build dataframe for modelling.
columns = ['PowerConsumption_Zone2', 'Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']
modelDf = df.copy()
NUM_TIME_STEPS = 3
for i in range(0, len(columns)):
    modelDf = addBackShiftedColumns(modelDf, columns[i],
                                    NUM_TIME_STEPS)
"""BACK SHIFTED AND PRINT"""
modelDf = modelDf.dropna()
print(modelDf.head())
y = modelDf[['PowerConsumption_Zone2']]
""" THIS IS INCOMPLETE"""
""" INCLUDE ALL THE VARIABLES AND COMPARE THE P-VALUES TO SELECT THE FEATURES"""
# X = modelDf[['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows', 'hour', 'dayofweek',
# 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']]
X = modelDf[['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']]

# Plot pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_pacf(df['PowerConsumption_Zone2'])
plt.show()


# Add intercept for OLS regression.
X = sm.add_constant(X)
TEST_DAYS = 10

# Split into test and train sets. The test data includes
# the latest values in the data.
lenData = len(X)
X_train = X[0:lenData - TEST_DAYS]
y_train = y[0:lenData - TEST_DAYS]
X_test = X[lenData - TEST_DAYS:]
y_test = y[lenData - TEST_DAYS:]

"""
MAKE OLS MODEL
"""

# Model and make predictions.
model = sm.OLS(y_train, X_train).fit()
print(model.summary())
predictions = model.predict(X_test)

# Show RMSE.
from sklearn import metrics

print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Plot the data.
xaxisValues = list(y_test.index)
plt.plot(xaxisValues, y_test, label='Actual', marker='o')
plt.plot(xaxisValues, predictions, label='Predicted', marker='o')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.title("Mean Power Usage in Zone2")
plt.show()

#
#
#
#\




# ##Correlation Matrix
#
# #Renaming axis labels
# axis_labels = ['Temp', 'Hum', 'Wind', 'Gen Diff Flows', 'Diff Flows', 'Power Z1', 'Power Z2', 'Power Z3']
# #Setting dimensions and generating heatmap with Seaborn
# fig, ax = plt.subplots(figsize=(12,10))
# sns.heatmap(df.corr(), annot=True, ax=ax, cmap = 'vlag', fmt='.1g', annot_kws={
#                 'fontsize': 14,
#                 'fontweight': 'regular',
#             }, xticklabels= axis_labels, yticklabels=axis_labels)
#
# #Setting Fontsize for labels
# plt.xticks(fontsize=11)
# plt.yticks(fontsize=11)
#
# #Generating plot
# plt.show()








#
#
# ## Feature Engineering extracts the hour, day of the week, quarter, month etc. from the datetime index
#
# def create_features(df):
#     """
#     Create time series features based on time series index.
#     """
#     df = df.copy()
#     df['hour'] = df.index.hour
#     df['dayofweek'] = df.index.dayofweek
#     df['quarter'] = df.index.quarter
#     df['month'] = df.index.month
#     df['year'] = df.index.year
#     df['dayofyear'] = df.index.dayofyear
#     df['dayofmonth'] = df.index.day
#     df['weekofyear'] = df.index.isocalendar().week
#     return df
#
# df = create_features(df)
#
# # # Perform decomposition using multiplicative decomposition.
# # tseries = seasonal_decompose(df['PowerConsumption_Zone2'], model='multiplicative', extrapolate_trend="freq")
# #
# # tseries.plot()
# # plt.show()
#
#
# #Calculating 10-day, 15-day and 30-day Simple Moving Average
#
# df['SMA10'] = df['PowerConsumption_Zone2'].rolling(10).mean()
# df['SMA15'] = df['PowerConsumption_Zone2'].rolling(15).mean()
# df['SMA30'] = df['PowerConsumption_Zone2'].rolling(30).mean()
#
# print(df.head())
#
#
# fig, ax = plt.subplots(figsize=(20, 10))
#
# # zone1 = sns.boxplot(data=df, x='hour', y='PowerConsumption_Zone1', palette = 'Oranges', showfliers=False)
# zone2 = sns.boxplot(data=df, x='hour', y='PowerConsumption_Zone2', palette = 'Reds', showfliers=False)
# # zone3 = sns.boxplot(data=df, x='hour', y='PowerConsumption_Zone3', palette = 'Blues', showfliers=False)
#
# plt.suptitle('KW by Hour', fontsize=15)
# plt.xlabel('hour', fontsize=12)
# plt.ylabel('Power Consumption in KW', fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
#
# #Generating plot
# plt.show()
#
#
# #Train and Test Split
# zone_2 = df['PowerConsumption_Zone2']
#
# train = zone_2.loc[zone_2.index < '10-01-2017']
# test = zone_2.loc[zone_2.index >= '10-01-2017']
#
# fig, ax = plt.subplots(figsize=(25, 8))
#
# train.plot(ax=ax, label='Training Set', title='Data Train/Test Split', color = "#011f4b")
# test.plot(ax=ax, label='Test Set', color="orange")
#
# ax.axvline('10-01-2017', color='black', ls='--')
# ax.legend(['Training Set', 'Test Set'])
#
# plt.title('Data Train/Test Split', fontsize=15)
# plt.xlabel('Date', fontsize=12)
# plt.ylabel('Power Consumption in KW', fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
#
# #Generating plot
# plt.show()
#
#
#
# ## Defining input and target variable
#
# #Training data goes from 1st January until 30th of September, input variables are all the columns in the dataset apart from Zone 1,2,3 consumption
# X_train = df.loc[:'10-01-2017',['Humidity', 'Temperature', 'WindSpeed','dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year','SMA10', 'SMA30', 'SMA15']]
# y_train = df.loc[:'10-01-2017', ['PowerConsumption_Zone2']]
#
# #Testing data goes from 1st January until 30th of September, input variables are all the columns in the dataset apart from Zone 1,2,3 consumption
# X_test = df.loc['10-01-2017':,['Humidity', 'Temperature', 'WindSpeed','dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year','SMA10', 'SMA30', 'SMA15']]
# y_test = df.loc['10-01-2017':, ['PowerConsumption_Zone2']]
#
#
#
#
# #Defining model and fitting
# reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
#                        n_estimators=1500,
#                        objective='reg:linear',
#                        max_depth=3,
#                        learning_rate=0.3,
#                        random_state = 48)
#
# reg.fit(X_train, y_train,
#         eval_set=[(X_train, y_train), (X_test, y_test)],
#         verbose=100)
#
#
#
#
# ##Assessing Feature Importance
# fi = pd.DataFrame(data=reg.feature_importances_,
#              index=X_train.columns,
#              columns=['importance'])
#
# fi.sort_values('importance').plot(kind='barh', title='Feature Importance', color = "#011f4b", figsize=(12,10))
# plt.title('Feature Importance Gradient Boosting Regressor', fontsize=15)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
#
# #Generating plot
# plt.show()
#
#
# ##Merging predictions with original dataset
# y_test = pd.DataFrame(y_test)
# y_test['prediction'] = reg.predict(X_test)
# df = df.merge(y_test[['prediction']], how='left', left_index=True, right_index=True)
#
# df.tail()
#
# ##Function to calculate regression metrics, evaluating accuracy
# def regression_results(y_true, y_pred):
#
#     # Regression metrics
#     explained_variance=metrics.explained_variance_score(y_true, y_pred)
#     mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred)
#     mse=metrics.mean_squared_error(y_true, y_pred)
#     mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
#     median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
#     r2=metrics.r2_score(y_true, y_pred)
#     mape = (1- metrics.mean_absolute_percentage_error(y_true, y_pred))
#
#
#     print('explained_variance: ', round(explained_variance,4))
#     print('mean_squared_log_error: ', round(mean_squared_log_error,4))
#     print('r2: ', round(r2,4))
#     print('MAE: ', round(mean_absolute_error,4))
#     print('MSE: ', round(mse,4))
#     print('RMSE: ', round(np.sqrt(mse),4))
#     print('MAPE: ', round(mape,4))
#
#
# #Apply function and print results
# regression_results(y_test['PowerConsumption_Zone2'], y_test['prediction'])
#
# # # Perform decomposition using multiplicative decomposition.
# # tseries = seasonal_decompose(df['PowerConsumption_Zone2'], model='multiplicative', extrapolate_trend="freq")
# # tseries2 = seasonal_decompose(df['PowerConsumption_Zone2'], model='additive', extrapolate_trend="freq")
# #
# # tseries.plot()
# # # plt.show()
# #
# # tseries2.plot()
# # plt.show()
# # # Extract the Components ----
# # # Actual Values = Product of (Seasonal * Trend * Resid)
# # dfComponents = pd.concat([tseries.seasonal, tseries.trend,
# #                           tseries.resid, tseries.observed], axis=1)
# # dfComponents.columns = ['seas', 'trend', 'resid', 'actual_values']
# # print(dfComponents.head())
# # print(dfComponents.iloc[1])
# #
# #
# # for column in dfComponents.iloc[1]:
# #     print(column)
# #
# #
# # for i, j in dfComponents.iterrows():
# #     print(i, j['seas'] * j['trend'] * j['resid'])
#
#
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from statsmodels.tsa.arima.model import ARIMA
# import statsmodels.api as sm
#
# PATH = Path("/Users/mahan/Desktop/Winter2023/Big-Data-4949/DataSets/powerconsumption.csv")
#
# data = pd.read_csv(PATH, sep=',', parse_dates=[0])
# data.Datetime = pd.to_datetime(data.Datetime, format='%m-%d-%Y')
# data.set_index('Datetime', inplace=True)
#
# # Remove PowerConsumption_Zone2 and PowerConsumption_Zone3 columns
# data.drop(['PowerConsumption_Zone1', 'PowerConsumption_Zone3'], axis=1, inplace=True)
#
# # Perform time series analysis for PowerConsumption_Zone1
# ts_data = data['PowerConsumption_Zone2']
# train_data = ts_data.iloc[:int(len(ts_data)*0.8)]
# test_data = ts_data.iloc[int(len(ts_data)*0.8):]
#
# # Plot time series data
# plt.figure(figsize=(10, 5))
# plt.plot(train_data.index, train_data, label='Train')
# plt.plot(test_data.index, test_data, label='Test')
# plt.legend(loc='best')
# plt.title('Power Consumption Time Series Data')
# plt.show()
#
# # Build ARIMA model
# model = ARIMA(train_data, order=(1, 1, 1))
# model_fit = model.fit()
#
# # Make predictions
# predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, typ='levels')
#
# # Plot predictions
# plt.figure(figsize=(10, 5))
# plt.plot(test_data.index, test_data, label='Actual')
# plt.plot(predictions.index, predictions, label='Predicted')
# plt.legend(loc='best')
# plt.title('Power Consumption Predictions')
# plt.show()
#
# # OLS regression model
# y = train_data
# X = sm.add_constant(data.drop('PowerConsumption_Zone2', axis=1).iloc[:int(len(ts_data)*0.8)])
#
# ols_model = sm.OLS(y, X)
# ols_results = ols_model.fit()
# print(ols_results.summary())
#
# # Identify most important features based on p-values
# p_values = ols_results.pvalues[1:]
# significant_features = p_values[p_values < 0.05]
# print('Significant features:\n', significant_features)
#
#
# # data = pd.read_csv('Electric_Production.csv')
# # data.DATE = pd.to_datetime(data.DATE, format='%m-%d-%Y')
# # data.set_index('DATE', inplace=True)
#
# # Perform seasonal decomposition
# decomposition = sm.tsa.seasonal_decompose(data['PowerConsumption_Zone2'], model='additive', period=12)
#
# # Plot seasonal decomposition
# fig, axs = plt.subplots(4, 1, figsize=(10, 10))
# axs[0].plot(data['PowerConsumption_Zone2'])
# axs[0].set_title('Original Data')
# axs[1].plot(decomposition.trend)
# axs[1].set_title('Trend')
# axs[2].plot(decomposition.seasonal)
# axs[2].set_title('Seasonality')
# axs[3].plot(decomposition.resid)
# axs[3].set_title('Residuals')
# plt.tight_layout()
# plt.show()
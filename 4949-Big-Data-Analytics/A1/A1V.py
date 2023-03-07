"""
Big Data Analysis for powerconsumption.csv
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
from pandas_profiling import ProfileReport
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.metrics import mean_squared_error

warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning))

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Load the dataset into a pandas DataFrame object
PATH = r"C:/datasets/"
CSV_DATA = "powerconsumption.csv"
df = pd.read_csv(PATH + CSV_DATA, sep=',')

print("\n---------- Original dataset ----------\n")
print(df.head(10))
print()
print(df.describe().T)
print("\n---------- Original dataset ----------\n")

# Check if there are any empty cells
print("\n---------- Check for empty cells ----------\n")
print(df.isna().any())

# Set datetime as the index instead of just a number 0 1 2 3 4...
print("\n---------- Set the index to Datetime ----------")
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)

# Our target variable is PowerConsumption_Zone2
print("---------- Drop PowerConsumption_Zone1 and PowerConsumption_Zone3 ----------\n")
df.drop(['PowerConsumption_Zone1', 'PowerConsumption_Zone3'], axis=1, inplace=True)

print("\n---------- Adjusted dataset ----------\n")
print(df.head(10))
print(df.info)
df.hist(figsize=(12, 12))
plt.show()

df.to_csv("C:/datasets/powerconsumption_adjusted.csv", index=False)

print("---------- PowerConsumption_Zone2 information ----------\n")
# Generate a report with the adjusted data
# prof = ProfileReport(df)
# prof.to_file(output_file='output.html')

# Plot for PowerConsumption_Zone2
plt.hist(df["PowerConsumption_Zone2"], bins=50)
plt.xlabel("Power Consumption (kW)")
plt.ylabel("Frequency")
plt.show()

y = df['PowerConsumption_Zone2']
X = df.drop(['PowerConsumption_Zone2'], axis=1)

# Fit the OLS regression model
X = sm.add_constant(X)  # add a constant column to X to estimate the intercept
model = sm.OLS(y, X).fit()

# Analyze the model results
print("\n---------- Basic OLS Regression Model ----------\n")
print(model.summary())

# Calculate the root mean squared error (RMSE)
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print("RMSE:", rmse)


# Plot PACF the first 3 values are outside the error range, we should back shift by 3
plot_pacf(df['PowerConsumption_Zone2'])
plt.show(alpha=0.89)


"""
Implement binning for the columns
"""
# Bins for the variables, using the ones from the histograms
temperature_bins = [0, 10, 20, 30, 40]
humidity_bins = [0, 20, 40, 60, 80, 100]
general_diffuse_flows_bins = [0, 400, 800, 1200, 1600]
diffuse_flow_bins = [0, 200, 400, 600, 800, 1000]
power_consumption_zone2_bins =[8000, 16000, 24000, 32000, 40000]


X = df.drop(['PowerConsumption_Zone2'], axis=1)
X['Temperature'] = pd.cut(X['Temperature'], bins=temperature_bins, labels=False)
X['Humidity'] = pd.cut(X['Humidity'], bins=humidity_bins, labels=False)
X['GeneralDiffuseFlows'] = pd.cut(X['Temperature'], bins=general_diffuse_flows_bins, labels=False)
X['DiffuseFlows'] = pd.cut(X['Temperature'], bins=diffuse_flow_bins, labels=False)
X['PowerConsumption_Zone2'] = pd.cut(X['Temperature'], bins=power_consumption_zone2_bins, labels=False)

# Joining the binned columns back into the dataframe object
X = pd.concat([X, pd.get_dummies(X['Temperature'], prefix='Temperature', drop_first=True)], axis=1)
X = pd.concat([X, pd.get_dummies(X['Humidity'], prefix='Humidity', drop_first=True)], axis=1)
X = pd.concat([X, pd.get_dummies(X['GeneralDiffuseFlows'], prefix='GeneralDiffuseFlows', drop_first=True)], axis=1)
X = pd.concat([X, pd.get_dummies(X['DiffuseFlows'], prefix='DiffuseFlows', drop_first=True)], axis=1)
X = X.drop(['Temperature', 'Humidity', 'GeneralDiffuseFlows', 'DiffuseFlows'], axis=1)


"""
OLS with back shifting of 3
"""

NUM_TIME_STEPS = 3  # includes number of significant time steps according to PACF graphs

# Create back-shifted columns for an attribute. (Wk2 Lab)
def addBackShiftedColumns(df, colName, timeLags):
    for i in range(1, timeLags + 1):
        newColName = colName + "_t-" + str(i)
        df[newColName] = df[colName].shift(i)
    if colName == 'PowerConsumption_Zone2':
        newColName = colName + "_t-" + str(143)
        df[newColName] = df[colName].shift(143)
        newColName = colName + "_t-" + str(144)
        df[newColName] = df[colName].shift(144)
    return df


# Columns to use, in this dataset all are significant from looking at p-values
columns = ['PowerConsumption_Zone2', 'Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']
modelDf = df.copy()
for i in range(0, len(columns)):
    modelDf = addBackShiftedColumns(modelDf, columns[i],
                                    NUM_TIME_STEPS)
print(modelDf.head())
y = modelDf[['PowerConsumption_Zone2']]
X = modelDf[['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']]


X = sm.add_constant(X)

# Calculate the root mean squared error (RMSE)
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print("\n---------- OLS with back shifting ----------\n")
print("RMSE:", rmse)

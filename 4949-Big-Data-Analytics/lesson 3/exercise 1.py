import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Do not show warning.
pd.options.mode.chained_assignment = None  # default='warn'

dta = sm.datasets.sunspots.load_pandas().data
print(dta)
plt.plot(dta['YEAR'], dta['SUNACTIVITY'])
plt.show()

# Show autocorrelation function.
# General correlation of lags with past lags.
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(dta['SUNACTIVITY'], lags=20)
plt.show()

# Show partial-autocorrelation function.
# Shows correlation of 1st lag with past lags.
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(dta['SUNACTIVITY'], lags=20)
plt.show()

print(dta)

""" Backtracking column """

##################################################################
# CONFIGURATION SECTION
NUM_DAYS        = 1200
NUM_TIME_STEPS  = 2
TEST_DAYS       = 10
##################################################################


# Creates time shifted columns for as many time steps needed.
def backShiftColumns(df, originalColName, numTimeSteps):
    dfNew  = df[[originalColName]]

    for i in range(1, numTimeSteps + 1):
        newColName = originalColName[0] + 't-' + str(i)
        dfNew[newColName] = dfNew[originalColName].shift(periods=i)
    return dfNew


def prepareSunDf(columns):
    df = dta

    # Create data frame with back shift columns for all features of interest.
    mergedDf = pd.DataFrame()
    for i in range(0, len(columns)):
        backShiftedDf = backShiftColumns(df, columns[i], NUM_TIME_STEPS)
        if i == 0:
            mergedDf = backShiftedDf
        else:
            mergedDf = mergedDf.merge(backShiftedDf, left_index=True, right_index=True)

    newColumns = list(mergedDf.keys())

    # Append stock symbol to column names.
    for i in range(0, len(newColumns)):
        mergedDf.rename(columns={newColumns[i]: "_" + newColumns[i]}, inplace=True)

    return mergedDf


columns = ['YEAR', 'SUNACTIVITY']
mergedDf = prepareSunDf(columns)
mergedDf = mergedDf.dropna()
print(mergedDf)

import seaborn as sns

corr = mergedDf.corr()
plt.figure(figsize=(8, 6))
ax = sns.heatmap(corr[['_SUNACTIVITY']], linewidth=0.5, vmin=-1, vmax=1, cmap="YlGnBu")
plt.title("Exercise 1")
plt.show()


""" Least Squares (OLS) Regression Code """


xfeatures = ['_St-1', '_St-2']
X = mergedDf[xfeatures]
y = mergedDf[['_SUNACTIVITY']]

# Add intercept for OLS regression.
import statsmodels.api       as sm
X = sm.add_constant(X)

# Split into test and train sets. The test data must be
# the latest data range.
lenData = len(X)
X_train = X[0:lenData-TEST_DAYS]
y_train = y[0:lenData-TEST_DAYS]
X_test  = X[lenData-TEST_DAYS:]
y_test  = y[lenData-TEST_DAYS:]

# Model and make predictions.
model       = sm.OLS(y_train, X_train).fit()
print(model.summary())
predictions = model.predict(X_test)

# Show RMSE and plot the data.
from sklearn  import metrics
import numpy as np
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

plt.plot(y_test, label='Actual', marker='o')
plt.plot(predictions, label='Predicted', marker='o')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.subplots_adjust(bottom=0.2)
plt.title("Example 3")
plt.show()

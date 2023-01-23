from pandas_datareader import data as pdr
import yfinance as yfin # Work around until
                        # pandas_datareader is fixed.
import datetime
import matplotlib.pyplot as plt
import pandas as pd

def getStock(stk, ttlDays):
    numDays = int(ttlDays)
    # Only gets up until day before during
    # trading hours
    dt        = datetime.date.today()
    # For some reason, must add 1 day to get current stock prices
    # during trade hours. (Prices are about 15 min behind actual prices.)
    dtNow     = dt + datetime.timedelta(days=1)
    dtNowStr  = dtNow.strftime("%Y-%m-%d")
    dtPast    = dt + datetime.timedelta(days=-numDays)
    dtPastStr = dtPast.strftime("%Y-%m-%d")
    yfin.pdr_override()
    df = pdr.get_data_yahoo(stk, start=dtPastStr, end=dtNowStr)
    return df

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Do not show warning.
pd.options.mode.chained_assignment = None  # default='warn'

##################################################################
# CONFIGURATION SECTION
NUM_DAYS        = 1200
NUM_TIME_STEPS  = 2
TEST_DAYS       = 30
##################################################################
# Creates time shifted columns for as many time steps needed.
def backShiftColumns(df, originalColName, numTimeSteps):
    dfNew  = df[[originalColName]].pct_change()

    for i in range(1, numTimeSteps + 1):
        newColName       = originalColName[0] + 't-' + str(i)
        dfNew[newColName]= dfNew[originalColName].shift(periods=i)
    return dfNew

def prepareStockDf(stockSymbol, columns):
    df = getStock(stockSymbol, NUM_DAYS)

    # Create data frame with back shift columns for all features of interest.
    mergedDf = pd.DataFrame()
    for i in range(0, len(columns)):
        backShiftedDf  = backShiftColumns(df, columns[i], NUM_TIME_STEPS)
        if(i==0):
            mergedDf = backShiftedDf
        else:
            mergedDf = mergedDf.merge(backShiftedDf, left_index=True,
                       right_index=True)

    newColumns = list(mergedDf.keys())

    # Append stock symbol to column names.
    for i in range(0, len(newColumns)):
        mergedDf.rename(columns={newColumns[i]: stockSymbol +\
                        "_" + newColumns[i]}, inplace=True)

    return mergedDf

columns  = ['Open', 'Close']
msftDf   = prepareStockDf('MSFT', columns)
aaplDf   = prepareStockDf('AAPL', columns)
mergedDf = msftDf.merge(aaplDf, left_index=True, right_index=True)
mergedDf = mergedDf.dropna()
print(mergedDf)

import seaborn as sns

corr = mergedDf.corr()
plt.figure(figsize = (4,4))
ax = sns.heatmap(corr[['MSFT_Open']],
            linewidth=0.5, vmin=-1,
            vmax=1, cmap="YlGnBu")
plt.show()


#####################



xfeatures = ['MSFT_Ct-2', 'GOOGL_Ct-1']
X = mergedDf[xfeatures]
y = mergedDf[['MSFT_Open']]

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
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))

plt.plot(y_test, label='Actual', marker='o')
plt.plot(predictions, label='Predicted', marker='o')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.show()


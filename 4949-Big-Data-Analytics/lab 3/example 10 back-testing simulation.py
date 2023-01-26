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

def getStock(stk, ttlDays):
    numDays = int(ttlDays)
    # Only gets up until day before during
    # trading hours
    dt = datetime.date.today()
    # For some reason, must add 1 day to get current stock prices
    # during trade hours. (Prices are about 15 min behind actual prices.)
    dtNow = dt + datetime.timedelta(days=1)
    dtNowStr = dtNow.strftime("%Y-%m-%d")
    dtPast = dt + datetime.timedelta(days=-numDays)
    dtPastStr = dtPast.strftime("%Y-%m-%d")
    yfin.pdr_override()
    df = pdr.get_data_yahoo(stk, start=dtPastStr, end=dtNowStr)
    return df

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

dfStock = getStock('AAPL', 1200)
print(dfStock)

TOTAL_DAYS = 15

# Build feature set with backshifted closing prices.
dfStock['Close_t_1'] = dfStock['Close'].shift(1)
dfStock['Volume_t_1']= dfStock['Volume'].shift(1)
dfStock              = dfStock.dropna()
dataDf               = dfStock[['Open', 'Close_t_1', 'Volume_t_1']]
splitRowNum          = len(dataDf) - TOTAL_DAYS
trainDf, testDf      = dataDf[0:splitRowNum], dataDf[splitRowNum:]

# Create training set and copy of the training set.
trainDf.tail(TOTAL_DAYS)
history      = trainDf.copy()
predictions  = []
FEATURE_LIST = ['Close_t_1', 'Volume_t_1']

cashBalance = 20000
def udpateCashBalance(cashBalance, purchasePrice, sellPrice):
    qty  = int(cashBalance / purchasePrice)
    cost = purchasePrice * qty
    cashBalance -= cost

    revenue      = sellPrice * qty
    cashBalance += revenue
    return cashBalance



# Iterate to make predictions for the evaluation set.
for i in range(0, len(testDf)):

    # Find the best model with most recent features.
    model = pm.auto_arima(history[['Open']],
                          exogenous=history[FEATURE_LIST],
                          start_p=1, start_q=1,
                          test='adf',       # Use adftest to find optimal 'd'
                          max_p=3, max_q=3, # Set maximum p and q.
                          d=None,           # Let model determine 'd'.
                          seasonal=False,    # No Seasonality.
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)

    lenHistory     = len(history)
    subDf          = history[FEATURE_LIST]
    featureArray   = np.array(subDf)
    lastFeatureRow = featureArray[lenHistory - 1]

    # Make prediction with most recent data prior to prediction.
    pred, confint = model.predict(n_periods=1,
                          exogenous=lastFeatureRow.reshape(1,-1),
                          return_conf_int=True)
    predictions.append(pred)

    # Extract next row in the test set.
    open        = testDf.iloc[i]['Open']        # Open price morning.
    close_t_1   = testDf.iloc[i]['Close_t_1']   # Close price day before.
    volume_t_1  = testDf.iloc[i]['Volume_t_1']  # Volume day before.

    # Add most recently available training data.
    history     = history.append({
        "Open":open,
        "Close_t_1":close_t_1,
        "Volume_t_1":volume_t_1},
        ignore_index=True)


if(pred > close_t_1):
        cashBalance = udpateCashBalance(cashBalance, close_t_1, open)



plt.plot(testDf.index, testDf['Open'], marker='o',
         label='Actual', color='blue')
plt.plot(testDf.index, predictions, marker='o',
         label='Predicted', color='orange')
plt.legend()
plt.xticks(rotation=70)
plt.show()

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(testDf['Open'], predictions))
print('testDf RMSE: %.3f' % rmse)


print("The final cash balance is: " + str(cashBalance))
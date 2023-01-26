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

TOTAL_DAYS = 5

# Build feature set with backshifted closing prices.
dfStock['Close_t_1'] = dfStock['Close'].shift(1)
dfStock              = dfStock.dropna()
dfX                  = dfStock[['Open', 'Close_t_1']]
size                 = len(dfX) - TOTAL_DAYS
train, test          = dfX[0:size], dfX[size:]

# Create training set and copy of the training set.
train.tail(TOTAL_DAYS)
history     = train.copy()
predictions = []

# Iterate to make predictions for the evaluation set.
for i in range(0, len(test)):
    lenOpen = len(history[['Close_t_1']])
    print("\n\nModel " + str(i))
    print(history.shape)

    model = pm.auto_arima(history[['Open']],
                          start_p=1, start_q=1,
                          test='adf',       # Use adftest to find optimal 'd'
                          max_p=3, max_q=3, # Set maximum p and q.
                          d=None,           # Let model determine 'd'.
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True)
    fc, confint = model.predict(n_periods=1,
                          return_conf_int=True)
    predictions.append(fc)
    open        = test.iloc[i]['Open']
    close_t_1   = test.iloc[i]['Close_t_1']
    history     = history.append({"Open":open, "Close_t_1":close_t_1},
                                 ignore_index=True)

plt.plot(test.index, test['Open'], marker='o',
         label='Actual', color='blue')
plt.plot(test.index, predictions, marker='o',
         label='Predicted', color='orange')
plt.legend()
plt.xticks(rotation=70)
plt.show()

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test['Open'], predictions))
print('Test RMSE: %.3f' % rmse)

import warnings
import statsmodels.tsa.arima.model as sma
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import datetime
from pandas_datareader import data as pdr
import yfinance as yfin
warnings.filterwarnings("ignore")


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


def buildModel(df, ar, i, ma):
    model = sma.ARIMA(df['Open'], order=(ar, i, ma)).fit()
    return model


def predictAndEvaluate(model, test, title):
    print("\n***" + title)
    print(model.summary())

    start = len(train)
    end = start + len(test) - 1
    predictions = model.predict(start=start, end=end, dynamic=True)
    mse = mean_squared_error(predictions, test["Open"])
    rmse = np.sqrt(mse)
    print("RMSE: " + str(rmse))
    if title == '2_0_1':
        plt.plot(test.index, test['Open'], label='Actual Values', color='blue')
        plt.plot(test.index, predictions, label='Predicted Values AR(20)', color='orange')
        plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    return rmse


stkName = 'MSFT'
dfStock = getStock(stkName, 400)

# Split the data.
NUM_TEST_DAYS = 5
lenData = len(dfStock)
train = dfStock.iloc[0:lenData - NUM_TEST_DAYS, :]
test = dfStock.iloc[lenData - NUM_TEST_DAYS:, :]

plt.plot(dfStock.index, dfStock['Open'])
plt.show()

modelStats = []
for ar in range(0, 5):
    for ma in range(0, 5):
        model = buildModel(train, ar, 0, ma)
        title = str(ar) + "_0_" + str(ma)
        rmse = predictAndEvaluate(model, test, title)
        modelStats.append({"ar": ar, "ma": ma, "rmse": rmse})

dfSolutions = pd.DataFrame(data=modelStats)
dfSolutions = dfSolutions.sort_values(by=['rmse'])
print(dfSolutions)
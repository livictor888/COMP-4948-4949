from pandas_datareader import data as pdr
import yfinance as yfin # Work around until
                        # pandas_datareader is fixed.
import pandas as pd
import datetime

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

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

#days since jan 2022: 365 + 11 (jan 11)
NUM_DAYS = 376
df       = getStock('AMZN', NUM_DAYS)
print("Amazon")
print(df)

import matplotlib.pyplot as plt
def showStock(df, title):
    plt.plot(df.index, df['Close'])
    plt.title(title)
    plt.xticks(rotation=70)
    plt.show()
showStock(df, "Amazon Close Prices")



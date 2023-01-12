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


# Get Southwestern stock for last 60 days
NUM_DAYS = 60
df       = getStock('LUV', NUM_DAYS)
print("South West Airlines")
print(df)

# Create weekly summary of closing price standard deviations
from pandas.tseries.frequencies import to_offset
series = df['Close'].resample('W').std()
series.index = series.index + to_offset("5D")
summaryDf = series.to_frame()

# Convert datetime index to date and then graph it.
summaryDf.index = summaryDf.index.date
print(summaryDf)
showStock(summaryDf, "Weekly S.D. Southwest Airlines")

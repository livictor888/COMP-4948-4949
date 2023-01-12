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

NUM_DAYS = 1000
# Search Yahoo for the correct symbols.
df       = getStock('TD.TO', NUM_DAYS)
print("Toronto Dominion Bank Stock")
print(df)


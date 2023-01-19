from pandas_datareader import data as pdr
import yfinance as yfin # Work around until
                        # pandas_datareader is fixed.
import datetime
import matplotlib.pyplot as plt

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

df = getStock('AMD', 1100)
print(df)

rolling_mean  = df['Close'].rolling(window=20).mean()
rolling_mean2 = df['Close'].rolling(window=50).mean()

#plt.figure(figsize=(10,30))
df['Close'].plot(label='AMD Close ', color='gray', alpha=0.3)
rolling_mean.plot(label='AMD 20 Day SMA', style='--', color='orange')
rolling_mean2.plot(label='AMD 50 Day SMA', style='--',color='magenta')

plt.legend()
plt.show()

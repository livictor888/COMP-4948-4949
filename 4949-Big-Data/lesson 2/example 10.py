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

df = getStock('AMD', 200)

# Calculating the moving averages.
rolling_mean  = df['Close'].rolling(window=20).mean()
rolling_mean2 = df['Close'].rolling(window=50).mean()

# Calculate the exponentially smoothed series.
exp20 = df['Close'].ewm(span=20, adjust=False).mean()
exp50 = df['Close'].ewm(span=50, adjust=False).mean()

#plt.figure(figsize=(10,30))
df['Close'].plot(label='AMD Close ', color='gray', alpha=0.3)
rolling_mean.plot(label='AMD 20 Day MA', style='--', color='orange')
rolling_mean2.plot(label='AMD 50 Day MA', style='--',color='magenta')
exp20.plot(label='AMD 20 Day ES', style='--',color='green')
exp50.plot(label='AMD 50 Day ES', style='--',color='blue', alpha=0.5)
plt.legend()
plt.show()


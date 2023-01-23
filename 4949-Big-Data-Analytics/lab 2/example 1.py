from pandas_datareader import data as pdr
import yfinance as yfin # Work around until
                        # pandas_datareader is fixed.
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
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

def getNewBalance(startBalance, startPrice, endPrice):
    qty = int(startBalance / startPrice)
    cashLeftOver = startBalance - qty * startPrice
    endValue = qty * endPrice
    balance = cashLeftOver + endValue
    return balance

def showBuyAndHoldEarnings(df, balance):
    startClosePrice = df.iloc[0]['Close']
    endClosePrice = df.iloc[len(df) - 1]['Close']
    newBalance = getNewBalance(balance, startClosePrice, endClosePrice)
    print("Buy and hold closing balance: $" + str(round(newBalance, 2)))

def showStrategyEarnings(df, balance, lt, st):
    buyPrice = 0
    buyDate = None
    sellDate = None
    bought = False

    buySellDates = []
    prices = []

    dfStrategy = pd.DataFrame(columns=['buyDt', 'buy$', 'sellDt',
                                       'sell$', 'balance'])
    dates = list(df.index)
    for i in range(0, len(df)):
        if (df.iloc[i]['Buy'] and not bought):
            buyPrice = df.iloc[i]['Close']
            buyDate = dates[i]
            bought = True
            buySellDates.append(buyDate)
            prices.append(buyPrice)

        elif (df.iloc[i]['Sell'] and bought):
            sellPrice = df.iloc[i]['Close']
            balance = getNewBalance(balance, buyPrice, sellPrice)
            sellDate = dates[i]
            buySellInfo = {'buyDt': buyDate, 'buy$': buyPrice,
                           'sellDt': sellDate, 'sell$': sellPrice,
                           'balance': balance, }
            dfStrategy = dfStrategy.append(buySellInfo, ignore_index=True)
            bought = False
            buySellDates.append(sellDate)
            prices.append(sellPrice)

    print(dfStrategy)
    print("\nMoving average strategy closing balance: $" + str(round(balance, 2)))
    return buySellDates, prices

def showBuyAndSellDates(df, startBalance):
    strategyDates, strategyPrices = showStrategyEarnings(df, startBalance, lt, st)
    plt.plot(df.index, df['Close'], label='Close')
    plt.plot(df.index, df['ema20'], label='ema20', alpha=0.4)
    plt.plot(df.index, df['ema50'], label='ema50', alpha=0.4)
    plt.scatter(strategyDates, strategyPrices, label='Buy/Sell', color='red')
    plt.xticks(rotation=70)
    plt.legend()
    plt.show()

def showInvestmentDifferences(dfStock, lt, st):
    df = dfStock.copy()
    df['ema50'] = df['Close'].ewm(span=lt).mean()
    df['ema20'] = df['Close'].ewm(span=st).mean()

    # Remove nulls.
    df.dropna(inplace=True)
    df.round(3)
    own_positions = np.where(df['ema20'] > df['ema50'], 1, 0)
    df['Position'] = own_positions
    df.round(3)

    df['Buy'] = (df['Position'] == 1) & (df['Position'].shift(1) == 0)
    df['Sell'] = (df['Position'] == 0) & (df['Position'].shift(1) == 1)

    START_BALANCE = 10000

    print("-------------------------------------------------------")
    showBuyAndHoldEarnings(df, START_BALANCE)
    print("-------------------------------------------------------")
    showBuyAndSellDates(df, START_BALANCE)

longterms  = [50]
shortterms = [40]
dfStock    = getStock('XOM', 1100)

for lt in longterms:
    for st in shortterms:
        print("\b******************************************************")
        print("Lt: " + str(lt))
        print("St: " + str(st))
        showInvestmentDifferences(dfStock, lt, st)

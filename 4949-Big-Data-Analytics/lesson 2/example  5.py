from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt

# Import Data
PATH = 'C:\\datasets\\'
FILE      = "drugSales.csv"
df        = pd.read_csv(PATH + FILE, parse_dates=['date'], index_col='date')
tseries   = seasonal_decompose(df['value'], model='additive',
                               extrapolate_trend='freq')

plt.plot(df['value'])
plt.title("Drug Sales", fontsize=16)
plt.show()

detrended = df['value'] - tseries.trend
plt.plot(detrended)
plt.title('Drug Sales After Subtracting Trend', fontsize=16)
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt

# Import Data
PATH = 'C:\\datasets\\'
FILE      = "drugSales.csv"
df        = pd.read_csv(PATH + FILE, parse_dates=['date'], index_col='date')
tseries   = seasonal_decompose(df['value'], model='multiplicative',
                               extrapolate_trend='freq')

plt.plot(df['value'])
plt.title("Drug Sales", fontsize=16)
plt.show()

deseasonalized = df.value.values / tseries.seasonal
plt.plot(deseasonalized)
plt.title('Drug Sales After De-Seasonalizing', fontsize=16)
plt.show()

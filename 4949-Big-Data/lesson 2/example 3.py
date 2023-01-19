from statsmodels.tsa.seasonal import seasonal_decompose
import pandas                 as pd
import matplotlib.pyplot      as plt

# Import data.
PATH = 'C:\\datasets\\'
FILE = "drugSales.csv"
df   = pd.read_csv(PATH + FILE,  parse_dates=['date'], index_col='date')
type(df.index)

# Perform decomposition using additive decomposition.
tseries = seasonal_decompose(df['value'], model='additive', extrapolate_trend="freq")

tseries.plot()
plt.show()

# Extract the Components ----
# Actual Values = Product of (Seasonal * Trend * Resid)
dfComponents = pd.concat([tseries.seasonal, tseries.trend,
                          tseries.resid, tseries.observed], axis=1)
dfComponents.columns = ['seas', 'trend', 'resid', 'actual_values']
print(dfComponents.head())

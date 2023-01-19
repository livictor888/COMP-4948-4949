from statsmodels.tsa.seasonal import seasonal_decompose
import pandas                 as pd
import matplotlib.pyplot      as plt
import matplotlib.dates       as mdates
# Import data.
PATH = 'C:\\datasets\\'
FILE = "AirPassengers.csv"
df   = pd.read_csv(PATH + FILE,  parse_dates=['date'], index_col='date')
type(df.index)

fig, ax = plt.subplots()

# Perform decomposition using multiplicative decomposition.
tseries  = seasonal_decompose(df['value'], model='multiplicative',
                              extrapolate_trend='freq')
trend    = tseries.trend
seasonal = tseries.seasonal

# Set vertical major grid.
ax.xaxis.set_major_locator(mdates.YearLocator(day=1))
ax.xaxis.grid(True, which = 'major', linewidth = 1, color = 'black')

# Set vertical minor grid.
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1,4,7,10),bymonthday=1))
ax.xaxis.grid(True, which = 'minor', linewidth = 1, color = 'red')

start, end = '2005-01', '2009-12'
ax.plot(seasonal.loc[start:end], color='green')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
plt.setp(ax.xaxis.get_minorticklabels(), rotation=70)

ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
plt.title("Seasonal Drug Sales")
plt.show()


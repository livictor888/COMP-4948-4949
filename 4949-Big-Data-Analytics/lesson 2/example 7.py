from pandas import read_csv
import matplotlib.pyplot as plt

PATH = 'C:\\datasets\\'
FILE   = 'daily-total-female-births.csv'
series = read_csv(PATH + FILE, header=0, index_col=0)
print(series.head())
series.plot(rot=45)
plt.show()

# Calculate rolling moving average 3 steps back.
print("\n*** Rolling mean")
rolling      = series.rolling(window=3)
rolling_mean = rolling.mean()
print(rolling_mean.head(5))

# Plot actual and rolling mean values.
plt.plot(series, color='blue', label='female births')
plt.plot(rolling_mean, color='red', label='rolling mean')
plt.legend()
plt.show()

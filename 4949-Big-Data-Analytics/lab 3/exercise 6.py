import pandas as pd
PATH        = "C:\\datasets\\"
FILE_NAME   = "Energy_Production.csv"
import matplotlib.pyplot as plt

# Import
data = pd.read_csv(PATH + FILE_NAME, index_col=0)

data.index = pd.to_datetime(data.index)
print(data.head())
print(data.describe())

fig, axes = plt.subplots(2, 1, figsize=(10,5), dpi=100, sharex=True)

# Usual Differencing
axes[0].plot(data[:], label='Original Series')
axes[0].plot(data[:].diff(1), label='Usual Differencing')
axes[0].set_title('Usual Differencing')
axes[0].legend(loc='upper left', fontsize=10)

# Seasonal Differencing
axes[1].plot(data[:], label='Original Series')
axes[1].plot(data[:].diff(12), label='Seasonal Differencing', color='green')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper left', fontsize=10)
plt.show()

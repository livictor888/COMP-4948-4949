import pandas as pd
import datetime

import matplotlib.pyplot as plt
PATH = "C:\\datasets\\"
FILE = 'shampoo.csv'
df   = pd.read_csv(PATH + FILE, index_col=0)
df.info()

# Plot data before differencing.
df.plot()
plt.xticks(rotation=45)
plt.show()

# Perform differencing.
dfDifferenced = df.diff()

# Plot data after differencing.
plt.plot(dfDifferenced)
plt.xticks(rotation=75)
plt.show()

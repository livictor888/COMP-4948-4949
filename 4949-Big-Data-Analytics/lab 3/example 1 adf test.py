import warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Import data
df = pd.read_csv(
"https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv", \
                  names=['value'], header=0)
print(df)
df.value.plot()
plt.title("www usage")
plt.show()

from statsmodels.tsa.stattools import adfuller
result = adfuller(df.value.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

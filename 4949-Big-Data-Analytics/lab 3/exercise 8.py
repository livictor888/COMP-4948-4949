import pandas as pd
PATH        = "C:\\datasets\\"
FILE_NAME   = "Energy_Production.csv"
import matplotlib.pyplot as plt

# Import
data = pd.read_csv(PATH + FILE_NAME, index_col=0)

data.index = pd.to_datetime(data.index)
print(data.head())
print(data.describe())

# truncate data to 300 rows
data = data[0:300]

import pmdarima as pm

# Seasonal - fit stepwise auto-ARIMA
smodel = pm.auto_arima(data, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=12,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',
                         suppress_warnings=True,
                         stepwise=True)

print(smodel.summary())

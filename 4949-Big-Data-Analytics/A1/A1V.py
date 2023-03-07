"""
Big Data Analysis for powerconsumption.csv
"""
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import xgboost as xgb
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning))

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)

# Load the dataset into a pandas DataFrame object
PATH = "C:\\Users\Victor\\PycharmProjects\\COMP-4948-4949\\4949-Big-Data-Analytics\\A1\\"
CSV_DATA = "powerconsumption.csv"
df = pd.read_csv(PATH + CSV_DATA, sep=',')

print("\n---------- Original dataset ----------\n")
print(df.head(10))
print()
print(df.describe().T)
print("\n---------- Original dataset ----------\n")

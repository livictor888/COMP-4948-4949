"""
Big Data Analysis for powerconsumption.csv
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
from pandas_profiling import ProfileReport
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.metrics import mean_squared_error

warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning))

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Load the dataset into a pandas DataFrame object
df = pd.read_csv('fake_bills.csv', delimiter=";")
print(df.head())



# Generate a report with the adjusted data
prof = ProfileReport(df)
prof.to_file(output_file='output.html')
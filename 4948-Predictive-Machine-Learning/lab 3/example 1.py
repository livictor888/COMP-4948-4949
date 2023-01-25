import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.api       as sm

PATH     = "C:\\datasets\\"
CSV_DATA = "winequality.csv"

dataset  = pd.read_csv(PATH + CSV_DATA)
# Show all columns.
pd.set_option('display.max_columns', None)

# Include only statistically significant columns.
X = dataset[['volatile acidity', 'chlorides', 'total sulfur dioxide',
             'pH', 'sulphates','alcohol']]
y = dataset['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Stochastic gradient descent models are sensitive to differences
# in scale so a MinMax is usually used.
from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler()
scalerX.fit(X_train)

print(y_train)

import numpy  as np
# Create array of y-values and reshape.
y_train = np.array(y_train).reshape(-1, 1)
print("\nModified y_train")
print(y_train)

# Build scaler for y.
scalerY = MinMaxScaler()
scalerY.fit(y_train)
y_scaled = scalerY.transform(y_train)

print("\nScaled y")
print(y_scaled)



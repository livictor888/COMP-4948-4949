import pandas                as pd
import numpy                 as np
from   sklearn               import metrics
from sklearn.model_selection import train_test_split
import statsmodels.api       as sm

PATH     = "C:\\datasets\\"
CSV_DATA = "housing.data"

df       = pd.read_csv(PATH + CSV_DATA,  header=None)

# Show all columns on one line.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(df.head())
print(df.tail())
print(df.describe())

dataset = df.values

# Split data into input (X) and output (Y) variables.
X = dataset[:,0:13]
y = dataset[:,13]

# Adding an intercept *** This is required ***. Don't forget this step.
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.
X = sm.add_constant(X)

# Split the data.
X_train, X_temp, y_train, y_temp = train_test_split(X,
         y, test_size=0.3, random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_temp,
         y_temp, test_size=0.5, random_state=0)

# Make predictions and evaluate with the RMSE.
model       = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test)
print(model.summary())
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))


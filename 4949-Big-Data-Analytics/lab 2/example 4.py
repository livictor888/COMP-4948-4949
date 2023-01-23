import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np
import statsmodels.api   as sm
from sklearn.preprocessing import MinMaxScaler

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Do not show warning.
pd.options.mode.chained_assignment = None

# Load the data.
PATH = "C:\\datasets\\"
FILE = 'DailyDelhiClimateTest.csv'
df   = pd.read_csv(PATH + FILE, parse_dates=['date'], index_col='date')
print(df)

# Create back-shifted columns for an attribute.
def addBackShiftedColumns(df, colName, timeLags):
      for i in range(1, timeLags+1):
            newColName = colName + "_t-" + str(i)
            df[newColName] = df[colName].shift(i)
      return df

# Build dataframe for modelling.
columns = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
modelDf = df.copy()
NUM_TIME_STEPS = 3
for i in range(0, len(columns)):
      modelDf = addBackShiftedColumns(modelDf, columns[i],
                                      NUM_TIME_STEPS)
modelDf = modelDf.dropna()
y = modelDf[['meantemp']]
X = modelDf[[ 'meantemp_t-1']]

# Add intercept for OLS regression.
X         = sm.add_constant(X)
TEST_DAYS = 10

# Split into test and train sets. The test data includes
# the latest values in the data.
lenData = len(X)
X_train = X[0:lenData-TEST_DAYS]
y_train = y[0:lenData-TEST_DAYS]
X_test  = X[lenData-TEST_DAYS:]
y_test  = y[lenData-TEST_DAYS:]

# Model and make predictions.
model       = sm.OLS(y_train, X_train).fit()
print(model.summary())
predictions = model.predict(X_test)

# Show RMSE.
from sklearn  import metrics
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Plot the data.
xaxisValues = list(y_test.index)
plt.plot(xaxisValues, y_test, label='Actual', marker='o')
plt.plot(xaxisValues, predictions, label='Predicted', marker='o')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.title("Mean temperature in Dehli")
plt.show()

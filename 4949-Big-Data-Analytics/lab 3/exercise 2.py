import warnings
warnings.filterwarnings("ignore")

import pandas                 as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.arima.model as sma
from   sklearn.metrics import mean_squared_error
from   math import sqrt
import numpy as np
import pmdarima as pm

PATH   = "C:\\datasets\\"
df = pd.read_csv(PATH + 'daily-total-female-births.csv',
                  header=0, index_col=0)
print(df.head())
print(df.describe())



# Split the data set so the test set is 7.
NUM_TEST_DAYS = 7
X    = df.values
size = len(X) - NUM_TEST_DAYS
train, test = X[0:size], X[size:]

def difference(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return np.array(diff)

# Create a list with the training array.
history     = [x for x in train]
predictions = []

# predict() receives the model coefficients and all past data (t-1, t-2, t-2) etc.
def predict(coef, history):
    yhat = 0.0
    for i in range(1, len(coef) + 1):
        # Make the prediction (yhat)
        # This multiplies L1coeff*L1
        # and L2coeff*L2 if it exists
        # and L3coeff*L3 if it exists
        yhat += coef[i - 1] * history[-i]
    return yhat  # Return the prediction.

for t in range(len(test)):
    print("History length: " + str(len(history)))

    #################################################################
    # Model building and prediction section.
    # Build day ahead model.
    model = pm.auto_arima(history, start_p=1, start_q=1,
                          test='adf',
                          max_p=3, max_q=3, m=0,
                          start_P=0, seasonal=False,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)

    print(model.summary())
    fc, confint = model.predict(n_periods=1,
                                return_conf_int=True)
    yhat = fc[0]
    #################################################################


    predictions.append(yhat)  # Store the prediction in a list.

    obs = test[t]  # Get the actual current value.
    history.append(obs)  # Append the actual current value to the training data.
    # Actual values will be used as t-1, t-2 etc next iteration.
    print('>predicted=%.3f, expected=%.3f' % (yhat, obs))

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

plt.plot(test, label='Actual', marker='o', color='blue')
plt.plot(predictions, label='Predictions', marker='o', color='orange')
plt.legend()
plt.title("AR Model")
plt.show()

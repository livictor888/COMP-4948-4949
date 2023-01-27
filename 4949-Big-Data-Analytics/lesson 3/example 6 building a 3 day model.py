from pandas                     import read_csv
import matplotlib.pyplot        as plt
from statsmodels.tsa.ar_model   import AutoReg
from sklearn.metrics            import mean_squared_error
from   math                     import sqrt
import warnings
warnings.filterwarnings("ignore")

# Load the data.
PATH = "C:\\datasets\\"
series = read_csv(PATH + 'daily-min-temperatures.csv',
                  header=0, index_col=0, parse_dates=True, squeeze=True)

# Plot ACF.
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series, lags=20)
plt.show()

# Plot PACF.
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(series, lags=20)
plt.show()


NUM_TEST_DAYS = 7

# Split dataset into test and train.
X       = series.values
lenData = len(X)
train   = X[0:lenData-NUM_TEST_DAYS]
test    = X[lenData-NUM_TEST_DAYS:]

# Train.
model     = AutoReg(train, lags=3)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)

print(model_fit.summary())

# Make predictions.
predictions = model_fit.predict(start=len(train),
                                end=len(train)+len(test)-1,
                                dynamic=False)

for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions[i], test[i]))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# Plot results.
plt.plot(test, marker='o', label='actual')
plt.plot(predictions, color='brown', linewidth=4,
         marker='o', label='predicted')

plt.legend()
plt.show()

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


NUM_TEST_DAYS = 3

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



# Use model coefficients from autoregression to make a prediction.
def makePrediction(t_1, t_2, t_3):
    intercept =   1.88820768
    t1Coeff   =   0.70018223
    t2Coeff   = - 0.05949822
    t3Coeff   =   0.19010829

    prediction = intercept + t1Coeff*t_1\
               + t2Coeff*t_2\
               + t3Coeff*t_3
    return prediction

testLen    = len(test)

t_1 = test[testLen-1]
t_2 = test[testLen-2]
t_3 = test[testLen-3]

futurePredictions = []
for i in range(0, NUM_TEST_DAYS):
    prediction = makePrediction(t_1, t_2, t_3)
    futurePredictions.append(prediction)
    t_3 = t_2
    t_2 = t_1
    t_1 = prediction

print("Here is a one week temperature forecast: ")
print(futurePredictions)


""" Example 3: Comparing Predictions / Comparing AR Models """

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings("ignore")

dta = sm.datasets.sunspots.load_pandas().data
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'), freq='Y')

# Show autocorrelation function.
# General correlation of lags with past lags.
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(dta['SUNACTIVITY'], lags=50)
plt.show()

# Split the data.
NUM_TEST_YEARS = 10
lenData        = len(dta)
dfTrain        = dta.iloc[0:lenData - NUM_TEST_YEARS, :]
dfTest         = dta.iloc[lenData-NUM_TEST_YEARS:,:]


def buildModelAndMakePredictions(AR_time_steps, dfTrain, dfTest):
    # This week we will use the ARIMA model.

    model = ARIMA(dfTrain['SUNACTIVITY'], order=(AR_time_steps, 0, 0), freq='Y').fit()
    print("\n*** Evaluating ARMA(" + str(AR_time_steps) + ",0,0)")
    print('Coefficients: %s' %model.params)

    # Strings which can be converted to time stamps are passed in.
    # For this case the entire time range for the test set is represented.
    predictions = model.predict('1999-12-31', '2008-12-31', dynamic=True)
    rmse = np.sqrt(mean_squared_error(dfTest['SUNACTIVITY'].values,
                                      np.array(predictions)))
    print('Test RMSE: %.3f' % rmse)
    print('Model AIC %.3f' % model.aic)
    print('Model BIC %.3f' % model.bic)
    return model, predictions


print(dfTest)
arma_mod20, predictionsARMA_20 = buildModelAndMakePredictions(2, dfTrain, dfTest)
arma_mod30, predictionsARMA_30 = buildModelAndMakePredictions(3, dfTrain, dfTest)
plt.plot(dfTest.index, dfTest['SUNACTIVITY'], label='Actual Values', color='blue')
plt.plot(dfTest.index, predictionsARMA_20, label='Predicted Values AR(20)', color='orange')
plt.plot(dfTest.index, predictionsARMA_30, label='Predicted Values AR(30)', color='brown')
plt.legend(loc='best')
plt.show()
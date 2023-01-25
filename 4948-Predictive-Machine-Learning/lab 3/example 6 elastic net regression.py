import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn                 import metrics
import statsmodels.api       as sm
import numpy as np
PATH     = "C:\\datasets\\"
CSV_DATA = "winequality.csv"
dataset  = pd.read_csv(PATH + CSV_DATA)

X = dataset[['volatile acidity', 'chlorides', 'total sulfur dioxide', 'sulphates',
             'alcohol']]

# Adding an intercept *** This is requried ***. Don't forget this step.
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.
X_withConst = sm.add_constant(X)
y = dataset['quality'].values

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api       as sm

# Show all columns.
pd.set_option('display.max_columns', None)

# Include only statistically significant columns.
X = dataset[['volatile acidity', 'chlorides', 'total sulfur dioxide',
             'pH', 'sulphates','alcohol']]
X = sm.add_constant(X)
y = dataset['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Stochastic gradient descent models are sensitive to scaling.
# Fit X scaler and transform X_train.
from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler()
X_train_scaled = scalerX.fit_transform(X_train)

# Build y scaler and transform y_train.
scalerY = StandardScaler()
y_train_scaled = scalerY.fit_transform(np.array(y_train).reshape(-1,1))

# Scale test data.
X_test_scaled = scalerX.transform(X_test)

def performLinearRegression(X_train, X_test, y_train, y_test, scalerY):
    model = sm.OLS(y_train, X_train).fit()
    scaledPredictions = model.predict(X_test) # make the predictions by the model
    predictions = scalerY.inverse_transform(np.array(scaledPredictions).reshape(-1,1))
    print(model.summary())
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    return predictions

predictions = performLinearRegression(X_train_scaled, X_test_scaled,
                                      y_train_scaled, y_test, scalerY)



from sklearn.linear_model import SGDRegressor
def performSGD(X_train, X_test, y_train, y_test, scalerY):
    sgd = SGDRegressor()
    sgd.fit(X_train, y_train)
    print("\n***SGD=")
    predictionsUnscaled = sgd.predict(X_test)
    predictions = scalerY.inverse_transform(
        np.array(predictionsUnscaled).reshape(-1,1))

    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test,
                                             predictions)))
predictions = performSGD(X_train_scaled, X_test_scaled,
                                      y_train_scaled, y_test, scalerY)



from sklearn.linear_model import Ridge
def ridge_regression(X_train, X_test, y_train, y_test, alpha, scalerY):
    # Fit the model
    ridgereg = Ridge(alpha=alpha)
    ridgereg.fit(X_train, y_train)
    y_pred_scaled = ridgereg.predict(X_test)

   # predictions = scalerY.inverse_transform(y_pred.reshape(-1,1))
    print("\n***Ridge Regression Coefficients ** alpha=" + str(alpha))
    print(ridgereg.intercept_)
    print(ridgereg.coef_)
    y_pred = scalerY.inverse_transform(np.array(y_pred_scaled).reshape(-1,1))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                         test_size=0.2, random_state=0)
alphaValues = [0,  0.16, 0.17, 0.18]
for i in range(0, len(alphaValues)):
    ridge_regression(X_train_scaled, X_test_scaled, y_train_scaled,
                     y_test, alphaValues[i], scalerY)



## LASSO REGRESSIOn


from sklearn.linear_model import Lasso
def performLassorRegression(X_train, X_test, y_train, y_test, alpha,
                            scalerY):
    lassoreg = Lasso(alpha=alpha)
    lassoreg.fit(X_train, y_train)
    y_pred_scaled = lassoreg.predict(X_test)
    y_pred = scalerY.inverse_transform(np.array(y_pred_scaled).reshape(-1,1))
    print("\n***Lasso Regression Coefficients ** alpha=" + str(alpha))
    print(lassoreg.intercept_)
    print(lassoreg.coef_)
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
alphaValues = [0, 0.1, 0.5, 1]
for i in range(0, len(alphaValues)):
    performLassorRegression(X_train_scaled, X_test_scaled, y_train_scaled,
                     y_test, alphaValues[i], scalerY)



#### elastic net regression



from sklearn.linear_model import ElasticNet

bestRMSE = 100000.03
def performElasticNetRegression(X_train, X_test, y_train, y_test, alpha, l1ratio, bestRMSE,
                                bestAlpha, bestL1Ratio, scalerY):
    model = ElasticNet(alpha=alpha, l1_ratio=l1ratio)
    # fit model
    model.fit(X_train, y_train)
    y_pred_scaled = model.predict(X_test)
    y_pred = scalerY.inverse_transform(np.array(y_pred_scaled).reshape(-1,1))
    print("\n***ElasticNet Regression Coefficients ** alpha=" + str(alpha)
          + " l1ratio=" + str(l1ratio))
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print(model.intercept_)
    print(model.coef_)
    try:
        if(rmse < bestRMSE):
            bestRMSE = rmse
            bestAlpha = alpha
            bestL1Ratio = l1ratio
        print('Root Mean Squared Error:', rmse)
    except:
        print("rmse =" + str(rmse))

    return bestRMSE, bestAlpha, bestL1Ratio

alphaValues = [0, 0.00001, 0.0001, 0.001, 0.01, 0.18]
l1ratioValues = [0, 0.25, 0.5, 0.75, 1]
bestAlpha   = 0
bestL1Ratio = 0

for i in range(0, len(alphaValues)):
    for j in range(0, len(l1ratioValues)):
        bestRMSE, bestAlpha, bestL1Ratio = performElasticNetRegression(
                         X_train_scaled, X_test_scaled, y_train_scaled,
                        y_test,
                         alphaValues[i], l1ratioValues[j], bestRMSE,
                         bestAlpha, bestL1Ratio, scalerY)

print("Best RMSE " + str(bestRMSE) + " Best alpha: " + str(bestAlpha)
      + "  " + "Best l1 ratio: " + str(bestL1Ratio))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn                 import metrics
import statsmodels.api       as sm
import numpy as np
PATH     = "C:\\datasets\\"
CSV_DATA = "petrol_consumption.csv"
dataset  = pd.read_csv(PATH + CSV_DATA)
#   Petrol_Consumption
X = dataset[['Petrol_tax','Average_income', 'Population_Driver_licence(%)']]

# Adding an intercept *** This is requried ***. Don't forget this step.
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.
X_withConst = sm.add_constant(X)
y = dataset['Petrol_Consumption'].values

X_train, X_test, y_train, y_test = train_test_split(X_withConst, y,
                                                    test_size=0.2, random_state=0)

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

def performLinearRegression(X_train, X_test, y_train, y_test):
    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test) # make the predictions by the model
    print(model.summary())
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    return predictions

predictions = performLinearRegression(X_train, X_test, y_train, y_test)



# LASSO
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

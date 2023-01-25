import  pandas as pd
from    sklearn.model_selection import train_test_split
PATH    = "C:\\datasets\\"
from   sklearn.linear_model    import LogisticRegression
from   sklearn                 import metrics
import numpy as np

# load the dataset
df = pd.read_csv(PATH + 'diabetes.csv', sep=',')
# split into input (X) and output (y) variables

X = df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI',
        'DiabetesPedigreeFunction',    'Age']]
y = df[['Outcome']]
# Split into train and test data sets.
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33)

# Perform logistic regression.
logisticModel = LogisticRegression(fit_intercept=True, random_state = 0,
                                   solver='liblinear')
logisticModel.fit(X_train,y_train)
y_pred=logisticModel.predict(X_test)

# Show model coefficients and intercept.
print("\nModel Coefficients: ")
print("\nIntercept: ")
print(logisticModel.intercept_)

print(logisticModel.coef_)

# Show confusion matrix and accuracy scores.
confusion_matrix = pd.crosstab(np.array(y_test['Outcome']), y_pred,
                               rownames=['Actual'],
                               colnames=['Predicted'])

print('\nAccuracy: ',metrics.accuracy_score(y_test, y_pred))
print("\nConfusion Matrix")
print(confusion_matrix)



# Import svm package
from sklearn import svm

# Create a svm Classifier using one of the following options:
# linear, polynomial, and radial
clf = svm.SVC(kernel='rbf')

# Train the model using the training set.
clf.fit(X_train, y_train)

# Evaluate the model.
y_pred = clf.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))




##### EXMAPLE 2 CODE

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

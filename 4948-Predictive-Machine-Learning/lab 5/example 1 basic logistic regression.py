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

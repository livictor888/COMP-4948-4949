#Import scikit-learn dataset library
from sklearn import datasets
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.simplefilter(action='ignore', category=UserWarning)

#Load dataset
iris = datasets.load_iris()

# Creating a DataFrame of given iris dataset.
import pandas as pd
data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})
iris['target_names']
print(data.head())

# Import train_test_split function
from sklearn.model_selection import train_test_split
X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y=data['species']  # Labels

feature_list = ['sepal length', 'sepal width', 'petal length', 'petal width']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
rf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=rf.predict(X_test)
rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Predict species for a single flower.
# sepal length = 3, sepal width = 5
# petal length = 4, petal width = 2
prediction = rf.predict([[3, 5, 4, 2]])
# 'setosa', 'versicolor', 'virginica'
print(prediction)




# Get numerical feature importances
importances = list(rf.feature_importances_)

# Present features and importance scores.
def showFeatureImportances(importances, feature_list):
    dfImportance = pd.DataFrame()
    for i in range(0, len(importances)):
        dfImportance = pd.concat([dfImportance, pd.DataFrame({"importance":[importances[i]], "feature":[feature_list[i]] })],
                                 ignore_index = True)

    dfImportance = dfImportance.sort_values(by=['importance'],
                                            ascending=False)
    print(dfImportance)
showFeatureImportances(importances, feature_list)


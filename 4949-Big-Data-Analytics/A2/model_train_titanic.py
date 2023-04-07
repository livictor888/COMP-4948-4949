# Actually use 4 features
# Final code

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score

def preprocess_data(df):
    # Fill missing values
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    return df

# Read the dataset
url = 'https://raw.githubusercontent.com/kelly-olsson/titanic/main/train.csv'
df = pd.read_csv(url)

# Preprocess the data
df = preprocess_data(df)

# Separate into x and y values
predictorVariables = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[predictorVariables]
y = df['Survived']

# Show chi-square scores for each feature
test = SelectKBest(score_func=chi2, k=4)
chiScores = test.fit(X, y)
np.set_printoptions(precision=3)

print("\nPredictor variables: " + str(predictorVariables))
print("Predictor Chi-Square Scores: " + str(chiScores.scores_))

# Select significant variables using the get_support() function
cols = chiScores.get_support(indices=True)
print(cols)
features = X.columns[cols]
print(np.array(features))

# Re-assign X with significant columns only after chi-square test
X = df[features]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Build logistic regression model and make predictions
logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear', random_state=0)
logisticModel.fit(X_train, y_train)

# Save the model
with open('model_pkl', 'wb') as files:
    pickle.dump(logisticModel, files)

# Load saved model
with open('model_pkl', 'rb') as f:
    loadedModel = pickle.load(f)

y_pred = loadedModel.predict(X_test)
print(y_pred)

# Show confusion matrix and accuracy scores
cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print('\nAccuracy:', accuracy)
print("\nConfusion Matrix")
print(cm)
print("Recall: " + str(recall))
print("Precision: " + str(precision))

# Create a single prediction.
singleSampleDf = pd.DataFrame(columns=['Pclass', 'Sex', 'Age', 'Fare'])
pClass =  3
sex = 0
age = 22
fare = 7.25

passengerData = {'Pclass':pClass, 'Sex':sex, 'Age':age, 'Fare':fare}
singleSampleDf = pd.concat([singleSampleDf,
                            pd.DataFrame.from_records([passengerData])])

singlePrediction = loadedModel.predict(singleSampleDf)
print("Single prediction: " + str(singlePrediction))
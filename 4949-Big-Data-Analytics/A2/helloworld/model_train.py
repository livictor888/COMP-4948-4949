import pandas as pd
import numpy as np
import pickle
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score


def preprocess_data(df):
    # Replace missing values
    df.fillna(df.mean(), inplace=True)

    return df


# Read the dataset
df = pd.read_csv('fake_bills.csv', delimiter=";")
print(df.head())

# Preprocess the data
df = preprocess_data(df)

# Separate into x and y values
predictorVariables = ['length', 'margin_low', 'margin_up', 'diagonal', 'height_right', 'height_left']
X = df[predictorVariables]
y = df['is_genuine']

# Show chi-square scores for each feature
test = SelectKBest(score_func=chi2, k=6)
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
singleSampleDf = pd.DataFrame(columns=features)

# Replace these values with the desired data points for prediction
data_point = {
    'length': 171.81,
    'margin_low': 4.66,
    'margin_up': 3.02,
    'diagonal': 111.76,
    'height_right': 104.23,
    'height_left': 103.54
}

billData = {k: data_point[k] for k in features}
singleSampleDf = pd.concat([singleSampleDf,
                            pd.DataFrame.from_records([billData])])

singlePrediction = loadedModel.predict(singleSampleDf)
print("Single prediction: " + str(singlePrediction))
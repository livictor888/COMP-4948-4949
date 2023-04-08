import pandas as pd
import numpy as np
import pickle
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
import matplotlib.pyplot as plt


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
predictorVariables = ['length', 'margin_low',
                      'margin_up', 'diagonal', 'height_right', 'height_left']
X = df[predictorVariables]
y = df['is_genuine']

# Show chi-square scores for each feature
test = SelectKBest(score_func=chi2, k='all')  # Changed k to 'all'
chiScores = test.fit(X, y)
np.set_printoptions(precision=3)

print("\nPredictor variables: " + str(predictorVariables))
print("Predictor Chi-Square Scores: " + str(chiScores.scores_))

# Plot the bar graph of predictor variables and their chi-square scores
plt.bar(predictorVariables, chiScores.scores_)
plt.title('Chi-Square Scores of Predictor Variables')
plt.xlabel('Predictor Variables')
plt.ylabel('Chi-Square Scores')
plt.show()

# Find the most significant predictor variables
sorted_idx = np.argsort(chiScores.scores_)[::-1]
top_features = np.array(predictorVariables)[sorted_idx]
print("\nMost significant predictor variables:")
print(top_features)


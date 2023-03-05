import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Load the dataset into a pandas DataFrame object
df = pd.read_csv('https://drive.google.com/uc?id=1r29kUG_mtmiQFVuUTJqogNo_l0jNz_iK')

# Split the dataset into a target variable and predictor variables
y = df['Loan_Status']
X = df.drop('Loan_Status', axis=1)

# Create a logistic regression object
logreg = LogisticRegression()

# Create a RFE object with 10 features
rfe = RFE(logreg)

# Fit the RFE object to the predictor variables
rfe.fit(X, y)

# Print the rankings of each predictor variable
print('Ranking of predictor variables:')
print(pd.DataFrame({'Predictor Variable': X.columns, 'Ranking': rfe.ranking_}))
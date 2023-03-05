
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

# Load the data
url = 'https://drive.google.com/file/d/1r29kUG_mtmiQFVuUTJqogNo_l0jNz_iK/view?usp=share_link'
df = pd.read_csv(url)

# Convert the 'Loan_Status' column to label encoding
le = LabelEncoder()
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

# Separate the target variable and predictor variables
y = df['Loan_Status']
X = df.drop(['Loan_Status', 'Loan_ID'], axis=1)

# Convert categorical columns to label encoding
cat_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))

# Impute missing values using KNN imputer
imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Add the target variable back to the imputed dataset
X_imputed['Loan_Status'] = y

# Save the cleaned dataset as a CSV file
X_imputed.to_csv('cleaned_loan_data.csv', index=False)

# Generate a summary using the cleaned data
from pandas_profiling import ProfileReport
# make a report with the clean data
prof = ProfileReport(X_imputed)
prof.to_file(output_file='output.html')
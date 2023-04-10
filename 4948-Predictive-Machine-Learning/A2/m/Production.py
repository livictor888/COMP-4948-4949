import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import pickle
import joblib
def remove_outlier(df, col_name):
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    df_out = df.loc[(df[col_name] > Q1 - 1.5 * IQR) & (df[col_name] < Q3 + 1.5 * IQR)]
    return df_out

def load_saved_model(file_name):
    return load_model(file_name)

# def load_saved_scaler(file_name):
#     with open(file_name, 'rb') as f:
#         return pickle.load(f)

# Load and prepare test data
# test_path = Path("test.csv")
PATH = Path("/Users/mahan/Desktop/Winter2023/Predictive-Machine- 4948/DataSets/car_purchasing.csv")

test_df = pd.read_csv(PATH, encoding="ISO-8859-1", sep=',')
test_df.drop(columns=['car purchase amount'], inplace=True)
test_df = pd.get_dummies(test_df, columns=['gender'])

test_df = remove_outlier(test_df, 'age')
test_df = remove_outlier(test_df, 'credit card debt')
test_df = remove_outlier(test_df, 'net worth')
test_df = remove_outlier(test_df, 'annual Salary')

X_test = test_df[['age', 'annual Salary', 'net worth']]

# Load saved scalers
def load_saved_scaler(file_name):
    return joblib.load(file_name)

# Load saved scalers
scaler_filename = "scaler.pkl"
scaler = load_saved_scaler(scaler_filename)
X_test_scaled = scaler.transform(X_test)

# Load saved base models
base_model_filenames = [
    "base_model_0.pkl",
    "base_model_1.pkl",
    "base_model_2.pkl",
    "base_model_3.pkl",
    "base_model_4.pkl",
    "base_model_5.h5",
    "base_model_6.h5",
    "base_model_7.pkl",
]
base_models = []

for filename in base_model_filenames:
    if filename.endswith(".h5"):
        model = load_saved_model(filename)
    elif filename.endswith(".pkl"):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
    base_models.append(model)
    print("model type:",filename, type(model))
print()

# Get base model predictions
df_base_predictions = pd.DataFrame()

for i, model in enumerate(base_models):
    if hasattr(model, 'predict'):
        predictions = model.predict(X_test_scaled)
    else:
        predictions = model.predict(X_test_scaled)
    df_base_predictions[f"model_{i+1}"] = predictions

# Load the stacked model
stacked_model_filename = "stacked_model.pkl"
with open(stacked_model_filename, 'rb') as f:
    stacked_model = pickle.load(f)

# Make predictions with the stacked model
stacked_predictions = stacked_model.predict(df_base_predictions)
print("Stacked Predictions:")
print(stacked_predictions)

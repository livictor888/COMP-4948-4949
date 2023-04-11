import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from pathlib import Path

# Load and preprocess the dataset
PATH = Path("Tesla.csv")

df = pd.read_csv(PATH)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Create additional features (moving averages)
window_sizes = [3, 7, 15]
for window in window_sizes:
    df[f"rolling_mean_{window}"] = df["Close"].rolling(window=window).mean()
    df[f"rolling_std_{window}"] = df["Close"].rolling(window=window).std()

# Drop rows with NaN values
df.dropna(inplace=True)

# Prepare features and target variable
X = df.drop(columns=["Adj Close"])
y = df["Adj Close"]

# Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model 1: OLS
X_train_OLS = sm.add_constant(X_train)
model_OLS = sm.OLS(y_train, X_train_OLS).fit()
X_test_OLS = sm.add_constant(X_test)
y_pred_OLS = model_OLS.predict(X_test_OLS)

# Model 2: MLPRegressor
model_MLP = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
model_MLP.fit(X_train, y_train)
y_pred_MLP = model_MLP.predict(X_test)

# Model 3: Neural Network with 3 layers
model_NN = Sequential()
model_NN.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
model_NN.add(Dense(32, activation='relu'))
model_NN.add(Dense(1, activation='linear'))
model_NN.compile(optimizer='adam', loss='mean_squared_error')
model_NN.fit(X_train, y_train, batch_size=16, epochs=100, validation_split=0.1)
y_pred_NN = model_NN.predict(X_test)

# Evaluation
models = {
    "OLS": (y_test, y_pred_OLS),
    "MLPRegressor": (y_test, y_pred_MLP),
    "Neural Network": (y_test, y_pred_NN.flatten())
}

for model_name, (y_true, y_pred) in models.items():
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"====Model {model_name} ====")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f'R^2 Score: {r2:.4f}')


""" Modify Example 4 so it can predict if a flower is red or white """

# importing libraries
import numpy as np
import pandas as pd

# ---------------------------- DATA ----------------------------------
# Setup data.

# Load the flower feature data into a DataFrame.
df = pd.DataFrame(columns=['Length', 'Width', 'IsRed'])
data = [
    {'Length': 3, 'Width': 1.5, 'IsRed': 1},
    {'Length': 2, 'Width': 1, 'IsRed': 0},
    {'Length': 4, 'Width': 1.5, 'IsRed': 1},
    {'Length': 3, 'Width': 1, 'IsRed': 0},
    {'Length': 3.5, 'Width': .5, 'IsRed': 1},
    {'Length': 2, 'Width': .5, 'IsRed': 0},
    {'Length': 5.5, 'Width': 1, 'IsRed': 1},
    {'Length': 1, 'Width': 1, 'IsRed': 0},
    {'Length': 4.5, 'Width': 1, 'IsRed': 1}]

df = pd.DataFrame.from_records(data)

y = np.array(df['IsRed'])
X = df.copy()
del X['IsRed']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=0)
# define standard scaler
scaler = StandardScaler()

# transform data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# ---------------------------- DATA ----------------------------------

from keras.layers import Dense
from keras import Sequential

# Build a network model of sequential layers.
model = Sequential()

NUM_COLS = 2
# Add 1st hidden layer. Note 1st hidden layer also receives data.
# The input array must contain two feature columns and any number of rows.
model.add(Dense(10, activation='sigmoid',
                input_shape=(NUM_COLS,)))

# Add 2nd hidden layer.
model.add(Dense(3, activation='sigmoid'))

# Add output layer.
model.add(Dense(1, activation='sigmoid'))

# Compile the model.
# Binary cross entropy is used to measure error cost for binary predictions.
model.compile(loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model.
# An epoch is one iteration for all samples through the network.
# verbose can be set to 1 to show detailed output during training.
model.fit(X_train_scaled, y_train, epochs=2000, verbose=1)

# Evaluate the model.
loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)

# Make predictions.
yhats = model.predict(X_test_scaled)
print("Actual:")
print(y_test)
print("Predicted: ")
print(yhats)
predictions = []

from sklearn.metrics import classification_report


def showClassificationReport(y_test, yhats):
    # Convert continous predictions to
    # 0 or 1.
    for i in range(0, len(yhats)):
        if (yhats[i] > 0.5):
            predictions.append(1)
        else:
            predictions.append(0)
    print(classification_report(y_test, predictions))


showClassificationReport(y_test, yhats)
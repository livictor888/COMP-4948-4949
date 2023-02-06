"""
Starting with the code from Example 11, set the optimal learning rate and modify the code to enable
grid searching for the optimal number of neurons
- examine the optimal number of neurons for the initial hidden layer.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

PATH = "C:\\datasets\\"
FILE = "Social_Network_Ads.csv"
data = pd.read_csv(PATH + FILE)
y = data["Purchased"]
X = data.copy()
del X['User ID']
del X['Purchased']
X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1})

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(data.head())

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Stochastic gradient descent models are sensitive to differences
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_trainScaled = scaler.transform(X_train)
X_testScaled = scaler.transform(X_test)
X_valScaled = scaler.transform(X_val)


def showResults(networkStats):
    dfStats = pd.DataFrame.from_records(networkStats)
    dfStats = dfStats.sort_values(by=['f1'])
    print(dfStats)


def evaluate_model(predictions, y_test):
    # predictions = model.predict(X_test)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print("Precision: " + str(precision) + " " + \
          "Recall: " + str(recall) + " " + \
          "F1: " + str(f1))
    return precision, recall, f1


print("\nLogistic Regression:")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_trainScaled, y_train)
predictions = clf.predict(X_testScaled)
evaluate_model(predictions, y_test)

COLUMN_DIMENSION = 1
# --------------------------------------------------------------
# Part 2
from keras.models import Sequential
from keras.layers import Dense

# shape() obtains rows (dim=0) and columns (dim=1)
n_features = X_trainScaled.shape[COLUMN_DIMENSION]


def getPredictions(model, X_test):
    probabilities = model.predict(X_test)

    predictions = []
    for i in range(len(probabilities)):
        if (probabilities[i][0] > 0.5):
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


# -------------------Model parameters---------------------------

# neuronList = [5, 25, 50, 100, 150]
neuronList = [5, 10, 25, 50, 75, 100, 125, 150]

# --------------------------------------------------------------

# ------------ Build model -------------------------------------
# Build model
import keras
from keras.optimizers import Adam  # for adam optimizer
from keras.optimizers import RMSprop


def create_model(numNeurons):
    model = Sequential()
    model.add(Dense(numNeurons, input_dim=3, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform'))

    # Use Adam optimizer with the given learning rate
    LEARNING_RATE = 0.0100
    # optimizer = Adam(lr=LEARNING_RATE)
    optimizer = RMSprop(learning_rate=LEARNING_RATE)
    # model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


networkStats = []
EPOCHS = 200
NUM_BATCHES = 60

for numNeurons in neuronList:
    model = create_model(numNeurons)
    history = model.fit(X_trainScaled, y_train, epochs=EPOCHS,
                        batch_size=NUM_BATCHES, verbose=1,
                        validation_data=(X_valScaled, y_val))
    predictions = getPredictions(model, X_testScaled)

    # Unfortunate need to format data.
    y_test.to_list()
    predictions = list(predictions)
    precision, recall, f1 = evaluate_model(predictions, y_test)
    networkStats.append({"precision": precision, "recall": recall,
                         "f1": f1, "# neurons": numNeurons})

print(networkStats)
print()
showResults(networkStats)
# --------------------------------------------------------------

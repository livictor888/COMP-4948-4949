import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, \
    accuracy_score, classification_report

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
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print("Precision: " + str(precision) + " " + \
          "Recall: " + str(recall) + " " + \
          "F1: " + str(f1))
    return precision, recall, f1


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
print(f"n_features: {n_features}")


def getPredictions(model, X_test):
    probabilities = model.predict(X_test)

    predictions = []
    for i in range(len(probabilities)):
        if (probabilities[i][0] > 0.5):
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


# --------------------Model parameters-------------------------
# batch_sizes = [10, 60, 100]
# epochList = [100, 200, 300]

# batch_sizes = [50, 100, 150]
# epochList = [100, 300, 500]

# batch_sizes = [10, 100, 200]
# epochList = [100, 300, 500]

# most optimal
# batch_sizes = [50, 100, 150]
# epochList = [200, 300, 400]

# batch_sizes = [50, 100, 150]
# epochList = [250, 300, 350]

# batch_sizes = [50, 100, 150]
# epochList = [250, 275, 300]

# most optimal
# batch_sizes = [75, 100, 125]
# epochList = [200, 300, 400]

# most optimal
batch_sizes = [75, 100, 125]
epochList = [300, 375, 400]

# batch_sizes = [100, 125, 150]
# epochList = [300, 375, 400]
# --------------------------------------------------------------

# --------------------------------------------------------------
# Model building section.
def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=n_features, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


networkStats = []
for batch_size in batch_sizes:
    for epochs in epochList:
        model = create_model()
        history = model.fit(X_trainScaled, y_train, epochs=epochs,
                            batch_size=batch_size, verbose=1,
                            validation_data=(X_valScaled, y_val))
        predictions = getPredictions(model, X_testScaled)

        precision, recall, f1 = evaluate_model(predictions, y_test)
        networkStats.append({"precision": precision, "recall": recall,
                             "f1": f1, "epochs": epochs,
                             "batch": batch_size})
showResults(networkStats)
# --------------------------------------------------------------

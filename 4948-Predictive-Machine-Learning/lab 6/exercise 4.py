from keras.layers import Dense
from sklearn.metrics import accuracy_score
from numpy import argmax
from sklearn.datasets import make_blobs
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

# fit model on dataset
def fitModel(trainX, trainy):
    # define model
    model = Sequential()
    model.add(Dense(15, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # fit model
    model.fit(trainX, trainy, epochs=200, verbose=0)
    return model


from sklearn.preprocessing   import LabelEncoder
import pandas as pd
from keras.utils import to_categorical

def getData():
        PATH = "/Users/pm/Desktop/DayDocs/data/"
        df = pd.read_csv(PATH + 'iris_old.csv')
        df.columns = ['Sepal L', 'Sepal W', 'Petal L', 'Petal W', 'Iris Type']

        # Convert text to numeric category.
        # 0 is setosa, 1 is versacolor and 2 is virginica
        df['y'] = LabelEncoder().fit_transform(df['Iris Type'])

        # Prepare the data.
        X = df[['Sepal L', 'Sepal W', 'Petal L', 'Petal W']]
        y = df['y']
        ROW_DIM = 0
        COL_DIM = 1

        x_array = X.values
        x_arrayReshaped = x_array.reshape(x_array.shape[ROW_DIM],
                                          x_array.shape[COL_DIM])

        y_array = y.values
        y_arrayReshaped = y_array.reshape(y_array.shape[ROW_DIM], 1)

        trainX, testX, trainy, testy = train_test_split(x_arrayReshaped,
                                                        y_arrayReshaped,
                                                        test_size=0.33)
        trainy = to_categorical(trainy)
        return trainX, testX, trainy, testy


def buildAndEvaluateIndividualModels():
    trainX, testX, trainy, testy = getData()
    NUM_MODELS  = 11
    yhats       = []
    scores      = []
    models      = []
    print("\n**** Single model results:")
    for i in range(0, NUM_MODELS):
        model                   = fitModel(trainX, trainy)
        models.append(model)
        predictions             = model.predict(testX)
        yhats.append(predictions)

        # Converts multi-column prediction set back to single column
        # so accuracy score can be calculated.
        singleColumnPredictions = argmax(predictions, axis=1)
        accuracy = accuracy_score(singleColumnPredictions, testy)
        scores.append(accuracy)
        print("Single model " + str(i) + "   accuracy: " + str(accuracy))

    print("Average model accuracy:      " + str(np.mean(scores)))
    print("Accuracy standard deviation: " + str(np.std(scores)))
    return models
models = buildAndEvaluateIndividualModels()






#example 4

# Evaluate ensemble
def buildAndEvaluateEnsemble(models):
    scores = []
    print("\n**** Ensemble model results: ")
    for trial in range(0, 11):
        # Generate new test data.
        _, testX, _, testy = getData()

        yhats  = []
        # Get predictions with pre-built models.
        for model in models:
            predictions = model.predict(testX)
            yhats.append(predictions)

        # Sum predictions for all models.
        # [[0.2, 0.3, 0.5], [0.3, 0.3, 0.4]...], # Model 1 results
        #  [0.3, 0.3, 0.4], [0.1, 0.1, 0.8]...], # Model 2 results
        #  [0.2, 0.2, 0.6], [0.3, 0.3, 0.4]...], # Model 3 results
        # Becomes
        # [[0.7, 0.8, 1.5],[0.7, 0.7, 1.6]...] # Summed results
        summed = np.sum(yhats, axis=0)

        # Converts multi-column prediction set back to single column
        # so accuracy score can be calculated. For example;
        # [[0.7, 0.8, 1.5],[0.7, 0.7, 1.6]...]
        # Becomes
        # [2, 2,....]
        singleColumnPredictions = argmax(summed, axis=1)

        accuracy = accuracy_score(singleColumnPredictions, testy)
        scores.append(accuracy)
        print("Ensemble model accuracy during trial " + str(trial) +\
              ": " + str(accuracy))

    print("Average model accuracy:      " + str(np.mean(scores)))
    print("Accuracy standard deviation: " + str(np.std(scores)))

buildAndEvaluateEnsemble(models)

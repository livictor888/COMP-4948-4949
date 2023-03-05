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
    model.add(Dense(15, input_dim=2, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # fit model
    model.fit(trainX, trainy, epochs=200, verbose=0)
    return model

def getData():
        # generate 2d classification dataset
        X, y = make_blobs(n_samples=500, centers=3, n_features=2,
                          cluster_std=2, random_state=2)
        # split into train and test
        trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.7)

        # Converts array to matrix of categories.
        # [0, 1, 2]

        # Becomes:
        # [[1, 0, 0],
        #  [0, 1, 0],
        #  [0, 0, 1]]
        trainy = tf.keras.utils.to_categorical(trainy)
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

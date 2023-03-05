from keras.models     import Sequential
from keras.layers     import Dense
from os               import makedirs
from os import path
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
from keras.utils import to_categorical
import pandas as pd
import numpy as np

PATH = './models/'

# fit model on dataset
def fit_model(trainX, trainy):
    # define model
    model = Sequential()
    model.add(Dense(25, input_dim=2, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(trainX, trainy, epochs=500, verbose=0)
    return model

def generateData():
    # generate 2d classification dataset
    X, y = make_blobs(n_samples=800, centers=3,
                      n_features=2,
                      cluster_std=2, random_state=2)

    # split into train and test
    trainX, tempX, trainy, tempy = train_test_split(X, y, test_size=0.6)
    testX, valX, testy, valy = train_test_split(tempX, tempy, test_size=0.5)
    return trainX, testX,  valX, trainy, testy, valy

def generateModels(trainX, trainy):
    # create directory for models
    if(not path.exists(PATH)):
        makedirs('./models')

    # fit and save models
    numModels = 5
    print("\nFitting models with training data.")
    for i in range(numModels):
        # fit model
        model = fit_model(trainX, trainy)
        # save model
        filename = PATH + 'model_' + str(i + 1) + '.h5'
        model.save(filename)
        print('>Saved %s' % filename)

trainX, testX,  valX, trainy, testy, valy = generateData()

# one hot encode output variable
trainy = to_categorical(trainy)
generateModels(trainX, trainy)

# load models from file
def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = PATH + 'model_' + str(i + 1) + '.h5'
        # load model from file
        model = load_model(filename)
        # add to list of models
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models

#trainX, testX, trainy, testy = generateData()

# load all models
numModels = 5
models    = load_all_models(numModels)
print('Loaded %d models' % len(models))

print("\nEvaluating single models with validation data.")
# evaluate standalone models on test dataset
# individual ANN models are built with one-hot encoded data.
for model in models:
    oneHotEncodedY = to_categorical(valy)
    _, acc = model.evaluate(valX, oneHotEncodedY, verbose=0)
    print('Model Accuracy: %.3f' % acc)

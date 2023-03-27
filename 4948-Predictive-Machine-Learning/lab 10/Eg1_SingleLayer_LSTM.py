""" Example 1: Single Layer LSTM """

from random import randint
from numpy import array
from numpy import argmax
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

NUM_FEATURES = 10
NUM_SAMPLES = 5
TARGET_INDEX = 2
NUM_WEIGHT_UPDATES = 100


# generate array of 5 numbers like [5, 8, 3, 0, 9].
# each number is >=0 and <10
def generate_sequence():
    return [randint(0, NUM_FEATURES - 1) for _ in range(NUM_SAMPLES)]


# one hot encode sequence
def oneHotEncode(sequence):
    encoding = list()
    # Convert [5, 8, 3, 0, 9]
    # to
    # [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    #  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    #  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    #  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    for value in sequence:
        # Create vector of zeros.
        vector = [0 for _ in range(NUM_FEATURES)]
        vector[value] = 1  # Add 1 to vector.
        encoding.append(vector)
    return array(encoding)


# decode a one hot encoded string
def oneHotDecode(encoded_seq):
    # gets index of element with the maximum value.
    return [argmax(vector) for vector in encoded_seq]


# generate one example for an lstm
def generateSample(targetIndex):
    # generate sequence such as [5, 8, 3, 0, 9]
    sequence = generate_sequence()
    # one hot encode sequence so [5, 8, 3, 0, 9] becomes
    # [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    #  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    #  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    #  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    encoded = oneHotEncode(sequence)
    # reshape sequence to be 3D
    X = encoded.reshape((1, NUM_SAMPLES, NUM_FEATURES))
    # y becomes 1-hot encoded version of 3rd element of [5,8,3,0,9]
    # [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = encoded[targetIndex].reshape(1, NUM_FEATURES)
    return X, y


# define model
model = Sequential()
model.add(LSTM(25, input_shape=(NUM_SAMPLES, NUM_FEATURES)))
model.add(Dense(NUM_FEATURES, activation='softmax'))

# Our output is a one-hot encoded vector so use categorical
# crossentropy.
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()

for i in range(10000):
    trainX, trainy = generateSample(TARGET_INDEX)
    # Update model - weights are updated and are not reset.
    model.fit(trainX, trainy, epochs=1, verbose=2)

# evaluate model
correct = 0
NUM_EVALUATIONS = 100
for i in range(NUM_EVALUATIONS):
    X, y = generateSample(TARGET_INDEX)

    yhat = model.predict(X)
    if oneHotDecode(yhat) == oneHotDecode(y):
        correct += 1
print('Accuracy: %f' % ((correct / NUM_EVALUATIONS)))

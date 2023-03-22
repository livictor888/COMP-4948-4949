""" Example 4: LSTM for Sentiment Analysis """

import pandas as pd
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Embedding, LSTM, Dense

PATH = "C:/PredML/"
FILE = "yelp_mini.csv"
data = pd.read_csv(PATH + FILE)

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Create a sentiment column.
# Ratings above 3 are positive, otherwise they are negative.
data['sentiment'] = ['pos' if (x > 3) else 'neg' for x in data['stars']]
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

from keras.preprocessing.text import Tokenizer

VOCABULARY_SIZE = 2500
tokenizer = Tokenizer(num_words=VOCABULARY_SIZE, lower=True, split=' ')
tokenizer.fit_on_texts(data['text'].values)

X = tokenizer.texts_to_sequences(data['text'].values)
X = tf.keras.utils.pad_sequences(X)
WORDS_PER_SENTENCE = X.shape[0]
NUM_REVIEWS = X.shape[1]

import numpy as np

VOCABULARY_SIZE = np.amax(X) + 1

word_info_sz = 128  # Size of output vector for each word.
# This can be changed.

# Stores info about word sequence -
# "Eat to live" vs. "Live to eat" are very different.
sentence_info_sz = 200  # Vector size for storing info about
# entire sequence.
# This can be changed.
batch_size = 32

model = Sequential()
model.add(Embedding(VOCABULARY_SIZE, word_info_sz))
model.add(LSTM(sentence_info_sz, dropout=0.2))
model.add(Dense(2, activation='softmax'))  # Two column one-hot encoded output.

# Target data is one-hot encoded so we must use ‘categorical_crossentropy’ for loss.
# Here we are using one-hot encoding so we must use categorical_crossentropy.
# One-hot encoding is a fancy way to say multi-column binary encoding.
#  Y_train
# [[0 1]
#  [1 0]
#  [0 1]
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
print(model.summary())

Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.20)

history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=4,
                    verbose=1, validation_data=(X_test, y_test))

score, acc = model.evaluate(X_test, y_test, verbose=2, batch_size=batch_size)
print("Score: %.2f" % (score))
print("Validation Accuracy: %.2f" % (acc))

import matplotlib.pyplot as plt


def showLoss(history):
    # Get training and test loss histories
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    plt.subplot(1, 2, 1)
    # Visualize loss history for training data.
    plt.plot(epoch_count, training_loss, label='Train Loss', color='red')

    # View loss on unseen data.
    plt.plot(epoch_count, validation_loss, 'r--', label='Validation Loss',
             color='black')

    plt.xlabel('Epoch')
    plt.legend(loc="best")
    plt.title("Loss")


def showAccuracy(history):
    # Get training and test loss histories
    training_loss = history.history['accuracy']
    validation_loss = history.history['val_accuracy']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    plt.subplot(1, 2, 2)
    # Visualize loss history for training data.
    plt.plot(epoch_count, training_loss, label='Train Accuracy', color='red')

    # View loss on unseen data.
    plt.plot(epoch_count, validation_loss, 'r--',
             label='Validation Accuracy', color='black')
    plt.xlabel('Epoch')
    plt.legend(loc="best")
    plt.title('Accuracy')


plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
showLoss(history)
showAccuracy(history)
plt.show()

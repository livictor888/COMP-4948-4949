""" Example 2: Predicting Wave Sequences """

from math import sin
from math import pi
from math import exp
from random import randint
from random import uniform
from numpy import array
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


# generate damped sine wave in [0,1]
def getWaveSequence(length, period, decay):
    return [0.5 + 0.5 * sin(2 * pi * i / period) * exp(-decay * i) \
            for i in range(length)]


# generate input and output pairs of damped sine waves
def generateWaveSamples(input_len, n_patterns, output_len):
    X, y = list(), list()
    for _ in range(n_patterns):
        p = randint(10, 20)
        d = uniform(0.01, 0.1)
        wave_sequence = getWaveSequence(input_len + output_len, p, d)

        x_data = wave_sequence[:-output_len]
        X.append(x_data)
        y_data = wave_sequence[-output_len:]
        y.append(y_data)  # Assigns next 5 values in sequence.
    X = array(X).reshape(n_patterns, input_len, 1)
    y = array(y).reshape(n_patterns, output_len)
    return X, y


# configure problem
INPUT_LEN = 50
OUTPUT_LEN = 5

# define model
model = Sequential()
model.add(LSTM(20, input_shape=(INPUT_LEN, 1)))
model.add(Dense(OUTPUT_LEN))
model.compile(loss='mae', optimizer='adam')
model.summary()

# fit model
NUM_SAMPLES = 10000
X, y = generateWaveSamples(INPUT_LEN, NUM_SAMPLES, OUTPUT_LEN)
history = model.fit(X, y, batch_size=10, epochs=1)

# evaluate model
NUM_SAMPLES = 1000
X, y = generateWaveSamples(INPUT_LEN, NUM_SAMPLES, OUTPUT_LEN)
loss = model.evaluate(X, y, verbose=0)
print('Mean squared error: %f' % loss)

print("\n*** Make predictions")
for i in range(0, 5):
    # prediction on new data
    X, y = generateWaveSamples(INPUT_LEN, 1, OUTPUT_LEN)
    yhat = model.predict(X, verbose=0)

    pyplot.title("Y and Yhat")
    pyplot.plot(y[0], label='y')
    pyplot.plot(yhat[0], label='yhat')
    pyplot.legend()
    pyplot.show()

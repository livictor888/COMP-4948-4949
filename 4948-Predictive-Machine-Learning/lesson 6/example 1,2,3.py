
# deeper mlp with tanh for the two circles classification problem
from sklearn.datasets       import make_circles
from sklearn.preprocessing  import MinMaxScaler
from keras.layers           import Dense
from keras.models           import Sequential
from keras.optimizers       import SGD
# from keras.initializers     import RandomUniform
import matplotlib.pyplot    as plt

# Generate 2d classification dataset.
X, y    = make_circles(n_samples=1000, noise=0.1, random_state=1)
scaler  = MinMaxScaler(feature_range=(-1, 1))
X       = scaler.fit_transform(X)

# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]

# # Define the model.
# model = Sequential()
# init = 'he_uniform'
#
# model.add(Dense(5, input_dim=2, activation='sigmoid', kernel_initializer=init))
# model.add(Dense(5, activation='sigmoid', kernel_initializer=init))
# model.add(Dense(5, activation='sigmoid', kernel_initializer=init))
# model.add(Dense(5, activation='sigmoid', kernel_initializer=init))
# model.add(Dense(5, activation='sigmoid', kernel_initializer=init))
# model.add(Dense(1, activation='sigmoid'))


# # Define the model.
# model = Sequential()
# init = 'he_uniform'
# model.add(Dense(5, input_dim=2, activation='relu', kernel_initializer=init))
# model.add(Dense(5, activation='relu', kernel_initializer=init))
# model.add(Dense(5, activation='relu', kernel_initializer=init))
# model.add(Dense(5, activation='relu', kernel_initializer=init))
# model.add(Dense(5, activation='relu', kernel_initializer=init))
# model.add(Dense(1, activation='sigmoid')) # output layer



import tensorflow as tf
leakyReLU = tf.keras.layers.LeakyReLU(alpha=0.3)

# Define the model.
model = Sequential()
model.add(Dense(5, input_dim=2, activation=leakyReLU, kernel_initializer='he_uniform'))
model.add(Dense(5, activation=leakyReLU, kernel_initializer='he_uniform'))
model.add(Dense(5, activation=leakyReLU, kernel_initializer='he_uniform'))
model.add(Dense(5, activation=leakyReLU, kernel_initializer='he_uniform'))
model.add(Dense(5, activation=leakyReLU, kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))



# Compile model.
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=1)

# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train Accuracy: %.3f, Test Accuracy: %.3f' % (train_acc, test_acc))

# Plot loss learning curves.
plt.subplot(211)
plt.title('Loss', pad=-40)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()

# Plot accuracy learning curves.
plt.subplot(212)
plt.title('Accuracy', pad=-40)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()
plt.show()

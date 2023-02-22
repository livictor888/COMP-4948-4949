
# Creating an overfit situation with the moons data set.
from sklearn.datasets       import make_moons
from keras.layers           import Dense
from keras.models           import Sequential
import matplotlib.pyplot    as plt

# Generate 2d classification dataset.
X, y    = make_moons(n_samples=100, noise=0.2, random_state=1)

# Split data into train and test.
n_train = 30
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]

# Define the model.
model = Sequential()
model.add(Dense(500, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])




# # Fit the model.
# history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=4000, verbose=1)




# Fit the model.
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models    import load_model

# simple early stopping
# patience:  # of epochs observed where no improvement before exiting.
# mode:      Could be max, min, or auto.
# min_delta: Amount of change needed to be considered an improvement.
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0.000001, patience=200)
# mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1,
# save_best_only=True)


es = EarlyStopping(monitor='val_loss', mode='max', verbose=1, min_delta=0.000001,
                   patience=200)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='max', verbose=1,
                   save_best_only=True)



# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=4000, verbose=0,
callbacks=[es, mc])

# load the saved model
model = load_model('best_model.h5')





# Evaluate the model.
train_loss, train_acc = model.evaluate(trainX, trainy, verbose=0)
test_loss, test_acc   = model.evaluate(testX, testy, verbose=0)
print('Train accuracy: %.3f, Test accuracy: %.3f' % (train_acc, test_acc))
print('Train loss: %.3f, Test loss: %.3f' % (train_loss, test_loss))

# Plot loss learning curves.
plt.subplot(211)
plt.title('Cross-Entropy Loss', pad=-40)
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



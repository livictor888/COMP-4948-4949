
# from sklearn.datasets   import make_regression
# from keras.layers       import Dense
# from keras.models       import Sequential
# from keras.optimizers   import SGD
# from matplotlib         import pyplot
#
# # Generate regression set.
# X, y = make_regression(n_samples=1000, n_features=20,
#                        noise=0.1, random_state=1)
#
# # Split data into train and test.
# n_train = 500
# trainX, testX = X[:n_train, :], X[n_train:, :]
# trainy, testy = y[:n_train], y[n_train:]
#
# normSizeEvaluations = []
#
# # Define the model.
# model = Sequential()
# model.add(Dense(25, input_dim=20, activation='relu',
#                 kernel_initializer='he_uniform'))
# model.add(Dense(1, activation='linear'))
#
# # # Compile the model.
# # model.compile(loss='mean_squared_error',
# #               optimizer=SGD(lr=0.01, momentum=0.9,  clipvalue=1.05))
#
#
# # Compile the model.
# model.compile(loss='mean_squared_error',
#               optimizer=SGD(lr=0.01, momentum=0.9))
#
# from sklearn.preprocessing  import StandardScaler
#
# # reshape 1d arrays to 2d arrays
# trainy = trainy.reshape(len(trainy), 1)
# testy  = testy.reshape(len(trainy), 1)
#
# # Scale y
# scaler = StandardScaler()
# scaler.fit(trainy)
# trainy = scaler.transform(trainy)
# testy  = scaler.transform(testy)
#
# # Scale x
# xscaler = StandardScaler()
# xscaler.fit(trainX)
# trainX = xscaler.transform(trainX)
# testX  = xscaler.transform(testX)
#
#
# # Fit the model.
# history = model.fit(trainX, trainy,
#                     validation_data=(testX, testy),
#                     epochs=200, verbose=1)
#
# # Evaluate the model.
# train_mse = model.evaluate(trainX, trainy, verbose=0)
# test_mse  = model.evaluate(testX, testy, verbose=0)
# print('Train loss: %.3f, Test loss: %.3f' % (train_mse, test_mse))
#
# normSizeEvaluations.append({'train mse':train_mse,
#                             'test mse':test_mse,
#                             'size':1})
#
# # Plot the loss during training.
# pyplot.title('Mean Squared Error - norm size: ')
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()
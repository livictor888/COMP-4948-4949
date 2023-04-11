"""GRID SEARCH FOR BEST PARAM"""

"""Initial result for me: Best Param: {'num_layers': 2, 'num_neurons': 16, 'activation_func': 'relu', 'learning_rate': 0.01, 'kernel_initializer': 'uniform', 'train_loss': 45131624.0, 'val_loss': 33958648.0}

SHIOULD be able to use either RandomSearchCV() or GridSearchCV() for at least some of these param searches

"""

#
# def grid_search_nn(X_train, y_train, X_val, y_val, hidden_layers, neurons, activations, learning_rates, kernel_initializers):
#     best_loss = float('inf')
#     best_params = None
#     results = []
#     for layer_size in hidden_layers:
#         for num_neurons in neurons:
#             for activation_func in activations:
#                 for learning_rate in learning_rates:
#                     for kernel_initializer in kernel_initializers:
#                         model = Sequential()
#                         model.add(Dense(num_neurons, input_dim=X_train.shape[1], activation=activation_func, kernel_initializer=kernel_initializer))
#                         for i in range(layer_size - 1):
#                             model.add(Dense(num_neurons, activation=activation_func, kernel_initializer=kernel_initializer))
#                         model.add(Dense(1, activation='linear'))
#                         opt = Adam(learning_rate=learning_rate)
#                         model.compile(loss='mse', optimizer=opt, metrics=['mae'])
#                         history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), verbose=0)
#                         train_loss = history.history['loss'][-1]
#                         val_loss = history.history['val_loss'][-1]
#                         results.append({
#                             'num_layers': layer_size,
#                             'num_neurons': num_neurons,
#                             'activation_func': activation_func,
#                             'learning_rate': learning_rate,
#                             'kernel_initializer': kernel_initializer,
#                             'train_loss': train_loss,
#                             'val_loss': val_loss
#                         })
#                         if val_loss < best_loss:
#                             best_loss = val_loss
#                             best_params = {
#                                 'num_layers': layer_size,
#                                 'num_neurons': num_neurons,
#                                 'activation_func': activation_func,
#                                 'learning_rate': learning_rate,
#                                 'kernel_initializer': kernel_initializer,
#                                 'train_loss': train_loss,
#                                 'val_loss': val_loss
#                             }
#     return results, best_params
#
#
# import numpy as np
# from sklearn.model_selection import train_test_split
#
# # Load your dataset and split it into training, validation, and test sets
# X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
#
# # Define the hyperparameters to search over
# hidden_layers = [1, 2, 3]
# neurons = [8, 16, 32]
# activations = ['relu', 'sigmoid']
# learning_rates = [0.001, 0.01, 0.1]
# kernel_initializers = ['uniform', 'normal', 'glorot_uniform']
# import datetime
#
# # Record the start time
# start_time = datetime.datetime.now()
# # Call the grid_search_nn function
# results, bestParam = grid_search_nn(X_train, y_train, X_val, y_val, hidden_layers, neurons, activations, learning_rates, kernel_initializers)
#
# # Print the results
# for r in results:
#     print(r)
# print("Best Param:", bestParam)
# # Record the end time
# end_time = datetime.datetime.now()
#
# # Calculate the elapsed time
# elapsed_time = end_time - start_time
#
# # Print the elapsed time
# print("Elapsed time: ", elapsed_time)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)\
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


#
# import itertools
# import numpy as np
# from tensorflow import keras
#
#
# def grid_search(X_train, y_train, X_val, y_val, X_test, y_test,
#                 layer_sizes=[(10,), (20,), (30,)],
#                 learning_rates=[0.001, 0.01, 0.1],
#                 num_layers=[1, 2, 3],
#                 activations=['relu', 'sigmoid'],
#                 kernel_initializers=['glorot_uniform', 'he_uniform']):
#     # Create a list of all possible parameter combinations
#     param_combinations = list(itertools.product(layer_sizes, learning_rates,
#                                                 num_layers, activations, kernel_initializers))
#
#     # Initialize variables to keep track of results
#     network_stats = []
#     best_rmse = np.inf
#     best_model = None
#
#     # Loop over all parameter combinations and train models
#     for params in param_combinations:
#
#         # Unpack the parameter combination
#         layer_size, learning_rate, num_layers, activation, kernel_initializer = params
#
#         # Create the model with the current parameter combination
#         model = keras.Sequential()
#         model.add(keras.layers.Dense(layer_size[0], input_dim=X_train.shape[1],
#                                      kernel_initializer=kernel_initializer,
#                                      activation=activation))
#         for i in range(num_layers - 1):
#             model.add(keras.layers.Dense(layer_size[0], kernel_initializer=kernel_initializer,
#                                          activation=activation))
#         model.add(keras.layers.Dense(1, kernel_initializer=kernel_initializer))
#
#         # Compile the model with mean squared error loss and the specified learning rate
#         optimizer = keras.optimizers.Adam(lr=learning_rate)
#         model.compile(loss='mean_squared_error', optimizer=optimizer)
#
#         # Train the model
#         history = model.fit(X_train, y_train, epochs=100, batch_size=10,
#                             verbose=0, validation_data=(X_val, y_val))
#
#         # Evaluate the model on the test set and save the RMSE
#         rmse = np.sqrt(model.evaluate(X_test, y_test, verbose=0))
#
#         # Keep track of the best model and its parameters
#         if rmse < best_rmse:
#             best_rmse = rmse
#             best_model = model
#
#         # Save the RMSE and parameters for this model
#         network_stats.append({"rmse": rmse, "layer_size": layer_size,
#                               "learning_rate": learning_rate, "num_layers": num_layers,
#                               "activation": activation, "kernel_initializer": kernel_initializer})
#
#     # Sort the results by RMSE in descending order
#     network_stats = sorted(network_stats, key=lambda x: x['rmse'])
#
#     # Print the results
#     for result in network_stats:
#         print(result)
#
#     # Return the best model and its RMSE
#     print("Best model RMSE:", best_rmse)
#     return best_model


# from sklearn.metrics import classification_report
# def showClassificationReport(y_test, yhats):
#     # Convert continous predictions to
#     # 0 or 1.
#     for i in range(0, len(yhats)):
#         if(yhats[i]>0.5):
#             predictions.append(1)
#         else:
#             predictions.append(0)
#     print(classification_report(y_test, predictions))
# showClassificationReport(y_test, yhats)

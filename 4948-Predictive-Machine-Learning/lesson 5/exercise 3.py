import pandas as pd
import tensorflow
from sklearn.model_selection import train_test_split
import numpy as np

PATH     = "C:\\datasets\\"
CSV_DATA = "housing.data"
df  = pd.read_csv(PATH + CSV_DATA,  header=None)

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

print(df.head())
print(df.tail())
print(df.describe())

dataset = df.values

# split into input (X) and output (Y) variables
X = dataset[:, 0:13]
y = dataset[:, 13]

ROW_DIM = 0
COL_DIM = 1

x_arrayReshaped = X.reshape(X.shape[ROW_DIM],
                            X.shape[COL_DIM])

# Convert DataFrame columns to vertical columns of target variables values.
y_arrayReshaped = y.reshape(y.shape[ROW_DIM], 1)

X_train, X_temp, y_train, y_temp = train_test_split(x_arrayReshaped,
                                                    y_arrayReshaped, test_size=0.3, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp,
                                                y_temp, test_size=0.5, random_state=0)

n_features = X_train.shape[1]

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV


def create_model(numNeurons=50, numHiddenLayers=1, initializer='normal', activation='relu'):
    # create model
    model = Sequential()
    model.add(Dense(25, kernel_initializer='uniform',
                    input_dim=n_features, activation='softplus'))

    for i in range(numHiddenLayers):
        model.add(Dense(numNeurons, kernel_initializer=initializer,
                        activation=activation))

    model.add(Dense(1, kernel_initializer='he_normal',  activation='softplus'))
    opt = tensorflow.keras.optimizers.Adam(learning_rate=0.005)
    # Compile model
    model.compile(loss='mse', optimizer=opt)
    return model

### Grid Building Section #######################
# Define the parameters to try out

params = { 'activation' : ['relu', 'sigmoid'],
           'numNeurons':   [50, 100, 200],
           'numHiddenLayers':[1,2,3],
           'initializer': ['normal']
          }


model = KerasRegressor(build_fn=create_model, epochs=100,
                       batch_size=9, verbose=1)
# grid = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, cv=3)

from sklearn.model_selection     import RandomizedSearchCV
grid = RandomizedSearchCV(model, param_distributions = params, cv = 3)

#################################################

grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))




from sklearn                 import metrics
predictions = grid.best_estimator_.predict(X_test)
print(predictions)
mse         = metrics.mean_squared_error(y_test, predictions)
print("Neural network MSE: " + str(mse))
print("Neural network RMSE: " + str(np.sqrt(mse)))

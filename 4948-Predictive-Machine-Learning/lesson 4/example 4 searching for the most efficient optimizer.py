import pandas                  as pd
import numpy                   as np
from   sklearn.model_selection import train_test_split
from   sklearn.metrics         import mean_squared_error
from keras.models              import Sequential
from keras.layers              import Dense

PATH     = "C:\\datasets\\"
CSV_DATA = "housing.data"
df       = pd.read_csv(PATH + CSV_DATA,  header=None)

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(df.head())
print(df.tail())
print(df.describe())

dataset = df.values

# split into input (X) and output (Y) variables
X = dataset[:,0:13]
y = dataset[:,13]

# Split the data.
X_train, X_temp, y_train, y_temp = train_test_split(X,
         y, test_size=0.3, random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_temp,
         y_temp, test_size=0.5, random_state=0)

def evaluateModel(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print("RMSE: " + str(rmse))
    return rmse

def showResults(networkStats):
    dfStats = pd.DataFrame.from_records(networkStats)
    dfStats = dfStats.sort_values(by=['rmse'])
    print(dfStats)

networkStats = []

### Model parameters ############################
optimizers   = ['SGD', 'RMSprop', 'Adagrad',
        'Adadelta', 'Adam', 'Adamax', 'Nadam']
#################################################

#################################################
# Build model
def create_model(optimizer="SGD"):
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal',
                        activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

for optimizer in optimizers:
        BATCH_SIZE = 10
        EPOCHS     = 100
        model = create_model(optimizer)
        history = model.fit(X_train, y_train, epochs=EPOCHS,
                            batch_size=BATCH_SIZE, verbose=1,
                            validation_data=(X_val, y_val))
        rmse = evaluateModel(model, X_test, y_test)
        networkStats.append({"rmse":rmse, "optimizer":optimizer})
showResults(networkStats)
#################################################



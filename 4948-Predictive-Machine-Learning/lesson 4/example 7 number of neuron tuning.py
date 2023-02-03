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
init_modes  = ['uniform', 'lecun_uniform', 'normal', 'zero',
              'glorot_normal',
              'glorot_uniform', 'he_normal', 'he_uniform']
#################################################


### Model parameters ############################
neuronList = [5, 25, 50, 100, 150]
#################################################

#################################################
# Build model
import keras
from   keras.optimizers import Adam #for adam optimizer
def create_model(numNeurons):

    model = Sequential()
    model.add(Dense(numNeurons,
                    input_dim=13, kernel_initializer='uniform',
                    activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform'))

    # Use Adam optimizer with the given learning rate
    LEARNING_RATE = 0.005
    optimizer = Adam(lr=LEARNING_RATE)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

for numNeurons in neuronList:
        BATCH_SIZE = 10
        EPOCHS     = 100
        model = create_model(numNeurons)
        history = model.fit(X_train, y_train, epochs=EPOCHS,
                            batch_size=BATCH_SIZE, verbose=1,
                            validation_data=(X_val, y_val))
        rmse = evaluateModel(model, X_test, y_test)
        networkStats.append({"rmse":rmse, "# neurons":numNeurons})
showResults(networkStats)
#################################################

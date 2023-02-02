# importing libraries
import numpy as np
import pandas as pd

### DATA           #######################################
# Setup data.
candidates = {'gmat': [780,750,690,710,680,730,690,720,
 740,690,610,690,710,680,770,610,580,650,540,590,620,
 600,550,550,570,670,660,580,650,660,640,620,660,660,
 680,650,670,580,590,690],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,
 3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,
 3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,
 3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,
 1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,
 5,1,2,1,4,5],
              'admitted': [1,1,1,1,1,1,0,1,1,0,0,1,
 1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,
 0,0,1]}

df = pd.DataFrame(candidates,columns= ['gmat', 'gpa',
                                       'work_experience','admitted'])
y = np.array(df['admitted'])
X = df.copy()
del X['admitted']

from   sklearn.model_selection import train_test_split
from sklearn.preprocessing     import StandardScaler

# Split data.
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25,
                                                 random_state=0)
# define standard scaler
scaler = StandardScaler()

# transform data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
##########################################################

from tensorflow.keras        import Sequential
from tensorflow.keras.layers import Dense

# Build a network model of sequential layers.
model = Sequential()

NUM_COLS = 3
# Add 1st hidden layer. Note 1st hidden layer also receives data.
# The input array must contain two feature columns and any number of rows.
model.add(Dense(10, activation='sigmoid',
                input_shape=(NUM_COLS,)))

# Add 2nd hidden layer.
model.add(Dense(3, activation='sigmoid'))

# Add output layer.
model.add(Dense(1,  activation='sigmoid'))

# Compile the model.
# Binary cross entropy is used to measure error cost for binary predictions.
model.compile(loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model.
# An epoch is one iteration for all samples through the network.
# verbose can be set to 1 to show detailed output during training.
model.fit(X_train_scaled, y_train, epochs=2000, verbose=1)

# Evaluate the model.
loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)

# Make predictions.
yhats = model.predict(X_test_scaled)
print("Actual:")
print(y_test)
print("Predicted: ")
print(yhats)
predictions = []

from sklearn.metrics import classification_report
def showClassificationReport(y_test, yhats):
    # Convert continous predictions to
    # 0 or 1.
    for i in range(0, len(yhats)):
        if(yhats[i]>0.5):
            predictions.append(1)
        else:
            predictions.append(0)
    print(classification_report(y_test, predictions))
showClassificationReport(y_test, yhats)

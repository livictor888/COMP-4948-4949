""" Neural Network Applied - No Scaling --> BAD! """

# importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------- DATA -------------------------
# input data
# Setup data.
candidates = {'gmat': [780, 750, 690, 710, 680, 730, 690, 720,
                       740, 690, 610, 690, 710, 680, 770, 610, 580, 650, 540, 590, 620,
                       600, 550, 550, 570, 670, 660, 580, 650, 660, 640, 620, 660, 660,
                       680, 650, 670, 580, 590, 690],
              'gpa': [4, 3.9, 3.3, 3.7, 3.9, 3.7, 2.3, 3.3,
                      3.3, 1.7, 2.7, 3.7, 3.7, 3.3, 3.3, 3, 2.7, 3.7, 2.7, 2.3,
                      3.3, 2, 2.3, 2.7, 3, 3.3, 3.7, 2.3, 3.7, 3.3, 3, 2.7, 4,
                      3.3, 3.3, 2.3, 2.7, 3.3, 1.7, 3.7],
              'work_experience': [3, 4, 3, 5, 4, 6, 1, 4, 5,
                                  1, 3, 5, 6, 4, 3, 1, 4, 6, 2, 3, 2, 1, 4, 1, 2, 6, 4, 2, 6, 5, 1, 2, 4, 6,
                                  5, 1, 2, 1, 4, 5],
              'admitted': [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1,
                           1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0,
                           0, 0, 1]}

df = pd.DataFrame(candidates, columns=['gmat', 'gpa',
                                       'work_experience', 'admitted'])
y = np.array(df['admitted'])
X = df.copy()
del X['admitted']

# Split data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=0)

# define standard scaler
# scaler = StandardScaler()
# # transform data
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


# ----------------------- DATA -------------------------


# defining sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# defining derivative of sigmoid activation function
def sigmoid_derivative(x):
    return x * (1 - x)


# Pass data through layers.
# Each layer sums applied weights and bias, passes to
# an activation function.
def feedForward(X_scaled):
    sum1 = np.dot(X_scaled, weights_0_1) + bias_0_1
    sum1 = np.array(sum1)
    layer_1 = sigmoid(sum1)  # Activation function.
    sum2 = np.dot(layer_1, weights_1_2) + bias_1_2
    layer_2 = sigmoid(sum2)  # Activation function.
    return layer_1, layer_2


# Update weights and bias from front to back.
def backPropogate(weights_0_1, bias_0_1, weights_1_2, bias_1_2):
    yDf = pd.DataFrame(data=y_train, columns=['admitted'])

    # Calculate prediction error.
    layer_2_error = layer_2 - np.array(yDf['admitted']).reshape(-1, 1)
    # Get rate of change of cost function.
    layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)

    # Determine layer 1 error as cost rate of change * layer 2 weights.
    layer_1_error = layer_2_delta.dot(weights_1_2.T)
    # Get rate of change for layer 1.
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

    # Update weights and bias.
    weights_1_2 -= layer_1.T.dot(layer_2_delta) * learning_rate
    weights_0_1 -= layer_0.T.dot(layer_1_delta) * learning_rate

    bias_1_2 -= np.sum(layer_2_delta, axis=0, keepdims=True) * learning_rate
    bias_0_1 -= np.sum(layer_1_delta, axis=0, keepdims=True) * learning_rate
    return weights_0_1, bias_0_1, weights_1_2, bias_1_2


# defining learning rate
learning_rate = 0.1

# These weights would normally be generated randomly
# or with kernel initializers.
weights_0_1 = np.array([[0.10473281, 0.23991864, 0.51106061, 0.97739018],
                        [0.46591006, 0.54318817, 0.58782883, 0.68117129],
                        [0.0502301, 0.22142866, 0.86126238, 0.72482657]])
bias_0_1 = np.array([[0.15025418, 0.73481849, 0.90219478, 0.30605943]])
weights_1_2 = np.array([[0.62996657], [0.25984049], [0.72180012], [0.81730325]])
bias_1_2 = np.array([[0.09842751]])

# training loop
EPOCHS = 10000
for i in range(EPOCHS):
    layer_0 = X_train
    # Feed data forward.
    layer_1, layer_2 = feedForward(X_train)

    # Back propagate to update weights and bias.
    weights_0_1, bias_0_1, weights_1_2, bias_1_2 = \
        backPropogate(weights_0_1, bias_0_1, weights_1_2, bias_1_2)

    # print(f"\nEpoch #{i}:")
    # print(layer_2)

# printing output
print('\nActual data:')
print(y_train)
print('\nANN predictions:')
print(layer_2)
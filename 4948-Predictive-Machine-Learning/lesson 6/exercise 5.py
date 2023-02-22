# mlp with unscaled data for the regression problem
from sklearn.datasets import make_regression
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt

# Generate the regression dataset.
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)

plt.hist(y)
plt.title("Unscaled Input")
plt.show()

# Split into train and test.
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]

clipResults = []


def buildModel(clipSize):
    # Define the model.
    model = Sequential()
    model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='linear'))

    # # Compile the model.
    # opt = SGD(lr=0.01, momentum=0.9, clipnorm=clipSize)
    # model.compile(loss='mean_squared_error', optimizer=opt)

    opt = SGD(lr=0.01, momentum=0.9, clipvalue=clipSize)
    model.compile(loss='mean_squared_error', optimizer=opt)

    # Fit the model.
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=1)

    # Evaluate the model.
    train_mse = model.evaluate(trainX, trainy, verbose=0)
    test_mse = model.evaluate(testX, testy, verbose=0)
    print('Train MSE: %.3f, Test MSE: %.3f' % (train_mse, test_mse))
    clipResults.append({'train mse': train_mse, 'test mse': test_mse,
                        'clip size': clipSize})
    # Plot losses during training.
    plt.title('Losses')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


clipSizes = [0.1, 0.5, 1, 5, 10]
for i in range(0, len(clipSizes)):
    buildModel(clipSizes[i])

for clipResult in clipResults:
    print(clipResult)

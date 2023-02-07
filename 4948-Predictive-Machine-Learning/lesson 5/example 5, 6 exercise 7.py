
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
plt.style.use('ggplot')


import matplotlib.pyplot as plt

# Create numeric target for iris type.
dataset = pd.read_csv('C:\\datasets\\iris_v2.csv')
dataset.iris_type = pd.Categorical(dataset.iris_type)


# Prepare x and y.
dataset['flowertype'] = dataset.iris_type.cat.codes
del dataset['iris_type']
y = dataset['flowertype']
X = dataset
del X['flowertype']

# Split X and y.
X_train, X_test, y_train, y_test = train_test_split(X,
                                    y, test_size=0.30)
# Scale X and Y.
scX=StandardScaler()
scalerX = scX.fit(X_train)
trainX_scaled = scalerX.transform(X_train)
testX_scaled  = scalerX.transform(X_test)

# Create and fit model.
model = MLPClassifier()
model.fit(trainX_scaled, y_train)
print(model.get_params()) # Show model parameters.


# Evaluate model.
predicted_y = model.predict(testX_scaled)
print(metrics.classification_report(y_test, predicted_y))
print(metrics.confusion_matrix(y_test, predicted_y))

def showLosses(model):
    plt.plot(model.loss_curve_)
    plt.title("Loss Curve")
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
showLosses(model)

parameters={
'solver': ['adam', 'sgd'],
'learning_rate': ['constant', 'adaptive', 'invscaling'],
'hidden_layer_sizes': [(200,200), (300,200), (150,150)],
'activation': ["logistic", "relu", "tanh"]
}
model2 = GridSearchCV(estimator=model, param_grid=parameters,
                            scoring='accuracy', # average='macro'),
                            n_jobs=-1, cv=4, verbose=1,
                            return_train_score=False)

model2.fit(trainX_scaled, y_train)
print("Best parameters: ")
print(model2.best_params_)
y_pred = model2.predict(testX_scaled)

print("Report with grid: ")
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
showLosses(model2.best_estimator_)
# import necessary libraries
import matplotlib.pylab as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.exceptions import DataConversionWarning
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import RidgeClassifier
import statsmodels.api       as sm
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
from keras.optimizers import Adam, SGD
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from   sklearn               import metrics
from imblearn.over_sampling import SMOTE
import warnings
from sklearn.model_selection import KFold
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DataConversionWarning)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from mlxtend.classifier import EnsembleVoteClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

# from pandas_profiling import ProfileReport
# "C:\Users\mahan\Desktop\Winter2023\Predictive-Machine- 4948\DataSets\car_purchasing.csv"
################################# DATA LOADING & CLEANING #################################
PATH = Path("/Users/mahan/Desktop/Winter2023/Predictive-Machine- 4948/DataSets/car_purchasing.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
df = pd.read_csv(PATH, encoding="ISO-8859-1", sep=',')
df = df.drop_duplicates()
df.drop(columns=['customer name','customer e-mail','country'],inplace=True)
df = pd.get_dummies(df, columns=['gender'])
print("Null", df.isnull().sum())
print(df.info())
print(df.head())
sns.pairplot(df)
plt.show()
sns.heatmap(df.corr(), annot= True)
plt.show()
fig,axes = plt.subplots(2,2,figsize=(10,10))
cols =[ 'credit card debt','age', 'annual Salary', 'net worth']
for i in range(4):
    sns.scatterplot(x='car purchase amount', y=cols[i], data=df, ax=axes[i%2,i//2])
plt.show()


num_cols=['age','annual Salary','credit card debt','net worth']
plt.figure(figsize=(15,15))
for i in enumerate(num_cols):
    plt.subplot(2,2,i[0]+1)
    sns.scatterplot(x=df[i[1]],y=df['car purchase amount'])
plt.figure(figsize=(15,15))
plt.show()
for i in enumerate(num_cols):
    plt.subplot(2,2,i[0]+1)
    sns.boxplot(df[i[1]])
plt.show()



def remove_outlier(df,col_name):
    Q1=df[col_name].quantile(0.25)
    Q3=df[col_name].quantile(0.75)
    IQR=Q3-Q1
    df_out=df.loc[(df[col_name]>Q1-1.5*IQR)&(df[col_name]<Q3+1.5*IQR)]
    return df_out
df=remove_outlier(df,'age')
df=remove_outlier(df,'credit card debt')
df=remove_outlier(df,'net worth')
df=remove_outlier(df,'annual Salary')

results = {}


# Split data into train and test sets
X = df.drop('car purchase amount', axis=1)
columns = X.columns
print(columns)
y = df['car purchase amount']
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X=scaler.fit_transform(X)
# X = sm.add_constant(X) # double check this is needed
y=scaler.fit_transform(y.values.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


################################## 3 FEATURE SELECTION TECHNIQUES #################################


#### Random Forest Feature Importance ####

# Use the random grid to search for best hyperparameters
rf = RandomForestRegressor()
random_grid = \
    {'bootstrap': [True],
     'max_depth': [4, 6, None],
     'max_features': ['auto'],
     'min_samples_leaf': [15],
     'min_samples_split': [15],
     'n_estimators': [400, 800, 1600]}

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, n_jobs=-1)
# Fit the random search model
rf_random.fit(X_train, y_train)
print("Best parameters to use for random forest")
print(rf_random.best_params_)
# Best parameters to use for random forest{'n_estimators': 1600, 'min_samples_split': 15, 'min_samples_leaf': 15, 'max_features': 'auto', 'max_depth': 6, 'bootstrap': True}
print("----------------------------------------")


# We create the new rf with the best random forest params suggested above
rf = RandomForestRegressor(n_estimators=1600, min_samples_split=15, min_samples_leaf=15, max_features='auto',
                           max_depth=6, bootstrap=True)
rf.fit(X_train, y_train)

# Find feature importance's
importances = list(rf.feature_importances_)

# Present features and importance scores.
def showFeatureImportances(importances, feature_list):
    dfImportance = pd.DataFrame()
    for i in range(0, len(importances)):
        dfImportance = dfImportance.append({"importance": importances[i],
                                            "feature": feature_list[i]},
                                           ignore_index=True)

    dfImportance = dfImportance.sort_values(by=['importance'],
                                            ascending=False)
    print(dfImportance)


showFeatureImportances(importances, columns)
"""   importance           feature
0    0.405492               age
1    0.393707     annual Salary
3    0.200542         net worth
2    0.000147  credit card debt
5    0.000062          gender_1
4    0.000049          gender_0"""

#### RFE  ####
from sklearn.linear_model import LinearRegression
# Create a logistic regression estimator
estimator =LinearRegression()

# Create the RFE object and specify the number of features to select
rfe = RFE(estimator, n_features_to_select=3, step=1)

# Fit the RFE object to the data
rfe.fit(X, y)

# Print the selected features
print("RFE Selected Features:")
for i, col in enumerate(columns):
    if rfe.support_[i]:
        print(col)



#### FFS  ####


# Initialize a logistic regression model. performs forward feature selection on the lung cancer
logreg = LinearRegression()

# Perform forward feature selection with 5-fold cross-validation.
sfs = SequentialFeatureSelector(logreg, direction="forward", n_features_to_select=3, cv=5)
sfs.fit(X_train, y_train)

# Print the selected feature indices and names.
print("FFS Selected Features:", list(df.drop(columns=['car purchase amount']).columns[sfs.get_support()]))

#
# ################################## MODELS #################################
#

# Split data into train and test sets
X = df[['age', 'annual Salary', 'net worth' ]]
y = df['car purchase amount']
X = sm.add_constant(X) # double check this is needed

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# X=scaler.fit_transform(X)
# y=scaler.fit_transform(y.values.reshape(-1,1))

# Make predictions and evaluate with the RMSE.
model       = sm.OLS(y_train, X_train_scaled).fit()
predictions = model.predict(X_test_scaled)

results['Model 1 OLS'] = {
    'R-squared': r2_score(y_test, predictions),
    'MSE': mean_squared_error(y_test, predictions),
    'MAE': mean_absolute_error(y_test, predictions)
}
print(model.summary())
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))



# Split data into train and test sets
X = df[['age', 'annual Salary', 'net worth' ]]
y = df['car purchase amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from    keras.models import Sequential
from    keras.layers import Dense
model=Sequential()
model.add(Dense(10,activation='relu',input_dim=3))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='linear'))

print(model.summary())
model.compile(optimizer='adam',loss='mean_squared_error')#,metrics=['mean_absolute_error'])
history=model.fit(X_train,y_train,batch_size=16, epochs=50,validation_split=0.2)
y_pred=model.predict(X_test)

results['Model 2 adam opt'] = {
    'R-squared': r2_score(y_test, y_pred),
    'MSE': mean_squared_error(y_test, y_pred),
    'MAE': mean_absolute_error(y_test, y_pred)
}

from sklearn.metrics import r2_score
ss = r2_score(y_test, y_pred)
print("r2", ss)
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss During Training or Validation')
plt.ylabel('Training & Validation Losses')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

loss = model.evaluate(X_test, y_test)
print("Multi-Layer Perceptron (MLP) loss: ", loss)



"""GRID SEARCH FOR BEST PARAM"""

"""Initial result for me: Best Param: {'num_layers': 2, 'num_neurons': 16, 'activation_func': 'relu', 'learning_rate': 0.01, 'kernel_initializer': 'uniform', 'train_loss': 45131624.0, 'val_loss': 33958648.0}
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

# Split the data.
X_train, X_temp, y_train, y_temp = train_test_split(X,
         y, test_size=0.3, random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_temp,
         y_temp, test_size=0.5, random_state=0)


# set best parameters
num_layers = 2
num_neurons = 16
activation_func = 'relu'
learning_rate = 0.01
kernel_initializer = 'uniform'

# create model with best parameters
model = Sequential()
model.add(Dense(num_neurons, input_dim=X_train.shape[1], activation=activation_func, kernel_initializer=kernel_initializer))
for i in range(num_layers - 1):
    model.add(Dense(num_neurons, activation=activation_func, kernel_initializer=kernel_initializer))
model.add(Dense(1, activation='linear'))
opt = Adam(learning_rate=learning_rate)
model.compile(loss='mse', optimizer=opt, metrics=['mae'])
from   sklearn.metrics         import mean_squared_error

# Fit the model.
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models    import load_model

# simple early stopping
# patience:  # of epochs observed where no improvement before exiting.
# mode:      Could be max, min, or auto.
# min_delta: Amount of change needed to be considered an improvement.
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0.000001, patience=200)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1,
save_best_only=True)

# fit model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4000, verbose=0,
callbacks=[es, mc])

# load the saved model
model = load_model('best_model.h5')


# train model on full training set
# history = model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_data=(X_val, y_val))
predictions = model.predict(X_test)

results['Model 3 gridchosen param'] = {
    'R-squared': r2_score(y_test, predictions),
    'MSE': mean_squared_error(y_test, predictions),
    'MAE': mean_absolute_error(y_test, predictions)
}

print("RMSE: " + str(np.sqrt(mean_squared_error(y_test, predictions))))# evaluate model on validation set
val_loss = model.evaluate(X_val, y_val)[0]
print('Validation loss:', val_loss)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss During Training or Validation (best param)')
plt.ylabel('Training & Validation Losses')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()


# Print results
for model, metrics in results.items():
    print(model)
    print(metrics)
    print(f"R-squared: {metrics['R-squared']:.2f}")
    print(f"MSE: {metrics['MSE']:.2f}")
    print(f"MAE: {metrics['MAE']:.2f}")
    print()

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


from sklearn.linear_model    import LinearRegression
from sklearn.linear_model    import ElasticNet
from sklearn.tree            import DecisionTreeRegressor
from sklearn.svm             import SVR
from sklearn.ensemble        import AdaBoostRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.ensemble        import ExtraTreesRegressor
from sklearn.metrics         import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy  as np
import pandas as pd


def getUnfitModels():
    models = list()
    models.append(ElasticNet())
    models.append(SVR(gamma='scale'))
    models.append(DecisionTreeRegressor())
    models.append(AdaBoostRegressor())
    models.append(RandomForestRegressor(n_estimators=200))
    models.append(ExtraTreesRegressor(n_estimators=200))
    return models

def evaluateModel(y_test, predictions, model):
    mse = mean_squared_error(y_test, predictions)
    rmse = round(np.sqrt(mse),3)
    print(" RMSE:" + str(rmse) + " " + model.__class__.__name__)

def fitBaseModels(X_train, y_train, X_test, models):
    dfPredictions = pd.DataFrame()

    # Fit base model and store its predictions in dataframe.
    for i in range(0, len(models)):
        models[i].fit(X_train, y_train)
        predictions = models[i].predict(X_test)
        colName = str(i)
        # Add base model predictions to column of data frame.
        dfPredictions[colName] = predictions
    return dfPredictions, models

def fitStackedModel(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Split data into train, test and validation sets.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.70)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50)

# Get base models.
unfitModels = getUnfitModels()

# Fit base and stacked models.
dfPredictions, models = fitBaseModels(X_train, y_train, X_val, unfitModels)
stackedModel          = fitStackedModel(dfPredictions, y_val)

# Evaluate base models with validation data.
print("\n** Evaluate Base Models **")
dfValidationPredictions = pd.DataFrame()
for i in range(0, len(models)):
    predictions = models[i].predict(X_test)
    colName = str(i)
    dfValidationPredictions[colName] = predictions
    evaluateModel(y_test, predictions, models[i])

# Evaluate stacked model with validation data.
stackedPredictions = stackedModel.predict(dfValidationPredictions)
print("\n** Evaluate Stacked Model **")
evaluateModel(y_test, stackedPredictions, stackedModel)



# Create and fit model.
model = MLPRegressor()
model.fit(X_train, y_train)
print(model.get_params()) # Show model parameters.

from sklearn import metrics
# Evaluate model.
predicted_y = model.predict(X_test)
# print(metrics.classification_report(y_test, predicted_y))
mse = mean_squared_error(y_test, predicted_y)
rmse = round(np.sqrt(mse), 3)
print(" RMSE:" + str(rmse))
# val_loss = model.evaluate(X_val, y_val)[0]
# print('Validation loss:', val_loss)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss During Training or Validation (best param)')
plt.ylabel('Training & Validation Losses')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

# print(metrics.confusion_matrix(y_test, predicted_y))

def showLosses(model):
    plt.plot(model.loss_curve_)
    plt.title("Loss Curve")
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
showLosses(model)



# #### Logistic Regression & Bagged Decision Trees Ridge Models ####
#
# models = {
#     'Linear Regression': LinearRegression(),
#     'Bagged Decision Trees Ridge': BaggingClassifier(RidgeClassifier(), max_samples=0.4, max_features=3,
#                                                      n_estimators=100),
#     # 'SGCClassifier': SGDClassifier(random_state=42),
#     # 'Decision Tree': DecisionTreeClassifier(random_state=42),
#     # 'Random Forest': RandomForestClassifier(random_state=42),
#     # 'Gradient Boosting': GradientBoostingClassifier(random_state=42),
#     # 'Bagged Decision Trees': BaggingClassifier(random_state=42),
#     # 'SVM': SVC(random_state=42),
# }
#
# score_dict = {}
# for name, model in models.items():
#     print("Model:", name)
#     print("---------------")
#
#     # Cross-validation
#     kfold = KFold(n_splits=3)
#     r2 = cross_val_score(model, X_train, y_train, cv=kfold, scoring="r2")
#
#     f1 = cross_val_score(model, X_train, y_train, cv=kfold, scoring="mean_squared_error")
#     accuracy = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
#     precision = cross_val_score(model, X_train, y_train, cv=kfold, scoring="average_precision")
#     print("Mean F1 score:", np.mean(f1))
#     print("Mean accu score:", np.mean(accuracy))
#     print("Mean precision score:", np.mean(precision))
#     score_dict[name] = ("F1:", np.mean(f1), "Accuracy:", np.mean(accuracy), "Precision:", np.mean(precision))
#
# print(score_dict)


#
# ########## This section is used to find the best classifier (in our case RidgeClassifier() ) for the BaggingClassifier model ########
# # Create classifiers
# knn = KNeighborsClassifier()
# svc = SVC()
# rg = RidgeClassifier()
#
# # Build array of classifiers.
# classifierArray = [knn, svc, rg]
#
#
# def showStats(classifier, scores):
#     print(classifier + ":    ", end="")
#     strMean = str(round(scores.mean(), 2))
#
#     strStd = str(round(scores.std(), 2))
#     print("Mean: " + strMean + "   ", end="")
#     print("Std: " + strStd)
#
#
# def evaluateModel(model, X_test, y_test, title):
#     print("\n*** " + title + " ***")
#     predictions = model.predict(X_test)
#     report = classification_report(y_test, predictions)
#     print(report)
#
#
# # Search for the best classifier
# print("Search for the best classifier:")
# for clf in classifierArray:
#     modelType = clf.__class__.__name__
#
#     # Create and evaluate stand-alone model.
#     # Create and evaluate stand-alone model.
#     clfModel = clf.fit(X_train, y_train)
#     evaluateModel(clfModel, X_test, y_test, modelType)
#
#     # max_features means the maximum number of features to draw from X.
#     # max_samples sets the percentage of available data used for fitting.
#     bagging_clf = BaggingClassifier(clf, max_samples=0.4, max_features=3,
#                                     n_estimators=100)
#     baggedModel = bagging_clf.fit(X_train, y_train)
#     evaluateModel(baggedModel, X_test, y_test, "Bagged: " + modelType)
#
#
#



#
# #### Stacked  Model ####
#
# print("Stacked Model")
# def getUnfitModels():
#     models = list()
#     models.append(LinearRegression())
#     models.append(RandomForestRegressor(n_estimators=10))
#     models.append(KNeighborsRegressor(n_neighbors=5))
#     return models
#
#
# def evaluateModel(y_test, predictions, model):
#     print("\n*** " + model.__class__.__name__)
#     mse = mean_squared_error(y_test, predictions)
#     r2 = r2_score(y_test, predictions)
#     print("MSE:", mse)
#     print("R2 score:", r2)
#
#
# def fitBaseModels(X_train, y_train, X_test, models):
#     dfPredictions = pd.DataFrame()
#     clf = LinearRegression()
#     # Fit base model and store its predictions in dataframe.
#     for i in range(0, len(models)):
#         kfold = KFold(n_splits=10)
#         mse = -cross_val_score(clf, X_train, y_train, cv=kfold, scoring="neg_mean_squared_error")
#         r2 = cross_val_score(clf, X_train, y_train, cv=kfold, scoring="r2")
#         print("Mean MSE score:", np.mean(mse))
#         print("Mean R2 score:", np.mean(r2))
#         models[i].fit(X_train, y_train)
#         predictions = models[i].predict(X_test)
#         colName = str(i)
#         dfPredictions[colName] = predictions
#     return dfPredictions, models
#
#
# def fitStackedModel(X, y):
#     model = LinearRegression()
#     model.fit(X, y)
#     return model
#
#
# # Split data into train, test and validation sets.
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.70)
# X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50)
#
# # Get base models.
# unfitModels = getUnfitModels()
#
# # Fit base and stacked models.
# dfPredictions, models = fitBaseModels(X_train, y_train, X_val, unfitModels)
# stackedModel = fitStackedModel(dfPredictions, y_val)
#
# # Evaluate base models with validation data.
# print("\n** Evaluate Base Models **")
# dfValidationPredictions = pd.DataFrame()
# for i in range(0, len(models)):
#     predictions = models[i].predict(X_test)
#     colName = str(i)
#     dfValidationPredictions[colName] = predictions
#
# # Evaluate stacked model with validation data.
# stackedPredictions = stackedModel.predict(dfValidationPredictions)
# print("\n** Evaluate Stacked Model **")
# evaluateModel(y_test, stackedPredictions, stackedModel)


# from sklearn.datasets             import make_regression
# from sklearn.pipeline import Pipeline
# from   skorch                     import NeuralNetRegressor
# import torch.nn as nn
# import numpy    as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from   sklearn.metrics          import mean_squared_error
# import torch.nn.functional as   F



# from torch import optim
#
# class MyNeuralNet(nn.Module):
#     def __init__(self, num_neurons):
#         super(MyNeuralNet, self).__init__()
#
#         self.num_units = num_neurons
#         self.dense0 = nn.Linear(4, num_neurons)
#         self.dense1 = nn.Linear(num_neurons, 10)
#         self.output = nn.Linear(10, 1)
#
#     def forward(self, X, **kwargs):
#         X = F.relu(self.dense0(X))
#         X = F.relu(self.dense1(X))
#         X = self.output(X)
#         return X
#
# def buildModel(x, y):
#     nn = NeuralNetRegressor(MyNeuralNet, verbose=1, train_split=False)
#     # Trains the Neural Network with fixed hyperparameters
#     pipeline = Pipeline([ ('nn', nn)])
#
#     params = {
#       'nn__max_epochs': [30,50,60],
#       'nn__lr': [0.01, 0.015, 0.007],
#       'nn__module__num_neurons': [15,20,25],
#       'nn__optimizer': [optim.Adam, optim.SGD, optim.RMSprop]}
#
#     # The grid search module is instantiated
#     gs = GridSearchCV(pipeline, param_grid=params, refit=True, cv=3,
#                     scoring='neg_mean_squared_error', verbose=1)
#
#     return gs.fit(x, y)
#
# def evaluateModel(model, X_test, y_test, scalerY):
#     print(model)
#     y_pred_scaled = model.predict(X_test)
#     y_pred = scalerY.inverse_transform(y_pred_scaled)
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     print("RMSE: " + str(rmse))
#
# # Prep the data.
# # This is a toy dataset for regression, 1000 data points with 20 features each
# import torch
# import pandas as pd
#
#
# X_train, X_test, y_train, y_test =\
#     train_test_split(X, y, test_size=0.2)
#
# from sklearn.preprocessing import StandardScaler
# scalerX      = StandardScaler()
# scaledXTrain = scalerX.fit_transform(X_train)
# scaledXTest  = scalerX.transform(X_test)
#
# scalerY      = StandardScaler()
# scaledYTrain = scalerY.fit_transform(np.array(y_train).reshape(-1,1))
#
# # Must convert the data to PyTorch tensors
# X_train_tensor = torch.tensor(scaledXTrain, dtype=torch.float32)
# X_test_tensor  = torch.tensor(scaledXTest, dtype=torch.float32)
# y_train_tensor = torch.tensor(list(scaledYTrain), dtype=torch.float32)
# y_test_tensor = torch.tensor(list(y_test), dtype=torch.float32)
#
# # Build the model.
# model   = buildModel(X_train_tensor, y_train_tensor)
#
# print("Best parameters:")
# print(model.best_params_)
#
# # Evaluate the model.
# evaluateModel(model.best_estimator_, X_test_tensor, y_test_tensor, scalerY)



# train = df.sample(frac=0.95, random_state=786)
# test  = df.drop(train.index)
# train.reset_index(inplace=True, drop=True)
# test.reset_index(inplace=True, drop=True)
# print('Data for Modeling: ' + str(train.shape))
# print('Unseen Data For Predictions: ' + str(test.shape))
#
# # show data types.
# from pycaret.classification import *
# exp_clf101 = setup(data = train, target = 'default', session_id=123)
# print(exp_clf101)
#
# print("*** Showing best model")
# best_model = compare_models()
# print(best_model)
#
# # Need this step for PyCharm
# print("*** Showing best model")
# regression_results = pull()
# print(regression_results)




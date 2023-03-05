from sklearn.feature_selection import RFE
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.exceptions import FitFailedWarning
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from mlxtend.classifier import EnsembleVoteClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore")

""" Data Cleaning """

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)

# Load the dataset into a pandas DataFrame object
PATH = "C:/datasets/"
CSV_DATA = "Paitients_Files_Train.csv"
df = pd.read_csv(PATH + CSV_DATA, sep=',')

print("\n---------- Raw data ----------\n")
print(df.head(10))  # View a snapshot of the data.
print()
print(df.describe().T)  # View stats including counts which highlight missing values.

# Split the dataset into a target variable and predictor variables
y = df['Sepsis']
X = df.drop(['ID', 'Sepsis'], axis=1)

# Create a KNN imputer object with n_neighbors=5
imputer = KNNImputer(n_neighbors=5)

# Impute the missing values in the predictor variables
X_imputed = imputer.fit_transform(X)

# Convert the imputed predictor variables back into a pandas DataFrame object
X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)

# Create dummy variables for the 'Sepsis' column
y_dummies = pd.get_dummies(y, prefix='Sepsis').drop(['Sepsis_Negative'], axis=1)

# Combine the imputed predictor variables and dummy target variable into a new DataFrame
df = pd.concat([X_imputed_df, y_dummies], axis=1)

print("\n---------- Cleaned up data ----------\n")
print(df.head(10))
print()
print(df.describe().T)

""" Feature Selection - Recursive Feature Elimination (RFE) """

# impute missing values
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# define the target variable and predictors
target = 'Sepsis_Positive'
predictors = [col for col in df_imputed.columns if col not in [target, 'ID']]

# create the logistic regression model with 'l2' penalty
lr_model = LogisticRegression(max_iter=10000000, random_state=42)

# create the RFE object with 10 features to select
rfe = RFE(estimator=lr_model, n_features_to_select=5)

# fit the RFE object on the imputed dataset
rfe.fit(df_imputed[predictors], df_imputed[target])

# print the selected features
selected_features = [predictors[i] for i in range(len(predictors)) if rfe.support_[i]]
print("\n\n*** Recursive Feature Elimination")
print('\nSelected features:', selected_features)

""" Feature Selection - Forward Feature Selection (FFS) """

# impute missing values
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# split data into features and target variable
X = df_imputed.drop(['Sepsis_Positive'], axis=1)
y = df_imputed['Sepsis_Positive']

# Perform logistic regression.
lr_model = logisticModel = LogisticRegression(max_iter=10000000, random_state=42)

# forward feature selection with f_regression
# f_regression returns F statistic for each feature.
ffs = f_regression(X, y)

featuresDf = pd.DataFrame()
for i in range(0, len(X.columns)):
    featuresDf = featuresDf.append({"feature": X.columns[i],
                                    "ffs": ffs[0][i]}, ignore_index=True)
featuresDf = featuresDf.sort_values(by=['ffs'], ascending=False)
print("\n\n*** Forward Feature Selection")
print("\nSignificant features in descending F-statistic values:")
print(featuresDf)

""" Feature Selection - Random Forest """

# Split the dataset into a target variable and predictor variables
y = df['Sepsis_Positive']
X = df.drop(['Sepsis_Positive'], axis=1)
feature_list = ['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age', 'Insurance']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7)

# Use the random grid to search for best hyperparameters
rf = RandomForestRegressor()
random_grid = \
    {'bootstrap': [True],
     'max_depth': [4, 6, None],
     'max_features': [1.0],
     'min_samples_leaf': [15],
     'min_samples_split': [15],
     'n_estimators': [400, 800, 1600]}

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=9, cv=3, n_jobs=-1)
# Fit the random search model
rf_random.fit(X_train, y_train)
print("\n\n*** Random Forest Feature Selection")
print("\nBest parameters to use for random forest")
print(rf_random.best_params_)
print("----------------------------------------")

# We create the new rf with the best random forest params suggested above
rf = RandomForestRegressor(n_estimators=800, min_samples_split=15, min_samples_leaf=15, max_features='auto',
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


showFeatureImportances(importances, X_train.columns)

""" Logistic Regression with Cross Fold Validation """

# Split the dataset into a target variable and predictor variables
y = df['Sepsis_Positive']

# chosen variables based on feature selection
X = df[['PL', 'M11', 'BD2', 'PRG', 'Age']]

# Create a KNN imputer object with n_neighbors=5
imputer = KNNImputer(n_neighbors=5)

# Impute the missing values in the predictor variables
X_imputed = imputer.fit_transform(X)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

print("\n\n\n----- Logistic Regression with Cross Fold Validation and Scaling -----")

# prepare cross validation with three folds and 1 as a random seed.
kfold = KFold(n_splits=3, shuffle=True)

accuracyList = []
precisionList = []
recallList = []
f1List = []

foldCount = 0

for train_index, test_index in kfold.split(df):
    # use index lists to isolate rows for train and test sets.
    # Get rows filtered by index and all columns.
    # X.loc[row number array, all columns]
    X_train = X.iloc[train_index, :]
    X_test = X.iloc[test_index, :]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    # Perform logistic regression.
    logisticModel = LogisticRegression(fit_intercept=True, solver='lbfgs', penalty='l2')
    # Fit the model.
    logisticModel.fit(X_train, y_train.values.ravel())

    y_pred = logisticModel.predict(X_test)
    y_prob = logisticModel.predict_proba(X_test)

    # Show confusion matrix and accuracy scores.
    y_test_array = np.array(y_test)

    print("\n***K-fold: " + str(foldCount))
    foldCount += 1

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average=None, zero_division=1)
    recall = metrics.recall_score(y_test, y_pred, average=None)
    f1 = metrics.f1_score(y_test, y_pred, average=None)

    accuracyList.append(accuracy)
    precisionList.append(precision)
    recallList.append(recall)
    f1List.append(f1)

    print('\nAccuracy: ', accuracy)
    print(classification_report(y_test, y_pred))

    average_precision = average_precision_score(y_test, y_pred)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    # calculate scores
    auc = roc_auc_score(y_test, y_prob[:, 1], )
    print('Logistic: ROC AUC=%.3f' % (auc))

print("\n\nAccuracy, Precision, Recall, F1, and their respective standard deviations For All Folds:")
print("**********************************************************************************************")

print("\nAverage Accuracy: " + str(np.mean(accuracyList)))
print("Accuracy std: " + str(np.std(accuracyList)))

print("\nAverage Precision: " + str(np.mean(precisionList)))
print("Precision std: " + str(np.std(precisionList)))

print("\nAverage Recall: " + str(np.mean(recallList)))
print("Recall std: " + str(np.std(recallList)))

print("\nAverage F1: " + str(np.mean(f1List)))
print("F1 std: " + str(np.std(f1List)))

# """ Finding the best hyper-parameters for the classifiers/models later """
#
# # --------------------------------------------------------------
# # Grid search for optimal hyperparameter tuning
# # --------------------------------------------------------------
#
# print("\n\n ----- Finding the best hyperparameters to perform hyperparameter tuning "
#       "for the classifiers/models later ----- \n")
#
# # Initialize K-Fold cross-validation
# k_fold = KFold(n_splits=3, shuffle=True)
#
# # Define the hyperparameter grid for RandomForestClassifier
# rf_param_grid = {
#     'n_estimators': [800],
#     'max_depth': [5, 10, 15, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
#
# # Define the hyperparameter grid for XGBClassifier
# xgb_param_grid = {
#     'learning_rate': [0.01, 0.1, 0.5],
#     'max_depth': [3, 5, 7],
#     'subsample': [0.5, 0.8, 1.0]
# }
#
# # Create a GridSearchCV object for RandomForestClassifier
# rf_grid_search = GridSearchCV(RandomForestClassifier(), rf_param_grid, cv=k_fold, scoring='f1')
#
# # Fit the GridSearchCV object to the data
# rf_grid_search.fit(X_train, y_train)
#
# # Print the best hyperparameters and F1 score for RandomForestClassifier
# print("Best hyperparameters for RandomForestClassifier:", rf_grid_search.best_params_)
#
# # Create a GridSearchCV object for XGBClassifier
# xgb_grid_search = GridSearchCV(XGBClassifier(), xgb_param_grid, cv=k_fold, scoring='f1', error_score='raise')
#
# # Fit the GridSearchCV object to the data
# xgb_grid_search.fit(X_train, y_train)
#
# # Print the best hyperparameters and F1 score for XGBClassifier
# print("Best hyperparameters for XGBClassifier:", xgb_grid_search.best_params_)
#
# # Define parameter grid for LogisticRegression
# param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
#                  'penalty': ['l1', 'l2']}
#
# # create the logistic regression model with 'l2' penalty
# model_lr = LogisticRegression(fit_intercept=True, solver='lbfgs', penalty='l2')
#
# # Perform grid search to find best hyperparameters
# grid_search_lr = GridSearchCV(model_lr, param_grid_lr, cv=k_fold, scoring='f1')
# grid_search_lr.fit(X_train, y_train)
#
# # Print best hyperparameters
# print("Best hyperparameters for LogisticRegression:", grid_search_lr.best_params_)
#
# # Define parameter grid for DecisionTreeClassifier
# param_grid_dt = {'max_depth': [3, 5, 7, 9, 11],
#                  'min_samples_split': [2, 5, 10],
#                  'min_samples_leaf': [1, 2, 4]}
#
# # Create DecisionTreeClassifier model
# model_dt = DecisionTreeClassifier()
#
# # Perform grid search to find best hyperparameters
# grid_search_dt = GridSearchCV(model_dt, param_grid_dt, cv=k_fold, scoring='f1')
# grid_search_dt.fit(X_train, y_train)
#
# # Print best hyperparameters
# print("Best hyperparameters for DecisionTreeClassifier:", grid_search_dt.best_params_)
#
# # Define parameter grid for AdaBoostClassifier
# param_grid_ab = {'n_estimators': [50, 100, 200],
#                  'learning_rate': [0.01, 0.1, 1],
#                  'base_estimator__max_depth': [1, 3, 5]}
#
# # Create base estimator for AdaBoostClassifier
# base_estimator_ab = DecisionTreeClassifier()
#
# # Create AdaBoostClassifier model
# model_ab = AdaBoostClassifier(base_estimator=base_estimator_ab)
#
# # Perform grid search to find best hyperparameters
# grid_search_ab = GridSearchCV(model_ab, param_grid_ab, cv=k_fold, scoring='f1')
# grid_search_ab.fit(X_train, y_train)
#
# # Print best hyperparameters
# print("Best hyperparameters for AdaBoostClassifier:", grid_search_ab.best_params_)
#
# # Define the parameter grid
# param_grid = {
#     'n_estimators': [100, 500, 1000],
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.1, 1],
# }
#
# # Create an instance of the XGBClassifier
# xgb = XGBClassifier()
#
# # Create the GridSearchCV object with 5-fold cross-validation
# grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='f1')
#
# # Fit the GridSearchCV object to the data
# grid_search.fit(X_train, y_train)
#
# # Print the best hyperparameters and the corresponding score
# print(f"Best hyperparameters for XGBClassifier: {grid_search.best_params_}")

""" Bagged Model with Cross Fold Validation, Hyperparameter Tuning, and Scaling"""

print("\n\n ----- Bagged Model with Cross Fold Validation, Hyperparameter Tuning, and Scaling -----")
# Create classifiers
knn = KNeighborsClassifier()
svc = SVC()
rg = RidgeClassifier()
# lr = LogisticRegression(fit_intercept=True, solver='liblinear')
lr = LogisticRegression(max_iter=10000000, random_state=42)

# Build array of classifiers.
classifierArray = [knn, svc, rg, lr]

# Initialize K-Fold cross-validation
k_fold = KFold(n_splits=3, shuffle=True)


def showStats(classifier, scores):
    print(classifier + ":    ", end="")
    strMean = str(round(scores.mean(), 2))
    strStd = str(round(scores.std(), 2))
    print("Mean: " + strMean + "   ", end="")
    print("Std: " + strStd)


# Create a KNN imputer object with n_neighbors=5
imputer = KNNImputer(n_neighbors=5)

# Impute the missing values in the predictor variables
X_imputed = imputer.fit_transform(X)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)


def evaluateModel(model, X_test, y_test, title):
    print("\n\n*** " + title + " ***")
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    # print(report)
    # Calculate evaluation metrics
    print("-- Average evaluation metrics over cross fold validation folds --")
    acc_scores = cross_val_score(model, X, y, cv=k_fold, scoring='accuracy')
    precision_scores = cross_val_score(model, X, y, cv=k_fold, scoring='precision')
    recall_scores = cross_val_score(model, X, y, cv=k_fold, scoring='recall')
    f1_scores = cross_val_score(model, X, y, cv=k_fold, scoring='f1')
    # Print evaluation metrics
    showStats("Accuracy", acc_scores)
    showStats("Precision", precision_scores)
    showStats("Recall", recall_scores)
    showStats("F1 Score", f1_scores)


# Search for the best classifier.
for clf in classifierArray:
    modelType = clf.__class__.__name__

    # Create and evaluate stand-alone model.
    clfModel = clf.fit(X_train, y_train)
    evaluateModel(clfModel, X_test, y_test, modelType)

    # max_features means the maximum number of features to draw from X.
    # max_samples sets the percentage of available data used for fitting.
    # did hyperparameter tuning for "max_features" and "n_estimators"
    bagging_clf = BaggingClassifier(clf, max_samples=0.4, max_features=5, n_estimators=1000)
    baggedModel = bagging_clf.fit(X_train, y_train)
    evaluateModel(baggedModel, X_test, y_test, "BAGGED: " + modelType)

""" Ensemble Model with Cross Fold Validation, Hyperparameter Tuning, and Scaling"""

print("\n\n ----- Ensemble Model with Cross Fold Validation, Hyperparameter Tuning, and Scaling -----")

X = df[['PL', 'M11', 'BD2', 'PRG', 'Age']]

# Scale the predictor variables using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# with tuned hyperparemeter
ada_boost = AdaBoostClassifier(learning_rate=0.1, n_estimators=200)
grad_boost = GradientBoostingClassifier(learning_rate=0.01, max_depth=7, subsample=1.0)
xgb_boost = XGBClassifier(learning_rate=0.01, max_depth=3, n_estimators=500)
eclf = EnsembleVoteClassifier(clfs=[ada_boost, grad_boost, xgb_boost], voting='hard')
lr = LogisticRegression(C=1, penalty='l2', max_iter=100000)

# Build array of classifiers.
"""Only need the nsembleVoteClassifier"""
# classifiers = [ada_boost, grad_boost, xgb_boost, eclf, lr]
classifiers = [eclf]

# Set up KFold cross-validation
kf = KFold(n_splits=10, shuffle=True)

# Loop through the classifiers and perform cross-validation
for clf in classifiers:
    print()
    print(clf.__class__.__name__)
    # Initialize variables for storing the results across folds
    all_predictions = []
    all_true_labels = []
    all_accuracy_scores = []
    all_precision_scores = []
    all_recall_scores = []
    all_f1_scores = []
    for train_index, test_index in kf.split(X):
        # Split the data into train and test sets for this fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Train the classifier on the training set for this fold
        clf.fit(X_train, y_train)
        # Use the classifier to make predictions on the test set for this fold
        predictions = clf.predict(X_test)
        # Append the predictions and true labels to the running lists
        all_predictions.extend(predictions)
        all_true_labels.extend(y_test)
        # Compute the evaluation metrics for this fold
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        all_accuracy_scores.append(accuracy)
        all_precision_scores.append(precision)
        all_recall_scores.append(recall)
        all_f1_scores.append(f1)

    # Compute the average evaluation metrics across all folds for this classifier
    avg_accuracy = np.mean(all_accuracy_scores)
    std_accuracy = np.std(all_accuracy_scores)
    avg_precision = np.mean(all_precision_scores)
    std_precision = np.std(all_precision_scores)
    avg_recall = np.mean(all_recall_scores)
    std_recall = np.std(all_recall_scores)
    avg_f1 = np.mean(all_f1_scores)
    std_f1 = np.std(all_f1_scores)
    # Print the evaluation metrics for this classifier
    print("\nAverage evaluation metrics over cross fold validation folds:")
    print(f"Accuracy:\tmean: {avg_accuracy} \tstd: {std_accuracy}")
    print(f"Precision: \t{avg_precision} \tstd: {std_precision}")
    print(f"Recall: \tmean: {avg_recall} \ttsd: {std_recall}")
    print(f"F1-score: \t{avg_f1} \tstd: {std_f1}")

""" Stacked Model with Cross Fold Validation, Hyperparameter Tuning, and Scaling"""

# Initialize K-Fold cross-validation
k_fold = KFold(n_splits=3, shuffle=True)


def getUnfitModels():
    models = list()

    # models.append(LogisticRegression(max_iter=10000000))
    models.append(LogisticRegression(C=1, penalty='l2'))
    models.append(DecisionTreeClassifier(max_depth=11, min_samples_leaf=2, min_samples_split=10))
    models.append(AdaBoostClassifier())
    # tuned the hyperparameter n_estimators=800 from what was suggested in the best parameters to use for random forest
    models.append(RandomForestClassifier(max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=800))

    models.append(GradientBoostingClassifier(learning_rate=0.01, max_depth=7, subsample=1.0))
    models.append(XGBClassifier())
    return models


def evaluateStackedModel(y_test, predictions, model):
    # Calculate evaluation metrics
    print("\nAverage evaluation metrics over cross fold validation folds:")
    acc_scores = cross_val_score(model, X, y, cv=k_fold, scoring='accuracy')
    precision_scores = cross_val_score(model, X, y, cv=k_fold, scoring='precision')
    recall_scores = cross_val_score(model, X, y, cv=k_fold, scoring='recall')
    f1_scores = cross_val_score(model, X, y, cv=k_fold, scoring='f1')
    # Print evaluation metrics
    showStats("Accuracy", acc_scores)
    showStats("Precision", precision_scores)
    showStats("Recall", recall_scores)
    showStats("F1 Score", f1_scores)


def fitBaseModels(X_train, y_train, X_test, models):
    dfPredictions = pd.DataFrame()

    # Fit base model and store its predictions in dataframe.
    for i in range(0, len(models)):
        models[i].fit(X_train, y_train)
        predictions = models[i].predict(X_test)
        colName = str(i)
        dfPredictions[colName] = predictions
    return dfPredictions, models


def fitStackedModel(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model


# Create a KNN imputer object with n_neighbors=5
imputer = KNNImputer(n_neighbors=5)

# Impute the missing values in the predictor variables
X_imputed = imputer.fit_transform(X)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

# Get base models.
unfitModels = getUnfitModels()

# Fit base and stacked models.
dfPredictions, models = fitBaseModels(X_train, y_train, X_test, unfitModels)
stackedModel = fitStackedModel(dfPredictions, y_test)

# Evaluate base models with validation data.
print("\n---------- Evaluate Base Models ----------")
for i in range(0, len(models)):
    scores = cross_val_score(models[i], X_test, y_test, cv=5)
    print("\n*** " + models[i].__class__.__name__)
    predictions = models[i].predict(X_test)
    evaluateStackedModel(y_test, predictions, models[i])

# Evaluate stacked model with validation data.
print("\n---------- Evaluate Stacked Model ----------")
scores = cross_val_score(stackedModel, dfPredictions, y_test, cv=5)
stackedPredictions = stackedModel.predict(dfPredictions)
evaluateStackedModel(y_test, stackedPredictions, stackedModel)

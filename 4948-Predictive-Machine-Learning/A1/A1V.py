from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from mlxtend.classifier import EnsembleVoteClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning))

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)

# Load the dataset into a pandas DataFrame object
PATH = "C:/datasets/"
CSV_DATA = "loans.csv"
df = pd.read_csv(PATH + CSV_DATA, sep=',')

"""
                                                    Clean the data
"""

print("\n---------- Data pre-cleaning ----------\n")
print(df.head(10))
print()
print(df.describe().T)

# Convert the 'Loan_Status' column with label encoding
le = LabelEncoder()
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

# Separate the target variable and predictor variables
y = df['Loan_Status']
X = df.drop(['Loan_Status', 'Loan_ID'], axis=1)

# Convert categorical columns to label encoding
cat_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))

# Impute missing values using KNN imputer
imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Add the target variable back to the imputed dataset
X_imputed['Loan_Status'] = y

print("\n---------- Cleaned up data ----------\n")
print(X_imputed.head(10))
print()
print(X_imputed.describe().T)

# Save the cleaned dataset as a CSV file
X_imputed.to_csv("C:/datasets/cleaned_loan_data.csv", index=False)

df = X_imputed


"""
Generate a summary using the cleaned data
"""
# from pandas_profiling import ProfileReport
# # make a report with the clean data
# prof = ProfileReport(X_imputed)
# prof.to_file(output_file='output.html')


# Correlation matrix
plt.figure(figsize=(11, 11))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title('Correlation Matrix')
plt.show()


"""
PLOTS OF VARIABLES
"""
# Plot a histogram of the 'Temperature' column
plt.hist(df['Temperature'], bins=20)
plt.title('Histogram of Temperature')
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.show()
#
# # Coapplicant Income plots
# plt.figure(figsize=(10, 6))
# sns.histplot(x='CoapplicantIncome', data=df)
# plt.title('Coapplicant Income Distribution')
# plt.show()
#
# # Loan Amount plots
# plt.figure(figsize=(10, 6))
# sns.histplot(x='LoanAmount', data=df)
# plt.title('Loan Amount Distribution')
# plt.show()
#
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='LoanAmount', data=df)
# plt.title('Loan Amount Boxplot')
# plt.show()
#
# # Loan Term plots
# plt.figure(figsize=(10, 6))
# sns.histplot(x='Loan_Amount_Term', data=df)
# plt.title('Loan Term Distribution')
# plt.show()
#
# # Credit History plots
# plt.figure(figsize=(10,6))
# sns.countplot(x='Credit_History', data=df)
# plt.title('Credit History Distribution')
# plt.show()
#
# # Property Area plots
# plt.figure(figsize=(10, 6))
# sns.countplot(x='Property_Area', data=df)
# plt.title('Property_Area Distribution')
# plt.show()
#
# # Loan Status Plots
# plt.figure(figsize=(10, 6))
# sns.countplot(x='Loan_Status', data=df)
# plt.title('Loan Status Distribution')
# plt.show()

"""
                                                    Feature Selection
"""


# Split data into train and test sets
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


"""
Feature Selection with Recursive Feature Elimination (RFE)
"""

# # Separate the target variable from the feature variables
# X = df.drop('Loan_Status', axis=1)
# y = df['Loan_Status']

# Create a logistic regression model
model = LogisticRegression()

# Perform recursive feature elimination
rfe = RFE(estimator=model, n_features_to_select=5)
rfe.fit(X, y)

# Print the selected features
selected_features = X.columns[rfe.support_]
print("\n\n*** Recursive Feature Elimination")
print('Selected Features: {}'.format(', '.join(selected_features)))


"""
Feature Selection with Forward Feature Selection (FFS)
"""

# Create a logistic regression model
model = LogisticRegression(max_iter=1000)

# Use f_regression for forward feature selection
# choose the feature with the highest F statistic
ffs = f_regression(X, y)

featuresDf = pd.DataFrame()
for i in range(0, len(X.columns)):
    featuresDf = featuresDf.append({"feature": X.columns[i],
                                    "ffs": ffs[0][i]}, ignore_index=True)
featuresDf = featuresDf.sort_values(by=['ffs'], ascending=False)
print("\n\n*** Forward Feature Selection")
print("\nSignificant features in descending F-statistic values:")
# Show the top 5 features
print(featuresDf.head())


"""
Feature Selection with Random Forest
"""

# Use the random grid to search for best hyperparameters
rf = RandomForestRegressor()
random_grid = \
    {'bootstrap': [True],
     'max_depth': [2, 10, None],
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


# Create the new rf with the best random forest params suggested above
rf = RandomForestRegressor(n_estimators=800, min_samples_split=15, min_samples_leaf=15, max_features=1.0,
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


"""
                                        Selecting the combination of features
                                        
                                        Recursive Feature:
                                        Gender, Married, Education, Credit_History, Property_Area
                                        Forward Feature:
                                        Credit_History, Married, Education, CoapplicantIncome, LoanAmount
                                        Random Forest:
                                        Credit_History, LoanAmount, ApplicantIncome, CoapplicantIncome, Property_Area
                                        
                                        Chosen features to use :
                                        X = df[['Credit_History', 
                                                'Property_Area', 
                                                'CoapplicantIncome', 
                                                'LoanAmount',
                                                'Married']]
"""


"""
                                        Logistic Regression with Cross Fold Validation and Scaling
"""

y = df['Loan_Status']
X = df[['Credit_History',
        'Property_Area',
        'CoapplicantIncome',
        'LoanAmount',
        'Married']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# print("\n\n***Logistic Regression with Cross Fold Validation and Scaling")
#
# # Create a logistic regression model
# model = LogisticRegression()
#
# # Define the metrics to evaluate the model
# scoring = {
#     'accuracy': 'accuracy',
#     'precision': make_scorer(precision_score),
#     'recall': make_scorer(recall_score),
#     'f1_score': make_scorer(f1_score)
# }
#
# # Perform cross-validation with 3 folds and a random seed of 1
# scores = cross_validate(model, X, y, cv=3, scoring=scoring)
#
# # Print the mean and standard deviation of scores across all folds
# print('Mean Accuracy: {:.2f}% (+/- {:.2f})'.format(scores['test_accuracy'].mean() * 100, scores['test_accuracy'].std()))
# print('Mean Precision: {:.2f}% (+/- {:.2f})'.format(scores['test_precision'].mean() * 100, scores['test_precision'].std()))
# print('Mean Recall: {:.2f}% (+/- {:.2f})'.format(scores['test_recall'].mean() * 100, scores['test_recall'].std()))
# print('Mean F1 Score: {:.2f}% (+/- {:.2f})'.format(scores['test_f1_score'].mean() * 100, scores['test_f1_score'].std()))

print("\n\n***Logistic Regression with Cross Fold Validation and Scaling")

# prepare cross validation with three folds and 1 as a random seed.
kfold = KFold(n_splits=3, shuffle=True)

accuracyList = []
precisionList = []
recallList = []
f1List = []

foldCount = 0

for train_index, test_index in kfold.split(df):
    X_train = X.iloc[train_index, :]
    X_test = X.iloc[test_index, :]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    # Perform logistic regression.
    logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear', random_state=42)  # Removed random state
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


"""
                                        Bagged Model with Cross Fold Validation
"""

y = df['Loan_Status']
X = df[['Credit_History',
        'Property_Area',
        'CoapplicantIncome',
        'LoanAmount',
        'Married']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


print("\n\n ----- Bagged Model with Cross Fold Validation -----")
# Create classifiers
knn = KNeighborsClassifier()
svc = SVC()
rg = RidgeClassifier()
lr = LogisticRegression(max_iter=1000, random_state=42)

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

# Scale the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)


def evaluateModel(model, X_test, y_test, title):
    print("\n\n*** " + title + " ***")
    predictions = model.predict(X_test)
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
    bagging_clf = BaggingClassifier(clf, max_samples=0.4, max_features=3,
                                    n_estimators=100)
    baggedModel = bagging_clf.fit(X_train, y_train)
    evaluateModel(baggedModel, X_test, y_test, "Bagged: " + modelType)


"""
                                        Ensemble Model
"""

y = df['Loan_Status']
X = df[['Credit_History',
        'Property_Area',
        'CoapplicantIncome',
        'LoanAmount',
        'Married']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("\n*** " +"Ensemble Model\n")
ada_boost = AdaBoostClassifier()
grad_boost = GradientBoostingClassifier()
xgb_boost = XGBClassifier()
lr = LogisticRegression(random_state=42)

eclf = EnsembleVoteClassifier(clfs=[ada_boost, grad_boost,
                                    xgb_boost, lr], voting='hard')
# classifiers = [eclf]
# for clf in classifiers:
#     print(clf.__class__.__name__)
#     clf.fit(X_train, y_train)
#     predictions = clf.predict(X_test)
#     report = classification_report(y_test, predictions)
#     print(report)


print(eclf.__class__.__name__)
eclf.fit(X_train, y_train)
predictions = eclf.predict(X_test)
report = classification_report(y_test, predictions)
print(report)


"""
                                        Stacked Model
"""

y = df['Loan_Status']
X = df[['Credit_History',
        'Property_Area',
        'CoapplicantIncome',
        'LoanAmount',
        'Married']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("\n*** " +"Stacked Model with Cross Fold Validation\n")


print("Stacked Model")

def getUnfitModels():
    models = list()
    models.append(LogisticRegression(max_iter=1000))
    models.append(DecisionTreeClassifier())
    models.append(AdaBoostClassifier())
    models.append(RandomForestClassifier(n_estimators=10))
    return models


def evaluateModel(y_test, predictions, model):
    print("\n*** " + model.__class__.__name__)
    report = classification_report(y_test, predictions)
    print(report)


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


# Split data into train, test and validation sets.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.70)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50)

# Get base models.
unfitModels = getUnfitModels()

# Fit base and stacked models.
dfPredictions, models = fitBaseModels(X_train, y_train, X_val, unfitModels)
stackedModel = fitStackedModel(dfPredictions, y_val)

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

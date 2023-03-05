# import necessary libraries
import matplotlib.pylab as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier

from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import warnings
from sklearn.model_selection import KFold

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning))
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

################################# DATA LOADING & CLEANING #################################
PATH = Path("C:\\Users\\victo\Documents\PyCharm\COMP-4948-4949\\4949-Big-Data-Analytics\A1\survey lung cancer.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
df = pd.read_csv(PATH)
df = df.drop_duplicates()

le = preprocessing.LabelEncoder()

df['GENDER'] = le.fit_transform(df['GENDER'])
df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])
df['SMOKING'] = le.fit_transform(df['SMOKING'])
df['YELLOW_FINGERS'] = le.fit_transform(df['YELLOW_FINGERS'])
df['ANXIETY'] = le.fit_transform(df['ANXIETY'])
df['PEER_PRESSURE'] = le.fit_transform(df['PEER_PRESSURE'])
df['CHRONIC DISEASE'] = le.fit_transform(df['CHRONIC DISEASE'])
df['FATIGUE'] = le.fit_transform(df['FATIGUE '])
df['ALLERGY'] = le.fit_transform(df['ALLERGY '])
df['WHEEZING'] = le.fit_transform(df['WHEEZING'])
df['ALCOHOL CONSUMING'] = le.fit_transform(df['ALCOHOL CONSUMING'])
df['COUGHING'] = le.fit_transform(df['COUGHING'])
df['SHORTNESS OF BREATH'] = le.fit_transform(df['SHORTNESS OF BREATH'])
df['SWALLOWING DIFFICULTY'] = le.fit_transform(df['SWALLOWING DIFFICULTY'])
df['CHEST PAIN'] = le.fit_transform(df['CHEST PAIN'])
df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])

################################# AUTO GENERATES A SUMMARY OF DF#################################
# from pandas_profiling import ProfileReport
# prof = ProfileReport(df)
# prof.to_file(output_file='output.html')


################################## DAT VISUALIZATION #################################
# plt.figure(figsize=(10, 10))
# sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
# plt.title('Correlation Matrix')
# plt.show()
#
# # Bar graph of gender distribution
# plt.figure(figsize=(6, 6))
# df['GENDER'].value_counts().plot(kind='bar')
# plt.xlabel('GENDER')
# plt.ylabel('Count')
# plt.title('Gender Distribution')
# plt.xticks(rotation=0)
# plt.show()
#
# # Pie chart of smoking status
# plt.figure(figsize=(6, 6))
# df['SMOKING'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, labels=['Non-smoker', 'Smoker'])
# plt.title('Smoking Status')
# plt.show()
#
# # Box plot of age by lung cancer status
# plt.figure(figsize=(6, 6))
# sns.boxplot(x='LUNG_CANCER', y='AGE', data=df)
# plt.xlabel('Lung Cancer')
# plt.ylabel('AGE')
# plt.title('Age Distribution by Lung Cancer Status')
# plt.xticks([0, 1], ['No', 'Yes'])
# plt.show()
#
# sns.heatmap(df.corr())
# plt.show()
# plt.figure(figsize=(15, 6))
# sns.countplot(x='AGE', data=df, hue='LUNG_CANCER')
# plt.show()
# sns.displot(x='AGE', data=df, hue='LUNG_CANCER')
# plt.show()
#
# data = df['LUNG_CANCER'].value_counts().values
# labels = df['LUNG_CANCER'].value_counts().index
# plt.pie(data, labels=['Smoke', 'NO Smoke'], autopct='%2.1f%%', shadow=True)
#
# X = ['YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
#      'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
# fig, ax = plt.subplots(nrows=4, ncols=3)  # 16 subplots
# fig.set_size_inches(16, 24)  # set figure size
#
# for i in range(4):
#     for j in range(3):
#         sns.countplot(x=df[X[3 * i + j]], ax=ax[i][j])  # count plot
# plt.show()
#
# smoke_yes = df.loc[df.SMOKING == 1, ["SMOKING", "LUNG_CANCER"]]
# smoke_no = df.loc[df.SMOKING == 0, ["SMOKING", "LUNG_CANCER"]]
#
# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
# ax1.pie(smoke_yes.LUNG_CANCER.value_counts(normalize=True), labels=["YES", "NO"], colors=["yellow", "green"],
#         autopct='%1.1f%%', shadow=True, )
# ax1.set_title("Lung Cancer & Smoking_YES")
#
# ax2.pie(smoke_no.LUNG_CANCER.value_counts(normalize=True), labels=["YES", "NO"], colors=["red", "green"],
#         autopct='%1.1f%%', shadow=True, )
# ax2.set_title("Lung Cancer & Smoking_NO")
# plt.show()
# plt.show()

# Split data into train and test sets
X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']
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
print("----------------------------------------")


# We create the new rf with the best random forest params suggested above
rf = RandomForestRegressor(n_estimators=1600, min_samples_split=15, min_samples_leaf=15, max_features='auto',
                           max_depth=None, bootstrap=True)
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


#### RFE  ####
print("RFE HEREERERERERE")
# Create a logistic regression estimator
estimator = LogisticRegression()

# Create the RFE object and specify the number of features to select
rfe = RFE(estimator, n_features_to_select=3, step=1)

# Fit the RFE object to the data
rfe.fit(X, y)

# Print the selected features
print("RFE Selected Features:")
for i, col in enumerate(X.columns):
    if rfe.support_[i]:
        print(col)



#### FFS  ####


# Initialize a logistic regression model. performs forward feature selection on the lung cancer
logreg = LogisticRegression(max_iter=10000000)

# Perform forward feature selection with 5-fold cross-validation.
sfs = SequentialFeatureSelector(logreg, direction="forward", n_features_to_select=3, cv=5)
sfs.fit(X_train, y_train)

# Print the selected feature indices and names.
print("FFS Selected Features:", list(df.drop(columns=['LUNG_CANCER']).columns[sfs.get_support()]))


### List of ALL of the features suggeseted by the three above methods (RFE, FFS, FI)
# X = df[['PEER_PRESSURE', 'YELLOW_FINGERS', 'COUGHING', 'SWALLOWING DIFFICULTY','ALLERGY', 'GENDER', 'AGE', 'SMOKING']]


## These features were selected
X = df[['SWALLOWING DIFFICULTY', 'COUGHING', 'ALLERGY', 'SMOKING', 'YELLOW_FINGERS']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)




################################## MODELS #################################


#### Logistic Regression & Bagged Decision Trees Ridge Models ####

models = {
    'Logistic Regression': LogisticRegression(),
    'Bagged Decision Trees Ridge': BaggingClassifier(RidgeClassifier(), max_samples=0.4, max_features=3,
                                                     n_estimators=100),
    # 'SGCClassifier': SGDClassifier(random_state=42),
    # 'Decision Tree': DecisionTreeClassifier(random_state=42),
    # 'Random Forest': RandomForestClassifier(random_state=42),
    # 'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    # 'Bagged Decision Trees': BaggingClassifier(random_state=42),
    # 'SVM': SVC(random_state=42),
}

score_dict = {}
for name, model in models.items():
    print("Model:", name)
    print("---------------")

    # Cross-validation
    kfold = KFold(n_splits=3)
    f1 = cross_val_score(model, X_train, y_train, cv=kfold, scoring="f1")
    accuracy = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
    precision = cross_val_score(model, X_train, y_train, cv=kfold, scoring="average_precision")
    print("Mean F1 score:", np.mean(f1))
    print("Mean accu score:", np.mean(accuracy))
    print("Mean precision score:", np.mean(precision))
    score_dict[name] = ("F1:", np.mean(f1), "Accuracy:", np.mean(accuracy), "Precision:", np.mean(precision))

print(score_dict)



########## This section is used to find the best classifier (in our case RidgeClassifier() ) for the BaggingClassifier model ########
# Create classifiers
knn = KNeighborsClassifier()
svc = SVC()
rg = RidgeClassifier()

# Build array of classifiers.
classifierArray = [knn, svc, rg]


def showStats(classifier, scores):
    print(classifier + ":    ", end="")
    strMean = str(round(scores.mean(), 2))

    strStd = str(round(scores.std(), 2))
    print("Mean: " + strMean + "   ", end="")
    print("Std: " + strStd)


def evaluateModel(model, X_test, y_test, title):
    print("\n*** " + title + " ***")
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)


    print(report)


# Search for the best classifier
print("Search for the best classifier:")
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



#### Ensemble Model ####


print("Ensemble Model")
ada_boost = AdaBoostClassifier()
grad_boost = GradientBoostingClassifier()
xgb_boost = XGBClassifier()
lr = LogisticRegression(random_state=42)

eclf = EnsembleVoteClassifier(clfs=[ada_boost, grad_boost,
                                    xgb_boost, lr], voting='hard')
classifiers = [ada_boost, grad_boost, xgb_boost, lr, eclf]
for clf in classifiers:
    print(clf.__class__.__name__)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    report = classification_report(y_test, predictions)
    print(report)


#### Stacked  Model ####

print("Stacked Model")
def getUnfitModels():
    models = list()
    models.append(LogisticRegression())
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

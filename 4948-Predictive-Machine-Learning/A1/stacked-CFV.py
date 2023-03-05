import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)

# Load the dataset into a pandas DataFrame object
df = pd.read_csv('https://drive.google.com/uc?id=1r29kUG_mtmiQFVuUTJqogNo_l0jNz_iK')

# Split the dataset into a target variable and predictor variables
y = df['Loan_Status']

# X = df.drop('Loan_Status', axis=1)
# X = df[['Credit_History', 'Education', 'Property_Area', 'Married', 'ApplicantIncome']]
X = df[['Credit_History', 'Education', 'Property_Area', 'Married', 'ApplicantIncome', 'Gender']]

# Initialize K-Fold cross-validation
k_fold = KFold(n_splits=3, shuffle=True)


def getUnfitModels():
    models = list()
    models.append(LogisticRegression())
    models.append(DecisionTreeClassifier())
    models.append(AdaBoostClassifier())
    models.append(RandomForestClassifier(n_estimators=100))
    return models


def showStats(classifier, scores):
    strMean = str(round(scores.mean(), 2))
    print(f"Average {classifier}: {strMean}")


def evaluateModel(y_test, predictions, model):
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


# Split data into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

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
    evaluateModel(y_test, predictions, models[i])

# Evaluate stacked model with validation data.
print("\n---------- Evaluate Stacked Model ----------")
scores = cross_val_score(stackedModel, dfPredictions, y_test, cv=5)
stackedPredictions = stackedModel.predict(dfPredictions)
evaluateModel(y_test, stackedPredictions, stackedModel)

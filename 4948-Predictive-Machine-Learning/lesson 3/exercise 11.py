from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import AdaBoostClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import classification_report
from sklearn.model_selection import train_test_split
import numpy  as np
import pandas as pd

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Prepare the data.
# Get the housing data
df = pd.read_csv("C:\\datasets\\iris_v2.csv")
dict_map = {'Iris-setosa':0,'Iris-versicolor':1, 'Iris-virginica':2}
df['target'] = df['iris_type'].map(dict_map)

y = df['target']
X = df.copy()
del X['target']
del X['iris_type']


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
print(dfPredictions.head(10))
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
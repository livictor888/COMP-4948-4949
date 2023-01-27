import pandas  as pd
from sklearn.metrics import classification_report


PATH = 'C:\\datasets\\housing_classification'
CSV     = ".csv"

# Get the housing data
df = pd.read_csv(PATH + CSV)

# Get the housing data
# df = pd.read_csv('housing_classification.csv')
# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head(5))

# Split into two sets
y = df['price']
X = df.drop('price', 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.model_selection import cross_val_score
from mlxtend.classifier      import EnsembleVoteClassifier
from xgboost                 import XGBClassifier, plot_importance
from sklearn.ensemble        import AdaBoostClassifier, GradientBoostingClassifier

ada_boost   = AdaBoostClassifier()
grad_boost  = GradientBoostingClassifier()
xgb_boost   = XGBClassifier()
classifiers = [ada_boost, grad_boost, xgb_boost]

for clf in classifiers:
    print(clf.__class__.__name__)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    report = classification_report(y_test, predictions)
    print(report)


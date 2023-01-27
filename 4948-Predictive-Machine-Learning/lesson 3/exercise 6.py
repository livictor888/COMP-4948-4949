import pandas  as pd
from sklearn.metrics import classification_report
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Get the housing data
df = pd.read_csv('C:\\datasets\\iris_v2.csv')

dict_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df['target'] = df['iris_type'].map(dict_map)

y = df['target']
X = df.copy()
del X['target']
del X['iris_type']

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

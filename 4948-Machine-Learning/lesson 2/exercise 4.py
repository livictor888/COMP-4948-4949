import sklearn.datasets as datasets
from   sklearn.tree import DecisionTreeClassifier
from   sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# check for the sklearn version, it has to be 0.21
import sklearn
print(sklearn.__version__)
breast_cancer = datasets.load_breast_cancer()

import pandas as pd
dfX =  pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
dfy = breast_cancer.target
print(dfX.head())
print(dfy)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    breast_cancer.data, breast_cancer.target, test_size=0.20,
    random_state=0)

print(breast_cancer)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

def showAccuracyScores(y_test, y_pred):
    print("\nModel Evaluation")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("")
    tn = cm[0][0]
    fp = cm[0][1]
    tp = cm[1][1]
    fn = cm[1][0]
    accuracy  = (tp + tn)/(tn + fp + tp + fn)
    precision = tp/(tp + fp)
    recall    = tp/(tp + fn)
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))

showAccuracyScores(y_test, y_pred)

from sklearn.tree import plot_tree
fig, ax = plt.subplots(figsize=(20, 10))

plot_tree(classifier.fit(X_train, y_train), max_depth=2, fontsize=4)
a = plot_tree(classifier,
              feature_names=['mean radius', 'mean texture', 'worst symmetry', 'worst fractal dimension'],
              class_names='Class',
              filled=True,
              rounded=True,
              fontsize=14)
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
PATH = 'C:\\datasets\\'
CSV     = "bill_authentication.csv"
dataset = pd.read_csv(PATH + CSV)
X       = dataset.drop('Class', axis=1)
y       = dataset['Class']
print(dataset)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=0)

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

plot_tree(classifier.fit(X_train, y_train), max_depth=4, fontsize=4)
a = plot_tree(classifier,
              feature_names=['Variance', 'Skewness', 'Kurtosis', 'Entropy'],
              class_names='Class',
              filled=True,
              rounded=True,
              fontsize=14)
plt.show()


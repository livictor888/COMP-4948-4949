from sklearn import datasets
iris = datasets.load_iris()

# Creating a DataFrame of given iris dataset.
import pandas as pd
data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})
iris['target_names']
print(data.head())

# Import train_test_split function
from sklearn.model_selection import train_test_split
X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y=data['species']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.preprocessing import StandardScaler
sc_x            = StandardScaler()
X_train_scaled  = sc_x.fit_transform(X_train)
X_test_scaled = sc_x.transform(X_test)

from sklearn              import metrics
from sklearn.ensemble     import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def buildModelAndPredict(clf, X_train_scaled, X_test_scaled, y_train, y_test, title):
    print("\n**** " + title)
    #Train the model using the training sets y_pred=rf.predict(X_test)
    clf_fit = clf.fit(X_train_scaled,y_train)
    y_pred = clf_fit.predict(X_test_scaled)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # For explanation see:
    # https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
    print(metrics.classification_report(y_test, y_pred, digits=3))

    # Predict species for a single flower.
    # sepal length = 3, sepal width = 5
    # petal length = 4, petal width = 2
    prediction = clf_fit.predict([[3, 5, 4, 2]])

    # 'setosa', 'versicolor', 'virginica'
    print(prediction)


lr = LogisticRegression(fit_intercept=True, solver='liblinear')
# add forest with 200 estimators and 3 max features to compare with logistic regression
rf = RandomForestClassifier(n_estimators=200, max_features=3)
buildModelAndPredict(lr, X_train_scaled, X_test_scaled, y_train, y_test, "Logistic Regression")
buildModelAndPredict(rf, X_train_scaled, X_test_scaled, y_train, y_test, "Random Forest")

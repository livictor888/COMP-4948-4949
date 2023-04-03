import matplotlib.pyplot as plt
import matplotlib.style as style
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

print("Before cleaning: \n", train_data.head())




""" -----DATA SETUP----- """

# one-hot encoding by using pandas get dummies function
train_data = pd.get_dummies(train_data, columns=["color"], prefix=["color"])
# change the type column by indexing ghoul, goblin, ghost to 1, 2 and 0
map_type = {"Ghoul": 1, "Goblin": 2, "Ghost": 0}
train_data.loc[:, "type"] = train_data.type.map(map_type)

# Set the ID column as the index
train_data = train_data.set_index('id')

print("After setup: \n", train_data.head())





""" -----TUNE AND TRAIN----- """

# separate target from the other variables
X = train_data.drop(["type"], axis=1)
y = train_data.type

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

lgbm = LGBMClassifier()  # Shortname the LGBMClassifier()
lgbm.fit(X_train, y_train)  # Train the lgbm on train sets

""" 
With Grid Search:
Parameters and intervals tested to find the optimized values
    ~3 million adjustments over ~9 hours

param_grid = {
    'class_weight': [None, 'balanced'],
    'boosting_type': ['gbdt', 'goss', 'dart'],
    'num_leaves': list(range(30, 150)),
    'learning_rate': [0.01,0.1,0.5],
    'subsample_for_bin': [20000,50000,100000,120000,150000],
    'min_child_samples': [20,50,100,200,500],
    'colsample_bytree': [0.6,0.8,1],
    "max_depth": [5,10,50,100]
}
"""

lgbm = LGBMClassifier()  # Shortname the LGBMClassifier()
lgbm.fit(X_train, y_train)  # Train the lgbm on train sets

# Using the optimized values
lgbm_tuned = LGBMClassifier(boosting_type='gbdt',  # Gradient Boosting Decision Tree
                            min_child_samples=20,
                            num_leaves=30,
                            subsample_for_bin=20000,  # number of samples for constructing feature histograms,higher
                            # value can improve estimation quality at the cost of memory usage and time
                            learning_rate=0.01,
                            max_depth=10,
                            n_estimators=40,
                            colsample_bytree=0.6)  # LightGBM Classifier with optimum parameters
lgbm_tuned.fit(X_train, y_train)

y_test_pred = lgbm_tuned.predict(X_test)  # Predicting X_test to find the solution
score = round(accuracy_score(y_test, y_test_pred), 3)  # Find accuracy of y_test and predictions, and round the result
print("Accuracy score: ", score)







"""-----SHOW RESULTS-----"""

sns.set_context("talk")
style.use('fivethirtyeight')

fi = pd.DataFrame()
fi['features'] = X.columns.values.tolist()
fi['importance'] = lgbm_tuned.booster_.feature_importance(importance_type='gain')

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x='importance', y='features', data=fi.sort_values(by='importance', ascending=True), ax=ax)
ax.set_xlabel('Importance', fontsize=16)
ax.set_ylabel('Features', fontsize=16)
plt.tight_layout()
plt.show()

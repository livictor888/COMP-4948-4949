import pandas  as pd
import numpy   as np
import pickle
import sklearn.metrics as metrics
# Setup data.
candidates = {'gmat': [780,750,690,710,680,730,690,720,
 740,690,610,690,710,680,770,610,580,650,540,590,620,
 600,550,550,570,670,660,580,650,660,640,620,660,660,
 680,650,670,580,590,690],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,
 3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,
 3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,
 3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,
 1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,
 5,1,2,1,4,5],
              'admitted': [1,1,1,1,1,1,0,1,1,0,0,1,
 1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,
 0,0,1]}

df = pd.DataFrame(candidates,columns= ['gmat', 'gpa',
                                       'work_experience','admitted'])
print(df)

# Separate into x and y values.
predictorVariables = ['gmat', 'gpa','work_experience']
X = df[predictorVariables]
y = df['admitted']

# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# You imported the libraries to run the experiments. Now, let's see it in action.

# Show chi-square scores for each feature.
# There is 1 degree freedom since 1 predictor during feature evaluation.
# Generally, >=3.8 is good)
test      = SelectKBest(score_func=chi2, k=3)
chiScores = test.fit(X, y) # Summarize scores
np.set_printoptions(precision=3)

print("\nPredictor variables: " + str(predictorVariables))
print("Predictor Chi-Square Scores: " + str(chiScores.scores_))

# Another technique for showing the most statistically
# significant variables involves the get_support() function.
cols = chiScores.get_support(indices=True)
print(cols)
features = X.columns[cols]
print(np.array(features))
from   sklearn.model_selection import train_test_split
from   sklearn.linear_model    import LogisticRegression

# Re-assign X with significant columns only after chi-square test.
X = df[['gmat', 'work_experience']]

# Split data.
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25,
                                                 random_state=0)

# Build logistic regression model and make predictions.
logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear',
                                   random_state=0)
logisticModel.fit(X_train,y_train)

# Save the model.
with open('model_pkl', 'wb') as files:
    pickle.dump(logisticModel, files)

# load saved model
with open('model_pkl' , 'rb') as f:
    loadedModel = pickle.load(f)

y_pred=loadedModel.predict(X_test)
print(y_pred)

# Show confusion matrix and accuracy scores.
from   sklearn                 import metrics
cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print('\nAccuracy: ',metrics.accuracy_score(y_test, y_pred))
print("\nConfusion Matrix")
recall = metrics.recall_score(y_test, y_pred)
print("Recall: " + str(recall))
precision = metrics.precision_score(y_test, y_pred)
print("Precision: " + str(precision))
print(cm)

# Create a single prediction.
singleSampleDf = pd.DataFrame(columns=['gmat', 'work_experience'])
gmat =  550
workExperience = 4

admissionsData = {'gmat':gmat, 'work_experience':workExperience}
singleSampleDf = pd.concat([singleSampleDf,
                            pd.DataFrame.from_records([admissionsData])])

singlePrediction = loadedModel.predict(singleSampleDf)
print("Single prediction: " + str(singlePrediction))

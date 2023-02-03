import nltk
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LogisticRegression
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import accuracy_score

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from   sklearn.feature_extraction.text import CountVectorizer
import math

# To get stop words.
nltk.download('stopwords')

# -------------------------------------------------------------
# Create lower case array of words with no punctuation.
# -------------------------------------------------------------
def createTokenizedArray(sentences):
    # Initialize tokenizer and empty array to store modified sentences.
    tokenizer = RegexpTokenizer(r'\w+')
    tokenizedArray = []
    for i in range(0, len(sentences)):
        # Convert sentence to lower case.
        sentence = sentences[i].lower()

        # Split sentence into array of words with no punctuation.
        words = tokenizer.tokenize(sentence)

        # Append word array to list.
        tokenizedArray.append(words)

    print(tokenizedArray)
    return tokenizedArray  # send modified contents back to calling function.

# -------------------------------------------------------------
# Create array of words with no punctuation or stop words.
# -------------------------------------------------------------
def removeStopWords(tokenList):
    stopWords = set(stopwords.words('english'))
    shorterSentences = []  # Declare empty array of sentences.

    for sentence in tokenList:
        shorterSentence = []  # Declare empty array of words in single sentence.
        for word in sentence:
            if word not in stopWords:

                # Remove leading and trailing spaces.
                word = word.strip()

                # Ignore single character words and digits.
                if (len(word) > 1 and word.isdigit() == False):
                    # Add remaining words to list.
                    shorterSentence.append(word)
        shorterSentences.append(shorterSentence)
    return shorterSentences

# -------------------------------------------------------------
# Removes suffixes and rebuids the sentences.
# -------------------------------------------------------------
def stemWords(sentenceArrays):
    ps = PorterStemmer()
    stemmedSentences = []
    for sentenceArray in sentenceArrays:
        stemmedArray = []  # Declare empty array of words.
        for word in sentenceArray:
            stemmedArray.append(ps.stem(word))  # Add stemmed word.

        # Convert array back to sentence of stemmed words.
        delimeter = ' '
        sentence = delimeter.join(stemmedArray)

        # Append stemmed sentence to list of sentences.
        stemmedSentences.append(sentence)
    return stemmedSentences

#-------------------------------------------------------------
# Creates a matrix of word vectors.
#-------------------------------------------------------------
def vectorizeList(stemmedList):
    #cv  = CountVectorizer(binary=True, ngram_range=(1, 4))
    cv = CountVectorizer(binary=True)

    cv.fit(stemmedList)
    X   = cv.transform(stemmedList)
    print("\nNumber vector size: " + str(X.shape))
    return X

import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report
# -------------------------------------------------------------
# Build model and predict scores.
#
# Parameters:
# X         - X contains the stemmed and vectorized sentences.
# target    - The target is the known rating (0 to 4).

# Returns X_test, y_test, and y_predicted values.
# -------------------------------------------------------------
def modelAndPredict(X, target):
    # Create training set with 75% of data and test set with 25% of data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, train_size=0.75
    )

    # Build the model with the training data.
    model = LogisticRegression(solver='newton-cg').fit(X_train, y_train)

    # Predict target values.
    y_prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_prediction)

    print("\n\n*** The accuracy score is: " + str(accuracy))

    print(classification_report(y_test, y_prediction))

    rmse2 = math.sqrt(metrics.mean_squared_error(y_test, y_prediction))
    print("RMSE: " + str(rmse2))

    # Your Python functions can return multiple values.
    return X_test, y_test, y_prediction

# Read in the file.
PATH = "C:\\datasets\\"

CLEAN_DATA = "cleanedMovieReviews.tsv"
df = pd.read_csv(PATH + CLEAN_DATA, skiprows=1,
                 sep='\t', names=('PhraseId', 'SentenceId', 'Phrase', 'Sentiment'))

# Prepare the data.
df['PhraseAdjusted'] = createTokenizedArray(df['Phrase'])
df['PhraseAdjusted'] = removeStopWords(df['PhraseAdjusted'])
df['PhraseAdjusted'] = stemWords(df['PhraseAdjusted'])
vectorizedList       = vectorizeList(df['PhraseAdjusted'])

# Get predictions and scoring data.
# Target is the rating that we want to predict.
X_test, y_test, y_predicted = modelAndPredict(vectorizedList, df['Sentiment'])


##### example 6



from sklearn import metrics

# Draw the confusion matrix.
def showConfusionMatrix(y_test, y_predicted):
    # You can print a simple confusion matrix with no formatting – this is easiest.
    cm = metrics.confusion_matrix(y_test.values, y_predicted)
    print(cm)

showConfusionMatrix(y_test, y_predicted)



### example 7

from collections import Counter
from nltk.util import ngrams


def generateWordList(wordDf, scoreStart, scoreEnd, n_gram_size):
    resultDf = wordDf[(wordDf['Sentiment'] >= scoreStart) & \
                      (wordDf['Sentiment'] <= scoreEnd)]

    sentences = [sentence.split() for sentence in resultDf['PhraseAdjusted']]
    wordArray = []
    for i in range(0, len(sentences)):
        wordArray += sentences[i]

    counterList = Counter(ngrams(wordArray, n_gram_size)).most_common(80)

    print("\n***N-Gram")

    for i in range(0, len(counterList)):
        print("Occurrences: ", str(counterList[i][1]), end=" ")
        delimiter = ' '
        print("  N-Gram: ", delimiter.join(counterList[i][0]))

    return counterList


# Create two column matrix.
dfSub = df[['Sentiment', 'PhraseAdjusted']]
SCORE_RANGE_START = 4
SCORE_RANGE_END = 4
SIZE = 1
counterList = generateWordList(dfSub, SCORE_RANGE_START, SCORE_RANGE_END, SIZE)

SIZE = 3
counterList = generateWordList(dfSub, SCORE_RANGE_START, SCORE_RANGE_END, SIZE)



# Create DataFrame.
simpleDataSet = {'PhraseAdjusted': ['the sky is blue'], 'Sentiment': [4] }
dfSimple      = pd.DataFrame(simpleDataSet, columns= ['Sentiment',
                                                      'PhraseAdjusted'])
SIZE          = 2
newNGrams   = generateWordList(dfSimple,SCORE_RANGE_START, SCORE_RANGE_END,
                                 SIZE)
print(newNGrams)

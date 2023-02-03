import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

sentence1 = "Despite its fresh perspective, Banks's Charlie's Angels update " \
            + "fails to capture the energy or style that made it the beloved phenomenon it is."

sentence2 = "This 2019 Charlie's Angels is stupefyingly entertaining and " \
            + "hilarious. It is a stylish alternative to the current destructive blockbusters."

reviewSentence = \
  "Parents need to know that this classic 1908 children's novel by L.M. " \
+ "Montgomery remains a perennial favorite thanks to its memorable heroine: " \
+ "irrepressible red-headed orphan Anne Shirley. Anne's adventures are full of " \
+ "amusing (and occasionally mildly dangerous) scrapes, but she's quick to learn " \
+ "from her mistakes and usually has only the best of intentions. Although Anne " \
+ "gets her best friend drunk in one episode (it's an honest mistake), there's very " \
+ "little here that's at all iffy for kids -- though younger readers might get a " \
+ "bit bogged down in the many descriptions of Anne's Prince Edward Island, Canada, " \
+ "home. A sad death may hit some kids hard, but the book's messages about the " \
+ "importance of love, friendship, family, and ambition are worth it."


# sentences = [sentence1, sentence2]
sentences = [reviewSentence]


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


tokenizedList = createTokenizedArray(sentences)


# To get stop words.
nltk.download('stopwords')


# -------------------------------------------------------------
# Create array of words with no punctuation or stop words.
# -------------------------------------------------------------
def removeStopWords(tokenList):
    stopWords = set(stopwords.words('english'))
    print(stopWords) # Exercise 3
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


sentenceArrays = removeStopWords(tokenizedList)
print(sentenceArrays)

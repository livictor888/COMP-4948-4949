
from nltk.tokenize import RegexpTokenizer

sentence1 = "Despite its fresh perspective, Banks's Charlie's Angels update " \
            + "fails to capture the energy or style that made it the beloved phenomenon it is."

sentence2 = "This 2019 Charlie's Angels is stupefyingly entertaining and " \
            + "hilarious. It is a stylish alternative to the current destructive blockbusters."

sentences = [sentence1, sentence2]


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

from nltk.corpus import stopwords


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


sentenceArrays = removeStopWords(tokenizedList)
print(sentenceArrays)


def modifiedStopWords(sentences_lists):
    updated_list = []
    custom_stop_words = ['charlie', 'angels']
    for sentence_array in sentences_lists:
        for word in sentence_array:
            if word not in custom_stop_words:
                updated_list.append(word)
    return updated_list


output = modifiedStopWords(sentenceArrays)
print("The final answer is: ")
print(output)
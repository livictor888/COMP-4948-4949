""" Example 3: Tokenizing and Sequencing Sentences """

from keras.preprocessing.text import Tokenizer
# from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf

sentence1 = "This is not yellow."
sentence2 = "This is a blue moon."
sentence3 = "Hello all."
sentences = [sentence1, sentence2, sentence3]

# Restrict tokenizer to use top 2500 words.
tokenizer = Tokenizer(num_words=2500, lower=True, split=' ')
tokenizer.fit_on_texts(sentences)

# Convert to sequence of integers.
X = tokenizer.texts_to_sequences(sentences)
print(X)

# Showing padded sentences:
paddedX = tf.keras.utils.pad_sequences(X)
print(paddedX)

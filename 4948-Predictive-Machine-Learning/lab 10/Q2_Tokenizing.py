from keras.preprocessing.text import Tokenizer
import tensorflow as tf

sentence1 = "Victor Li is prepared to succeed."
sentence2 = "Victor Li sees opportunity in every challenge."
sentences = [sentence1, sentence2]

# Restrict tokenizer to use top 2500 words.
tokenizer = Tokenizer(num_words=2500, lower=True, split=' ')
tokenizer.fit_on_texts(sentences)

# Convert to sequence of integers.
X = tokenizer.texts_to_sequences(sentences)
print(X)

# Showing padded sentences:
paddedX = tf.keras.utils.pad_sequences(X)
print(paddedX)

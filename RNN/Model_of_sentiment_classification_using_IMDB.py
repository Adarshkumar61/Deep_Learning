# importing required libraries:
import tensorflow as tf
from tensorflow.keras.models import sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequence
from tensorflow.keras.layers import Dense, Embedding, simpleRNN

#laod and preprocess the data:
num_words = 10000 # means loading data with 10k frequent words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = num_words)

pad_seq = 200

x_train = pad_sequence(x_train, pad_seq = pad_seq)
x_test = pad_sequence(x_test, pad_seq = pad_seq)

#building the RNN model:
model = sequential([
    Embedding(input_dim = num_words, output_dim = 32, input_length = pad_seq)
    simpleRNN(units = 32, return_sequences = False),
    Dense(1, activation = 'sigmoid')
])

#compiling the model:
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_split = 0.2)
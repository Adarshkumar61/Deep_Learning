# importing required libraries:
import tensorflow as tf
from tensorflow.keras.models import sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequence
from tensorflow.keras.layers import Dense, Embedding, simpleRNN

#laod and preprocess the data:
num_words = 10000 # means loading data with 10k frequent words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = num_words)

max_len = 200

x_train = pad_sequence(x_train, max_len = max_len)
x_test = pad_sequence(x_test, max_len = max_len)

#building the RNN model:
model = sequential([
    Embedding(input_dim = num_words, output_dim = 32, input_length = max_len), #used Embedding to convert word indexes into word vectors
    simpleRNN(units = 32, return_sequences = False),
    Dense(1, activation = 'sigmoid')
])

#compiling the model:
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_data = (x_test, y_test))
# checking loss and accuracy:
loss, acc = model.evaluate(x_test, y_test)

print(f'test accuracy is: {acc:.4f}')
print(f'test loss is: {loss:.4f}')
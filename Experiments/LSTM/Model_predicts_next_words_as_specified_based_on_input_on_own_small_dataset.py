# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Embedding
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow .keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical

# data = [
#     "I love programming in Python",
#     "Deep learning is fascinating",
#     "LSTM networks are powerful for sequence prediction",
#     "Natural language processing is a key area in AI",
#     "TensorFlow makes building models easier",
#     "Keras is a high-level API for TensorFlow",
#     "Neural networks can learn complex patterns",
#     "Data science combines statistics and programming",
#     "Machine learning is a subset of AI",
#     "Artificial intelligence is transforming industries",
#     "I love robots"
# ]

# # tokenization:
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(data)
# total_words = len(tokenizer.word_index) +1 # num classes

# input_sequences = []
# for line in data:
#     token_list = tokenizer.texts_to_sequences([line])[0]
#     for i in range(1, len(token_list)):
#         n_seq = token_list[:i+1]
#         input_sequences.append(n_seq)

# #pad sequences:
# max_len = max([len(x) for x in input_sequences])
# input_sequences = pad_sequences(input_sequences, max_len=max_len, padding = 'pre')


# # splitting feature and label:
# x = input_sequences[:,:-1] 
# y = input_sequences[:,-1]
# y = to_categorical(y, num_classes = total_words)


# #model building:
# model = Sequential()
# model.add(Embedding(total_words, 50, input_length = max_len-1))
# model.add(LSTM(100))
# model.add(Dense(total_words, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(x, y, epochs = 500, verbose = 0)

# # Function to generate text
# def generate_text(seed_text, next_words):
#     for _ in range(next_words):
#         token_list = tokenizer.texts_to_sequences([seed_text])[0]
#         token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
#         predicted = model.predict(token_list, verbose=0)
#         predicted_word_index = np.argmax(predicted, axis=-1)[0]
#         output_word = tokenizer.index_word[predicted_word_index]
#         seed_text += " " + output_word
#     return seed_text
# # Example usage
# seed_text = "I love"
# generated_text = generate_text(seed_text, next_words=5)
# print(generated_text)




import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utlis import to_categorical

data = [
    # here data will be placed
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
total_words = len(tokenizer.word_index)+1

input_sequence  = []
for line in data:
    token_list = tokenizer.text_to_sequences([line])[0]
    for i in range(1, token_list):
        no_of_seq = token_list[:i+1]
        input_sequence.append(no_of_seq)

max_len = max([len(x) for x in input_sequence])
pad_seq = pad_sequences(input_sequence, maxlen = max_len, padding = 'pre')

x = input_sequence[:,:-1]
y = input_sequence[:,-1]
y = to_categorical(y, num_classes = total_words)

model = Sequential()
model.add(Embedding(total_words, 50, max_len-1))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(total_words, activation = 'softmax'))

model.compile(
    optimizer = 'adam',
    loss = 'mse'
)

model.fit(x,y, epochs = 100, verbose = 0)

# prediciton:

def chatbot(inp_text, next_word):
    for _ in next_word:
        token_list = tokenizer.text_to_sequences([inp_text])[0]
        token_list = pad_sequences([token_list], maxlen = max_len-1, padding = 'pre')

        prediction = model.predict(token_list, verbose = 0)
        predictions = np.argmax(prediction, axis= -1)[0]
        output = tokenizer.index_word([predictions])
        inp_text += ' ' + output
    return inp_text

inp_text = 'i love'
gen_text = chatbot(inp_text, next_word= 5)
print(gen_text)
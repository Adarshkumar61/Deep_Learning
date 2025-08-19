import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow .keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

data = [
    "I love programming in Python",
    "Deep learning is fascinating",
    "LSTM networks are powerful for sequence prediction",
    "Natural language processing is a key area in AI",
    "TensorFlow makes building models easier",
    "Keras is a high-level API for TensorFlow",
    "Neural networks can learn complex patterns",
    "Data science combines statistics and programming",
    "Machine learning is a subset of AI",
    "Artificial intelligence is transforming industries",
    "I love robots"
]

# tokenization:
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
total_words = len(tokenizer.word_index) +1

input_sequences = []
for line in data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_seq = token_list[:i+1]
        input_sequences.append(n_seq)

#pad sequences:


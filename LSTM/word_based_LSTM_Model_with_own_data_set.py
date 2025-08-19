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



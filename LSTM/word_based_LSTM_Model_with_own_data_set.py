import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow .keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


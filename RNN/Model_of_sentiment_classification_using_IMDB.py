# importing required libraries:
import tensorflow as tf
from tensorflow.keras.models import sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequence
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN

#laod and preprocess the data:

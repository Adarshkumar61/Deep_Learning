# importing required libraries:
import tensorflow as tf
from tensorflow.keras.models import sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequence
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN

#laod and preprocess the data:
num_words = 10000 # means loading data with 10k frequent words
(x_train, )
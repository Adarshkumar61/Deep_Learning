import tensorflow as tf
from tf.keras.datasets import cifar10
from tf.keras import layers, models
from tf.keras.utils import to_categorical
from tf.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# data loading:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


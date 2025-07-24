import tensorflow as tf
from tf.keras.datasets import cifar10
from tf.keras import layers, models
from tf.keras.utils import to_categorical
from tf.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# data loading:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train =  x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = models.sequential()

# BLOCK1:

model.add(layers.Conv2D(32, padding = 'same', activation = 'relu', input_shape = (28, 28,3)))
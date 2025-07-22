import tensorflow as tf
from tf.keras import models, layers
from tf.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


x_train = x_train / 255
x_test = x_test / 255

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = models.sequential()

#Block 1:
padding = 2
activation = 3
model.add(layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu', input_shape = (28, 28, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Dropout(0.2))


#Block 2:
model.add(layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Dropout(0.2))

#Block3:
model.add(layers.Conv2D(128, (3,3), padding = 'same', activation = 'relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3,3), padding = 'same', activation = 'relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Dropout(0.2))


# Connecting All Neurons together:
model.add(layers.Flatten())
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(10, activation = 'softmax'))
model.add(layers.Dropout(0.25))



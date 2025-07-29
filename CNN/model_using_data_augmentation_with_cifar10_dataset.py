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

#converted labels to one hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#data augmentation:
data = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
)

data.fit(x_train)

model = models.Sequential()

# BLOCK1:

model.add(layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu', input_shape = (28, 28,3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size = (2,2)))
model.add(layers.Dropout(0.2))

# BLOCK2:
model.add(layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size = (2,2)))
model.add(layers.Dropout(0.2))


# BLOCK3:
model.add(layers.Conv2D(128, (3,3), padding = 'same', activation = 'relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3,3), padding = 'same', activation = 'relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size = (2,2)))
model.add(layers.Dropout(0.2))


model.add(layers.Flatten())
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))
model.add(layers.Dropout(0.25))

model.complie(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

history = model.fit()
import tensorflow as tf
from tensorflow.keras import layers, models, input
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(1, 28, 28, -1).astype('float32') /255
x_test = x_test.reshape(1, 28, 28, -1).astype('float32') / 255

# model (using functional API):
inputs = input(shape = (28, 28, 1))
x = layers.Conv2D(32, (3, 3), activation = 'relu', name = 'conv1')(inputs)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Conv2D(64, (3,3), activation = 'relu')(x)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation = 'relu')(x)
outputs = layers.Dense(10, activation = 'softmax')(x)

model = models.model(inputs = inputs, outputs = outputs)

#compile:
model.compile(
    optimizer = 'adam',
    loss = 'sparse categorical crossentropy',
    metrics = ['accuracy']
)


#fitting the model:
model.fit(x_train, y_train, ephocs = 5, validation_data = (x_test, y_test))

# now creating an activation model for specific Conv2D

layer_outputs = [layer.output for layer in model.layers if isinstance(layer, layers.Conv2D)]
activation_model = models.model(input = model.input, output = layer_outputs)
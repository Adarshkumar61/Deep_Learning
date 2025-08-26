import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# laod data :
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#reshape and resize:

x_train = x_train.reshape(-1, 28,28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
# explaination:
# .reshape(-1, 28, 28, 1):
# Original shape = (60000, 28, 28)
#  channel dimension at the end:
# 28 x 28 x 1 → 1 means grayscale (not RGB)
# -1 tells Python to figure out the batch size automatically
#  .astype('float32') / 255:
# Converts pixel values (0–255) → float (0.0 to 1.0)
# Normalization helps model learn faster and perform better

# Build the model:
model = models.sequential([
    layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28, 28, 1)),
    # Learns 32 types of features like edges, corners, etc.
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation = 'relu'),
    # Another conv layer but now with 64 filters
  # Learns more complex features (patterns, shapes)
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    # Converts 2D matrix into a 1D array
    layers.Dense(128, activation = 'relu'),
    layers.Dense(10, activation = 'softmax')
])

#compile the model:

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)


# training :
history = model.fit(x_train, y_train, ephocs = 5, valdation_data = (x_test, y_test))


# evaluation:

test_loss, test_acc = model.evaluate(x_test, y_test)

print(f'Test accuracy is: {test_acc: .2f}')
# test accuracy is alomost 99%
# which is good

pred_on_x_test = model.predict(x_test)

img_index = 5

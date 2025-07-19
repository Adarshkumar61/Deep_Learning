import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# laod data :
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#reshape and resize:

x_train = x_train.reshape(-1, 28,28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# Build the model:
model = models.sequential([
    layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28, 28, 1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
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
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

model = models.Sequential()

#Block 1:
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


#compiling the Model:

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(x_train, y_train, ephocs = 10, validation_data = (x_test, y_test))

loss_test, acc_test = model.evaluate(x_test, y_test)

print(f'accuracy of the model is: {acc_test}')
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

img_index = 4
img = x_test[img_index]

actual = y_test[img_index]

prediction = model.predict(np.expand_dims(img, axis= 0))
predicted = np.argmax(prediction)

plt.imshow(img)
plt.title(f'predicted image: {predicted}, actual: {actual}')
plt.axis('off')
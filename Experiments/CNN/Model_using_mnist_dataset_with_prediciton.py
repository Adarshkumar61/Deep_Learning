import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tf.keras.datasets import mnist

#loading the Dataset:
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#checking the size of data:

x_train.shape # 60000 ,28,28 means there are 60k images and the dimension of image is 28 * 28 
x_test.shape # 10000, 28, 28 means there are 10k images in training and size of the image is 28* 28

# displaying an image:
plt.imshow(x_train[2])
#printing also the label:
print(y_train[2]) #it will print the label which is 4 

#now we will do the scaling the values:
# bw 0 to 1:
x_train = x_train / 255.0 
x_test = x_test / 255.0
x_train = x_train.reshape(-1, 28,28, 1).astype('float32')
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')
# -1 means it will automatically take the batch size


#building a Neural Network:

model = Sequential ([
    Conv2D(32, (3,3), activation = 'relu', input_shape = (28, 28, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation = 'relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation = 'relu'),
    Dense(10, activation = 'softmax')
])

#compiling the model :
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

#training the Neural Network:
model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_data = (x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
print('the test accuracy is: '.title(),test_acc)
import tensorflow as tf
import open_cv as cv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tensorflow import keras
from keras.datasets import datasets

#loading the Dataset:
(X_train, y_train), (X_test, y_test) = mnist.dataset()

#checking the size of data:

X_train.shape # 60000 ,28,28 means there are 60k images and the dimension of image is 28 * 28 
X_test.shape # 10000, 28, 28 means there are 10k images in training and size of the image is 28* 28

# displaying an image:
plt.imshow(X_train[2])
plt.show()
#printing also the label:
print(y_train[2]) #it will print the label which is 4 

#now we will do the scaling the values:
# bw 0 to 1:
X_train = X_train/255
X_test = X_test/255 

#building a Neural Network:

model = keras.sequential ([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(activation = 'relu'),
    keras.layers.Dense(activation = 'relu'),
    keras.layers.Dense(activation = 'sigmoid')
])

#compiling the model :
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

#training the Neural Network:

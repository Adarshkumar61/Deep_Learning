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

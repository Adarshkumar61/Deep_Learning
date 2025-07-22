import tensorflow as tf
from tf.keras import models, layers
from tf.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


x_train = x_train / 255
y_train = y_train / 255

import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

#dataset loading:

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
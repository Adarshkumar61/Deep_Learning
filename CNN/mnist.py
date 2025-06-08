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
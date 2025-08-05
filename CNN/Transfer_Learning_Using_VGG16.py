import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam


# declearling the variables:
img_size = 224
batch_size = 32
epochs = 15


#Data Augmentation:
datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split = 0.2,
    zoom_range = 0.1,
    rotation_range = 15,
    horizontal_flip = True
)


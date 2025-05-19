import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

#dataset loading:

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

#giving class names to images:
class_name= ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

#plotting:
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap= plt.cm.binary)
    plt.xlabel(class_name[train_labels[i][0]])
    plt.show()

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
from tensorflow.keras.preprocessing import preprocessor_input

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

#data split into training and validation:

train_gen = datagen.flow_from_directory(
    '/contentt/flowers',
    target_size = (img_size, img_size),
    batch_size = batch_size,
    subset = 'training',
    class_mode = 'categorical'
)

val_gen = datagen.flow_from_directory(
    '/contentt/flowers',
    target_size = (img_size, img_size),
    batch_size = batch_size,
    subset = 'testing',
    class_mode = 'categorical'
)

vgg = VGG16(include_top = False, weights = 'imagenet', input_shape = (img_size, img_size, 3))


#frezzing training:
for layer in vgg.layers:
    layer.trainable = False
#adding top layer:
x = Flatten()(vgg.output)
x = Dense(128, activation = 'relu')(x)
x = Dropout(0.2)(x)
output = Dense(train_gen.num_classes, activation = 'softmax')(x)


# calling the model:
model = Model(vgg.input, output = output)


# compling the model;
model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)


history = model.fit(train_gen, epochs = epochs, validation_data = val_gen)

# prediction:

img_path = ' '
img = image.load_img(img_path, target_size = (img_size, img_size))
img_in_arr = image.to_array(img)

exp_dim = np.expand_dims(img_in_arr, 0)
exp_dim = preprocess_input(exp_dim)

prediction = model.predict(exp_dim)
prediction_final = np.argmax(prediction[0])

class_label = list(train_gen.num_indices.keys())
print(f'Prediciton is: {class_label[prediction_final]}')

print(f'actual label is: {class_label[prediction_final]}')

plt.imshow(img_path)
plt.title(f'acutal class: {class_label[prediction_final]}, predicted: {class_label[prediction_final]}')
plt.axis('off')
plt.show()
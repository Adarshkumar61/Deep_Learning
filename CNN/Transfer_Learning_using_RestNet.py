import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import numpy as np
import matplotlib.pyplot as plt 


#here we will import the dataset from kaggle
# after that:

#Now we will goona split into training and val
img_size = 224
batch_size = 32
datagen = ImageDataGenerator(
    rescale = 1.225,
    validation_split = 0.2,
    rotation_range = 15,
    zoom_range = 0.1,
    horizontal_flip = True
)

train_gen = datagen.flow_from_directory(
    # link of the file,
    target_size = (img_size, img_size),
    batch_size = batch_size,
    class_mode = 'categorical',
    subset = 'training'
)

val_gen = datagen.flow_from_directory(
    '/content/flowers',
    target_size = (img_size, img_size),
    batch_size = batch_size,
    class_mode = 'categorical',
    subset = 'validation'
)

# Now We will Load the Model:
res_net = ResNet50(include_top = False, weight = 'imagenet', input_shape = (img_size, img_size, 3))

# freezing the res_net layer:
for layer in res_net.layers:
    layer.trainable = False
    
    #Now we will goona add custom TOP layer:
    x = res_net.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation = 'relu')(x)
    x = Dropout(0.2)(x)
    prediction = Dense(train_gen.num_classes, activation = 'softmax')(x)
    
    model = Model(res_net.input, output = prediction)
    
    # now we gonna compile the model:
    model.compile(
        optimizer = Adam(), 
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )
    
    # we will see the summary :
    model.summary()
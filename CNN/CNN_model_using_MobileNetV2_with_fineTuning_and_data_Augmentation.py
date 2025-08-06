import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import preprocess_input
from tensorflow.keras.preprocessing import image


# Paths to your data
path = ' '

# Data augmentation for training
datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function = preprocess_input,
    validation_split = 0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data generators
train_generator = datagen.flow_from_directory(
    path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = datagen.flow_from_directory(
    path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load MobileNetV2 base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model (initial training with frozen base)
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Fine-tuning: Unfreeze some layers of the base model
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training (fine-tuning)
history_finetune = model.fit(
    train_generator,
    epochs=10,
    validation_data = val_generator
)

# Save the model
model.save('mobilenetv2_finetuned.h5')


#prediction:

img_path = ''
img = image.load_img(img_path, target_size  = (224, 224))
img_to_arr = image.to_array(img)

expand_dim = np.expand_dims(img_to_arr, 0)

expand_dim = preprocess_input(expand_dim)

pred = model.predict(expand_dim)
prediction = np.argmax(pred[0])

class_label = list(train_generator.num_indices.keys())
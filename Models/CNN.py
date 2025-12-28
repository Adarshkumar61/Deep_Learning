# ============================
# CIFAR-10 CNN Classification
# ============================

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load CIFAR-10 Dataset
# ----------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ----------------------------
# Data Augmentation
# ----------------------------
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

datagen.fit(x_train)

# ----------------------------
# CNN Model Architecture
# ----------------------------
model = models.Sequential()

# Block 1
model.add(layers.Conv2D(32, (3, 3), padding="same",
                        activation="relu", input_shape=(32, 32, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Block 2
model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Block 3
model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Fully Connected Layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation="softmax"))

# ----------------------------
# Compile Model
# ----------------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------
# Train Model
# ----------------------------
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=15,
    validation_data=(x_test, y_test)
)

# ----------------------------
# Save & Load Model
# ----------------------------
model.save("cnn_cifar10_model.h5")
print("✅ Model saved successfully")

model = tf.keras.models.load_model("cnn_cifar10_model.h5")
print("✅ Model loaded successfully")

# ----------------------------
# Training Visualization
# ----------------------------
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

# ----------------------------
# Single Image Prediction
# ----------------------------
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

img_index = 1
image = x_test[img_index]
actual_label = class_names[np.argmax(y_test[img_index])]

prediction = model.predict(np.expand_dims(image, axis=0))
predicted_label = class_names[np.argmax(prediction)]

plt.imshow(image)
plt.axis("off")
plt.title(f"Predicted: {predicted_label} | Actual: {actual_label}")
plt.show()



# import tensorflow as tf
# print("TensorFlow version:", tf.__version__)

# import tensorflow as tf
# print(tf.__version__)

import tensorflow as tf
from tensorflow.keras import layers, models

# Define the CNN model
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()

    # Convolutional layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional layer 3
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flatten the output
    model.add(layers.Flatten())

    # Fully connected layer
    model.add(layers.Dense(64, activation='relu'))

    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Example usage
if __name__ == "__main__":
    input_shape = (28, 28, 1)  # Example for grayscale images of size 28x28
    num_classes = 10  # Example for 10 classes (e.g., digits 0-9)

    model = create_cnn_model(input_shape, num_classes) 
    model.summary()

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.Accuracy()])
    # model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    # train_images
    # train_labels
    # test_images
    #  test_labels
    #these are used when we add data and divide them into test and train
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
    
    #changing pixel values bw 0 and 1:
    
    train_images = train_images/255.0
    test_images = test_images/255.0
    
    # checking the shapes
    train_images.shape #output : (50000, 32, 32, 3)
    test_images.shape #output : (10000, 32, 32, 3)
    
    
    #Buidling a Convolution Neural Network:
    
    cnn_model = models.sequential([
        layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32, 3,3)),
        layers.MaxPooing((2,2)),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.MaxPooling((2,2)),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.Flatten(),
        layers.Dense(64, activation = 'relu'),
        layers.Dense(64, activation = 'softmax')
    ])
    
    #compiling the model:
    cnn_model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    #display the summary:
    cnn_model.summary()
    
    # Training the Model:
    cnn_training = cnn_model.fit(train_images, train_labels, epochs = 10, validation_data = (test_images, test_labels))
    
    
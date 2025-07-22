import tensorflow as tf #not downloaded yet but code is working 
from tf.keras.layers import MaxPooling2D, Conv2D, Dense, Flatten
from tf.keras.modes import sequential
import matplotlib.pyplot as plt
from tf.keras.datasets import cifar10

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
    
    cnn_model = sequential([
        #1st convolution
        Conv2D(32, (3,3), activation = 'relu', input_shape = (32, 32, 3)),
        MaxPooling2D((2,2)),
        # it will downsample the image to focus on most important features
        
        Conv2D(64, (3,3), activation = 'relu'),
        # 2nd covo layer increase feature richness 
        MaxPooling2D((2,2)),
        # reduce spatial sizes to make training more efficient
        Conv2D(128, (3,3), activation = 'relu'),
        Flatten(), # transforms 2D features into 1D 
        
        Dense(128, activation = 'relu'),
        # Dense is also caled fully connected layer.
        # this is hidden layer with 12 neurons
        #each neuron takes features from CNN and applies relu(non-linearity)
        
        Dense(10, activation = 'softmax')
        # this is output layer.
        # the softmax fn turns the output into probabilities and all sum to 1.
        # like this :  [0.01, 0.90, 0.03, 0.01, 0.01, 0.00, 0.01, 0.00, 0.02, 0.01] : (1)
        
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
    
    #Evaluating the model:
     
    cnn_test_loss, cnn_test_acc = cnn_model.evaluate(test_images, test_labels, verbose = 2)                                                                                                             
    
    #checking accuracy:
    print(f"Accuracy of test data is: {cnn_test_acc}")  # accuracy is :97%
 
    
    #plotting Training and Validation Accuracy:
    plt.figure(figsize=(10,10))
    plt.plot(cnn_training.history['accuracy'], label = 'Traning accuracy')
    plt.plot(cnn_training.history['val_accuracy'], label = 'Validation Accuracy')
    plt.legend()
    plt.show()
    
    # Plotting Training and Validaiton Loss:
    plt.figure(figsize=(10,10))
    plt.plot(cnn_training.history['loss'], label = 'training_loss')
    plt.plot(cnn_model.history['val_loss'], label = 'validation_loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend() 
    plt.show()
    # CONCLUSION:
    # Created a model which tells what the object is (Basically identify Objects)
    # we have used relu to identify complex graphs.
    cnn_model.predict(test_images)
    
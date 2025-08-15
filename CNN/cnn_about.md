1. CNN : convolutional Neural Network
CNN is mostly used for image , face, recognition, object recognition etc..
it uses neurons to predict output:

CNN architecture:

1. Input Layer:
image is represented as a matrix:
for a RGB image it has (h * w * 3(channel))

2. Convoution:
here fiters are used.
filter(Kernal) are small matrix (Eg. 3*3) slides over the image 
perform element wise operation multiplication + addition 

this process extract  Important features like edges, line, texture etc.

3. ReLu: (Non_linearity):
used after convolution to add non-linearity to:
remove negative values and keep only important features.

4. Pooling:
Pooling is used to downsample (reduce) the size of image.
to keep only important features

5. Flatten:
after convolution + relu + pooling flatten is used:
 we flatten the output into 1D vector to give fully connected layer

6. Fully connected dense layer :
it joins every input to every output neurons and gives final prediction

7. Output :
for classification:
we use Sigmoid: like (cat vs dog), (snake vs fish) etc..

for regression:
we use Softmax: lke (human , cat, dog, car) etc..

Summary

Image → Convolution → ReLU → Pooling → Conv → ReLU → Pooling → Flatten → Dense → Output

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import sequential
from tensorflow.keras.layers import LSTM, Dense

pred_len = 4 #here you can specify how many numbers you want to predict
seq_len  = 5 # how many input sequence we want to give

data = np.array([i for i in range(100)]) 

x, y = [], []
for i in range(len(data)-seq_len -pred_len+1): # here +1 is because we want to include last sequence also
    x.append(data[i:i+seq_len]) # x is input sequence and append first 5 numbers
    y.append(data[i+seq_len:i+seq_len+pred_len]) # y is output sequence and append next 4 numbers after input sequence

x = np.array(x)
y = np.array(y)

x = x.reshape(x.shape[0], x.shape[1], 1) # reshaping in 3d array as lstm take input in 3d array

model = sequential()
model.add(LSTM(50, activation = 'relu', return_sequences = False, input_shape = (seq_len, 1))) # return_sequence = False because we are not stacking LSTM layer
model.add(Dense(pred_len))

model.compile(
    optimizer = 'adam',
    loss = 'mse')

model.fit(x, y, epochs = 200, verbose = 0)

#prediction:

input_seqq = np.array([23, 24, 25, 26, 27])
input_seqq = input_seqq.reshape(1, seq_len, 1)

prediction = model.predict(input_seqq, verbose = 0)
# prediction = prediction.flatten() # it will convert 2d array to 1d array
prediciton = prediction.round().astype(int)
print(f'Predicted next {pred_len} numbers are: {prediction[0]}')
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

seq_len = 3

data = np.array([i for i in range(50)])

x, y = [], []
for i in range(len(data)- seq_len):
    x.appned(data[i:i+seq_len])
    y.append(data[i+seq_len])


    x= np.array(x)
    y = np.array(y)

    x = x.reshape(x.shape[0], x.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, activaiton = 'relu', input_shape= (seq_len, 1)))
    model.add(Dense(1))

    model.compile(
        optimizer = 'adam',
        loss = 'mse'
    )

    model.fit(x, y, epochs = 200, verbose = 1)
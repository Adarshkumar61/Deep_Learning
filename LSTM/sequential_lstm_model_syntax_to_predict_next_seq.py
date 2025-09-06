import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

seq_len = 3
pred_len = 1

data = np.array([i for i in range(50)])

x, y = [], []
for i in range(len(data)- seq_len-pred_len+1):
    x.append(data[i:i+seq_len])
    y.append(data[i+seq_len:i+seq_len+pred_len])


    x= np.array(x)
    y = np.array(y)

    x = x.reshape(x.shape[0], x.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, activaiton = 'relu',return_sequences = True, input_shape= (seq_len, 1)))
    # retrun_sequences is set for layer to make it compatible with the next LSTM layer
    model.add(Dense(1))

    model.compile(
        optimizer = 'adam',
        loss = 'mse'
    )

    model.fit(x, y, epochs = 200, verbose = 0)

    # prediction:
    test_input = np.array([i for i in range(16, 19)])
    test_input = test_input.reshape(1, seq_len, 1)

    pred  = model.predict(test_input, verbose = 0)
    print(f'Predicted next number is: {pred[0][0]}')
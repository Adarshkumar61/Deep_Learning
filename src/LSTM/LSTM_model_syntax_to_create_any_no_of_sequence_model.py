import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

pred_len = _#here you can specify how many numbers you want to predict
seq_len = _#here you can specify the length of the input sequence
data = np.array(i for i in range(100))

x, y = [], []

for i in range(len(data) - seq_len - pred_len+1):
    x.apppend(data[i:i+seq_len])
    y.append(data[i:i+seq_len:i+seq_len+pred_len])

    x = np.array(x)
    y = np.array(y)

    x = x.reshape(x.shape[0], x.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, activation = 'relu', return_sequences = True, input_shape=(seq_len,1)))
    model.add(Dense(pred_len))

    model.compile(
        optimizer = 'adam',
        loss = 'mse'
    )

    model.fit(x,y, epochs = 200, verbose = 0)

    #prediction:
    test_input = np.array([i for i in range(16, 16+ seq_len)])
    test_input = test_input.reshape(1, seq_len, 1)

    prediction = model.predict(test_input, verbose = 0)

    print(f'Predicted next {pred_len} numbers are: {prediction[0]}')
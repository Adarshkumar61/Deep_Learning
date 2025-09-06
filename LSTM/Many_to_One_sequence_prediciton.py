import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
seq_len = 3
pred_len = 1

data = np.array([i for i in range(50)])

x, y = [], []

for i in range(len(data)-seq_len-pred_len+1):
    x.append(data[i:i+seq_len])
    y.append(data[i+seq_len:i+seq_len+pred_len])

x = np.array(x)
y = np.array(y)

x = x.reshape(x.shape[0], x.shape[1], 1)

model = Sequential()
model.add(LSTM(50, activation = 'relu', return_sequences = False, input_shape = (seq_len, 1)))
model.add(Dense(pred_len))

model.compile(
    optimizer = 'adam',
    loss = 'mse'
)

model.fit(x, y, epochs = 200, verbose = 0)

# prediciton:

test_input = np.array([i for i in range(16, 19)])
test_input = test_input.reshape(1, seq_len, 1)

prediciton = model.predict(test_input, verbose = 0)
prediciton = prediciton.round().astype(int)
print(f'Predicted next {pred_len} number is: {prediciton[0][0]}')   
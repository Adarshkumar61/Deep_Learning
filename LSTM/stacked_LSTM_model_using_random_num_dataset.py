import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np

x =np.random.rand(1000, 10, 1)
y = np.random.rand(1000, 1)

model = Sequential([
    LSTM(50, return_sequences= True, input_shape=(10, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences = False),
    Dropout(0.2),
    Dense(1)
])

model.compile(
    optimizer = 'adam',
    loss = ',mse'
)



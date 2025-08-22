import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv('Apple.csv')
df.isnull().sum()

price = df['close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range= (0,1))
scaled_data = scaler.fit_transform(price)

x , y = [], []
seq_len = 60
for i in range(seq_len, len(scaled_data)):
    x.append(scaled_data[i-seq_len:i, 0])
    y.append(scaled_data[i, 0])

x, y = np.array(x), np.array(y)

x = np.reshape(x, (x.shape[0], x.shape[1], 1))

split = int(0.8 * len(X))
X_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    Bidirectional(LSTM(50, return_sequences = True), input_shape=  (x.shape[1], 1)),
    Dropout(0.2),
    Bidirectional(LSTM(50, return_sequences = False)),
    Dropout(0.2),
    Dense(1)
])

model.compile(
    optimizer = 'adam',
    loss = 'mse'
)

history = model.fit(X_train, y_train, epochs = 10, batch_size = 32, verbose = 0)

model.evaluate(x_test, y_test, verbose=0)
print("Model evaluation complete.")

prediction = model.predict(x)
prediction = scaler.inverse_transform(prediction.reshape(-1, 1))

real = scaler.inverse_transform(x_test.reshape(-1, 1))

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(real, color='blue', label='Real Stock Price')
plt.plot(prediction, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
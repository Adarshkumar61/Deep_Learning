# ==========================================
# Apple Stock Price Prediction using Bi-LSTM
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("Apple.csv")
df = df[['close']]  # use closing price only
df.dropna(inplace=True)

prices = df.values

# -------------------------------
# Train / Test Split
# -------------------------------
train_size = int(len(prices) * 0.8)
train_data = prices[:train_size]
test_data = prices[train_size:]

# -------------------------------
# Scaling (NO DATA LEAKAGE)
# -------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# -------------------------------
# Create Sequences
# -------------------------------
SEQ_LEN = 60

def create_sequences(data, seq_len):
    x, y = [], []
    for i in range(seq_len, len(data)):
        x.append(data[i - seq_len:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

x_train, y_train = create_sequences(train_scaled, SEQ_LEN)
x_test, y_test = create_sequences(test_scaled, SEQ_LEN)

# Reshape for LSTM
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# -------------------------------
# Build Bi-LSTM Model
# -------------------------------
model = Sequential([
    Bidirectional(LSTM(50, return_sequences=True),
                  input_shape=(SEQ_LEN, 1)),
    Dropout(0.2),

    Bidirectional(LSTM(50)),
    Dropout(0.2),

    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

# -------------------------------
# Train Model
# -------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=[early_stop],
    verbose=0
)

# -------------------------------
# Evaluation
# -------------------------------
model.evaluate(x_test, y_test, verbose=0)
print("✅ Model evaluation complete")

# -------------------------------
# Prediction
# -------------------------------
predicted = model.predict(x_test, verbose=0)
predicted_prices = scaler.inverse_transform(predicted)

real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# -------------------------------
# Visualization
# -------------------------------
plt.figure(figsize=(12, 6))
plt.plot(real_prices, label="Actual Stock Price", color="blue")
plt.plot(predicted_prices, label="Predicted Stock Price", color="red")
plt.title("Apple Stock Price Prediction using Bi-LSTM")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(True)
plt.show()

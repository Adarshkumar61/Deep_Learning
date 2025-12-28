# =========================================
# LSTM Multi-step Time Series Prediction
# =========================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# -------------------------------
# Configuration
# -------------------------------
SEQ_LEN = 5      # input sequence length
PRED_LEN = 4     # number of future steps to predict

# -------------------------------
# Generate Data
# -------------------------------
data = np.array([i for i in range(100)], dtype=np.float32)

x, y = [], []

for i in range(len(data) - SEQ_LEN - PRED_LEN + 1):
    x.append(data[i:i + SEQ_LEN])
    y.append(data[i + SEQ_LEN:i + SEQ_LEN + PRED_LEN])

x = np.array(x)
y = np.array(y)

# Reshape for LSTM: (samples, timesteps, features)
x = x.reshape(x.shape[0], x.shape[1], 1)

# -------------------------------
# Train / Test Split
# -------------------------------
split = int(0.8 * len(x))
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

# -------------------------------
# Build LSTM Model
# -------------------------------
model = Sequential([
    LSTM(50, input_shape=(SEQ_LEN, 1)),
    Dense(PRED_LEN)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

# -------------------------------
# Train Model
# -------------------------------
history = model.fit(
    x_train,
    y_train,
    epochs=200,
    batch_size=16,
    validation_data=(x_test, y_test),
    verbose=0
)

# -------------------------------
# Plot Training Loss
# -------------------------------
plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("LSTM Training Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()

# -------------------------------
# Prediction
# -------------------------------
input_seq = np.array([23, 24, 25, 26, 27], dtype=np.float32)
input_seq = input_seq.reshape(1, SEQ_LEN, 1)

prediction = model.predict(input_seq, verbose=0)
prediction = np.round(prediction[0]).astype(int)

print(f"📈 Input sequence      : {input_seq.flatten()}")
print(f"🔮 Predicted next {PRED_LEN} values: {prediction}")

# -------------------------------
# Visualization of Prediction
# -------------------------------
plt.figure(figsize=(8, 4))

# Plot input sequence
plt.plot(range(SEQ_LEN), input_seq.flatten(), marker="o", label="Input Sequence")

# Plot predicted values
plt.plot(
    range(SEQ_LEN, SEQ_LEN + PRED_LEN),
    prediction,
    marker="o",
    linestyle="--",
    label="Predicted Values"
)

plt.title("Multi-step Time Series Prediction using LSTM")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

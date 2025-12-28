"""
LSTM model for multi-step time series prediction.

This script demonstrates how LSTM can learn sequential
patterns and predict multiple future time steps.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import os


def run():
    print("\n[INFO] Starting LSTM time series prediction...")

    # -------------------------------
    # Configuration
    # -------------------------------
    SEQ_LEN = 5
    PRED_LEN = 4

    # -------------------------------
    # Generate Synthetic Data
    # -------------------------------
    data = np.array([i for i in range(100)], dtype=np.float32)

    x, y = [], []
    for i in range(len(data) - SEQ_LEN - PRED_LEN + 1):
        x.append(data[i:i + SEQ_LEN])
        y.append(data[i + SEQ_LEN:i + SEQ_LEN + PRED_LEN])

    x = np.array(x)
    y = np.array(y)

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
    print("[INFO] Training LSTM model...")
    history = model.fit(
        x_train,
        y_train,
        epochs=200,
        batch_size=16,
        validation_data=(x_test, y_test),
        verbose=0
    )

    # -------------------------------
    # Ensure outputs directory
    # -------------------------------
    os.makedirs("outputs", exist_ok=True)

    # -------------------------------
    # Save Model
    # -------------------------------
    model_path = "outputs/lstm_timeseries_model.h5"
    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}")

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

    loss_plot_path = "outputs/lstm_training_loss.png"
    plt.savefig(loss_plot_path)
    plt.close()

    print(f"[INFO] Training loss plot saved to {loss_plot_path}")

    # -------------------------------
    # Prediction
    # -------------------------------
    input_seq = np.array([23, 24, 25, 26, 27], dtype=np.float32)
    input_seq = input_seq.reshape(1, SEQ_LEN, 1)

    prediction = model.predict(input_seq, verbose=0)
    prediction = np.round(prediction[0]).astype(int)

    print(f"[INFO] Input sequence       : {input_seq.flatten()}")
    print(f"[INFO] Predicted next {PRED_LEN} values : {prediction}")

    # -------------------------------
    # Visualization of Prediction
    # -------------------------------
    plt.figure(figsize=(8, 4))

    plt.plot(
        range(SEQ_LEN),
        input_seq.flatten(),
        marker="o",
        label="Input Sequence"
    )

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

    pred_plot_path = "outputs/lstm_prediction.png"
    plt.savefig(pred_plot_path)
    plt.close()

    print(f"[INFO] Prediction plot saved to {pred_plot_path}")
    print("[SUCCESS] LSTM execution completed.\n")


if __name__ == "__main__":
    run()

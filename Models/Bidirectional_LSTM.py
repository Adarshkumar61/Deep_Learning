"""
Bidirectional LSTM model for stock price prediction.

This script predicts Apple stock closing prices using
a Bidirectional LSTM with proper scaling and evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler


def run():
    print("\n[INFO] Starting Bidirectional LSTM stock prediction...")

    # -------------------------------
    # Load Dataset
    # -------------------------------
    data_path = "data/Apple.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Apple.csv not found. Place it inside the 'data/' folder."
        )

    df = pd.read_csv(data_path)
    df = df[['close']]
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
    print("[INFO] Training Bi-LSTM model...")

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
    # Ensure outputs directory
    # -------------------------------
    os.makedirs("outputs", exist_ok=True)

    # -------------------------------
    # Save Model
    # -------------------------------
    model_path = "outputs/bilstm_stock_model.h5"
    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}")

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

    plot_path = "outputs/bilstm_stock_prediction.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"[INFO] Prediction plot saved to {plot_path}")
    print("[SUCCESS] Bidirectional LSTM execution completed.\n")


if __name__ == "__main__":
    run()

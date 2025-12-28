"""
Bidirectional LSTM stock price prediction using real market data.
Designed to be executed via main.py using run().
"""

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from datetime import date, timedelta
import os


def run():
    print("\n[INFO] Running Bidirectional LSTM Stock Prediction")

    # -------------------------------
    # Download Stock Data
    # -------------------------------
    end_date = date.today()
    start_date = end_date - timedelta(days=5000)

    df = yf.download("NFLX", start=start_date, end=end_date, progress=False)
    df = df[["Close"]]
    df.dropna(inplace=True)

    # -------------------------------
    # Scale Data
    # -------------------------------
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

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

    x, y = create_sequences(scaled_data, SEQ_LEN)
    x = x.reshape(x.shape[0], x.shape[1], 1)

    # -------------------------------
    # Train / Test Split (NO SHUFFLE)
    # -------------------------------
    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    # -------------------------------
    # Build Bi-LSTM Model
    # -------------------------------
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True),
                      input_shape=(SEQ_LEN, 1)),
        Bidirectional(LSTM(64)),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.summary()

    # -------------------------------
    # Train Model
    # -------------------------------
    print("[INFO] Training model...")
    model.fit(
        x_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_data=(x_test, y_test),
        verbose=1
    )

    # -------------------------------
    # Prediction
    # -------------------------------
    predicted = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted)
    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    # -------------------------------
    # Save Outputs
    # -------------------------------
    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(real_prices, label="Actual Price")
    plt.plot(predicted_prices, label="Predicted Price")
    plt.title("NFLX Stock Price Prediction (Bi-LSTM)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    plot_path = "outputs/bilstm_nflx_prediction.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"[SUCCESS] Prediction plot saved to {plot_path}\n")


if __name__ == "__main__":
    run()

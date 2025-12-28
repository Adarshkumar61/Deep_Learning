"""
RNN model for IMDB sentiment analysis.

This script trains a SimpleRNN on the IMDB dataset,
evaluates performance, and saves training results.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
import matplotlib.pyplot as plt
import os


def run():
    print("\n[INFO] Loading IMDB dataset...")

    # -------------------------------
    # Load & Preprocess Dataset
    # -------------------------------
    num_words = 10000     # Top 10,000 words
    max_len = 200         # Max review length

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

    x_train = pad_sequences(
        x_train, maxlen=max_len, padding="post", truncating="post"
    )
    x_test = pad_sequences(
        x_test, maxlen=max_len, padding="post", truncating="post"
    )

    # -------------------------------
    # Build RNN Model
    # -------------------------------
    model = Sequential([
        Embedding(input_dim=num_words, output_dim=32, input_length=max_len),
        SimpleRNN(units=32),
        Dense(1, activation="sigmoid")
    ])

    # -------------------------------
    # Compile Model
    # -------------------------------
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # -------------------------------
    # Train Model
    # -------------------------------
    print("\n[INFO] Training RNN model...")
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_data=(x_test, y_test),
        verbose=1
    )

    # -------------------------------
    # Evaluate Model
    # -------------------------------
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[INFO] Test Accuracy: {acc:.4f}")
    print(f"[INFO] Test Loss: {loss:.4f}")

    # -------------------------------
    # Ensure outputs directory
    # -------------------------------
    os.makedirs("outputs", exist_ok=True)

    # -------------------------------
    # Save Model
    # -------------------------------
    model_path = "outputs/rnn_imdb_model.h5"
    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}")

    # -------------------------------
    # Training Visualization
    # -------------------------------
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("RNN Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("RNN Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plot_path = "outputs/rnn_training_curves.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"[INFO] Training plots saved to {plot_path}")
    print("[SUCCESS] RNN execution completed.\n")


if __name__ == "__main__":
    run()

# ===============================
# IMDB Sentiment Analysis using RNN
# ===============================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
import matplotlib.pyplot as plt

# -------------------------------
# Load & Preprocess Dataset
# -------------------------------
num_words = 10000  # Use top 10,000 most frequent words
max_len = 200      # Max length of each review

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# Pad sequences so that all reviews have same length
x_train = pad_sequences(x_train, maxlen=max_len, padding="post", truncating="post")
x_test = pad_sequences(x_test, maxlen=max_len, padding="post", truncating="post")

# -------------------------------
# Build RNN Model
# -------------------------------
model = Sequential([
    # Converts word indices into dense word embeddings
    Embedding(input_dim=num_words, output_dim=32, input_length=max_len),

    # Simple RNN layer to capture sequential patterns in text
    SimpleRNN(units=32),

    # Output layer for binary classification (positive / negative)
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
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# -------------------------------
# Evaluate Model
# -------------------------------
loss, acc = model.evaluate(x_test, y_test, verbose=0)

print(f"✅ Test Accuracy: {acc:.4f}")
print(f"✅ Test Loss: {loss:.4f}")

# -------------------------------
# Training Visualization
# -------------------------------
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

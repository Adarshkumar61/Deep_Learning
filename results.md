📈 Results & Observations

The following results summarize the performance and behavior of each model.
The focus is on learning patterns, stability, and limitations, rather than maximizing benchmark scores.

🖼️ CNN – Image Classification (CIFAR-10)

The CNN model successfully learned discriminative visual features across 10 classes.

Training accuracy increased steadily with data augmentation and batch normalization.

A small gap between training and validation accuracy indicates controlled overfitting.

Deeper architectures improved performance but increased training time.

📌 Observation:
CNNs are effective for spatial feature extraction, but require regularization to generalize well.

📝 RNN – Sentiment Analysis (IMDB)

The RNN model learned short-term dependencies in textual data.

Performance was stable for shorter sequences but degraded for longer contexts.

Validation accuracy plateaued early due to vanishing gradient limitations.

📌 Observation:
Simple RNNs are suitable for basic sequence tasks but struggle with long-term dependencies.

📈 LSTM – Time Series Prediction

The LSTM model successfully captured temporal patterns in sequential data.

Training loss decreased smoothly, indicating stable learning.

Multi-step predictions followed the overall trend of the sequence.

📌 Observation:
LSTMs handle long-term dependencies better than RNNs, making them suitable for time-series forecasting.

📊 Bidirectional LSTM – Stock Price Prediction (Apple)

The Bi-LSTM model captured both forward and backward temporal dependencies.

Predictions closely followed the general price trend, though short-term volatility was smoothed.

Proper sequence creation and scaling significantly improved stability.

📌 Observation:
Bidirectional LSTMs are effective for trend modeling but should not be used as financial advice.

🧠 Overall Insights

Model complexity improves representation power but increases computation cost.

Correct data preprocessing and sequence handling are more important than model depth.

Architectural choice should depend on the problem domain, not just accuracy.



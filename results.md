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


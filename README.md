<h1 align="center" style="color:#7B61FF;">
🧠 Deep Learning Models using Python
</h1>

<p align="center" style="font-size:18px;">
CNN • RNN • LSTM • Bidirectional LSTM
</p>

<p align="center" style="font-size:16px;">
A structured, runnable Deep Learning repository focused on
<strong>implementation, experimentation, and engineering discipline</strong>.
</p>

<hr>

<h2 align="center">📌 Overview</h2>

<p align="center" style="font-size:16px;">
This repository contains <strong>four core Deep Learning models</strong> implemented using
<strong>Python and TensorFlow</strong>, covering major problem domains:
</p>

<p align="center" style="font-size:16px;">
🖼️ Image Classification &nbsp; | &nbsp;
📝 Natural Language Processing &nbsp; | &nbsp;
📈 Time Series Forecasting &nbsp; | &nbsp;
📊 Financial Prediction
</p>

<p align="center" style="font-size:16px;">
Each model is designed to be:
<br><br>
✔ Runnable from a single entry point (<code>main.py</code>)<br>
✔ Modular and reusable<br>
✔ Backed by saved results (plots & models)<br>
✔ Easy for recruiters and learners to understand
</p>

<hr>

<h2 align="center">🗂️ Project Structure</h2>

<pre align="center">
Deep_Learning/
│
├── main.py                  # Entry point to run all models
├── requirements.txt         # Required dependencies
├── README.md                # Documentation
│
├── models/                  # Final Deep Learning models
│   ├── __init__.py
│   ├── cnn.py
│   ├── rnn.py
│   ├── lstm.py
│   └── bidirectional_lstm.py
│
├── notebooks/               # Jupyter notebooks (experiments)
│
├── data/                    # Datasets (e.g., Apple.csv)
│
└── outputs/                 # Saved models & result plots
</pre>

<hr>

<h2 align="center">🚀 Models Implemented</h2>

<br>

<h3 align="center">🖼️ 1. Convolutional Neural Network (CNN)</h3>

<p align="center" style="font-size:16px;">
<strong>Task:</strong> Image Classification (CIFAR-10)<br>
<strong>Concepts:</strong> Convolution, Pooling, Batch Normalization, Dropout
</p>

<p align="center" style="font-size:16px;">
✔ Data Augmentation<br>
✔ Regularization using Dropout<br>
✔ Training & Validation Accuracy/Loss<br>
✔ Saved trained model
</p>

<p align="center">
<img src="Outputs/cnn_training_curves.png" width="700">
</p>

<hr>

<h3 align="center">📝 2. Recurrent Neural Network (RNN)</h3>

<p align="center" style="font-size:16px;">
<strong>Task:</strong> IMDB Movie Review Sentiment Analysis<br>
<strong>Concepts:</strong> Embeddings, Sequential Text Modeling
</p>

<p align="center" style="font-size:16px;">
✔ Text preprocessing & padding<br>
✔ Binary classification (positive / negative)<br>
✔ Accuracy & Loss tracking<br>
✔ Saved trained model
</p>

<p align="center">
<img src="Outputs/rnn_training_curves.png" width="700">
</p>

<hr>

<h3 align="center">📈 3. Long Short-Term Memory (LSTM)</h3>

<p align="center" style="font-size:16px;">
<strong>Task:</strong> Multi-step Time Series Prediction<br>
<strong>Concepts:</strong> Sequence learning, Temporal dependencies
</p>

<p align="center" style="font-size:16px;">
✔ Synthetic time-series data<br>
✔ Multi-step forecasting<br>
✔ Training loss visualization<br>
✔ Prediction visualization
</p>

<p align="center">
<img src="Outputs/lstm_training_loss.png" width="700">
</p>

<hr>

<h3 align="center">📊 4. Bidirectional LSTM (Bi-LSTM)</h3>

<p align="center" style="font-size:16px;">
<strong>Task:</strong> Stock Price Prediction (NFLX / Apple)<br>
<strong>Concepts:</strong> Bidirectional sequence learning, real-world data handling
</p>

<p align="center" style="font-size:16px;">
✔ Real financial dataset<br>
✔ Proper scaling (no data leakage)<br>
✔ Early stopping<br>
✔ Prediction vs actual price visualization
</p>

<p align="center">
<img src="outputs/bilstm_nflx_prediction.png" width="700">
</p>

<hr>

<h2 align="center">▶️ How to Run</h2>

<p align="center" style="font-size:16px;">
<strong>1️⃣ Clone the repository</strong><br><br>
<code>git clone https://github.com/Adarshkumar61/Deep_Learning.git</code>
</p>

<p align="center" style="font-size:16px;">
<strong>2️⃣ Install dependencies</strong><br><br>
<code>pip install -r requirements.txt</code>
</p>

<p align="center" style="font-size:16px;">
<strong>3️⃣ Run the main program</strong><br><br>
<code>python main.py</code>
</p>

<p align="center" style="font-size:16px;">
Select a model from the menu and it will execute automatically.
</p>

<hr>

<h2 align="center">📊 Results & Observations</h2>

<p align="center" style="font-size:16px;">
✔ CNN performs well with data augmentation but can overfit without regularization<br><br>
✔ Simple RNN works for short sequences but struggles with long dependencies<br><br>
✔ LSTM improves stability in time-series prediction<br><br>
✔ Bidirectional LSTM captures richer temporal patterns but increases computation cost
</p>

<hr>

<h2 align="center">🧠 Key Learning Outcomes</h2>

<p align="center" style="font-size:16px;">
✔ Understanding Deep Learning across multiple domains<br><br>
✔ Writing modular, reusable ML code<br><br>
✔ Proper dataset handling & preprocessing<br><br>
✔ Avoiding common ML mistakes (data leakage, wrong splits)<br><br>
✔ Saving and analyzing training results
</p>

<hr>

<h2 align="center">🔮 Future Improvements</h2>

<p align="center" style="font-size:16px;">
🚀 Replace Simple RNN with GRU & Attention<br><br>
🧠 Implement Transformer-based models<br><br>
📊 Add more evaluation metrics (RMSE, Precision/Recall)<br><br>
☁️ Deploy trained models using APIs or dashboards<br><br>
🤖 Integrate Deep Learning models with Robotics & Vision systems
</p>

<hr>

<h2 align="center">👨‍💻 Author</h2>

<p align="center" style="font-size:16px;">
<strong>Adarsh Kumar</strong><br><br>
🎓 BCA Student | 🤖 Robotics • AI • Deep Learning Enthusiast<br><br>
🔗 <a href="https://github.com/Adarshkumar61">GitHub Profile</a>
</p>

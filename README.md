<h1 align="center">🧠 Deep Learning Models using Python</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange">
  <img src="https://img.shields.io/badge/License-MIT-green">
  <img src="https://img.shields.io/badge/Status-Active-success">
</p>

<p align="center" style="font-size:17px;">
A <strong>structured, modular Deep Learning repository</strong> implementing core neural network
architectures using <strong>Python and TensorFlow</strong>.
</p>

<p align="center" style="font-size:16px;">
Focused on <strong>correct data handling, reproducibility, and engineering discipline</strong>
rather than black-box usage.
</p>

<hr>

<h2 align="center">📌 Table of Contents</h2>

<p align="center">
Overview • Project Structure • Models • Datasets • How to Run • Results • Learnings • Future Work
</p>

<hr>

<h2 align="center">📖 Overview</h2>

<p align="center" style="font-size:16px;">
This repository demonstrates <strong>four foundational Deep Learning models</strong>,
each solving a different real-world problem:
</p>

<p align="center" style="font-size:16px;">
🖼️ Computer Vision &nbsp; | &nbsp;
📝 Natural Language Processing &nbsp; | &nbsp;
📈 Time Series Forecasting &nbsp; | &nbsp;
📊 Financial Modeling
</p>

<p align="center" style="font-size:16px;">
Each model is implemented as a <strong>standalone module</strong>,
exposed via a <code>run()</code> function and executed through
a unified entry point (<code>main.py</code>).
</p>

<hr>

<h2 align="center">🗂 Project Structure</h2>

<pre align="center">
Deep_Learning/
│
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
├── CONTRIBUTING.md
│
├── models/
│   ├── cnn.py
│   ├── rnn.py
│   ├── lstm.py
│   └── bidirectional_lstm.py
│
├── data/
│   └── AAPL.csv
│
├── outputs/
│   ├── cnn_training_curves.png
│   ├── rnn_training_curves.png
│   ├── lstm_training_loss.png
│   └── bilstm_stock_prediction.png
│
└── notebooks/
</pre>

<hr>

<h2 align="center">🚀 Models Implemented</h2>

| Model | Task | Dataset | Core Concepts |
|------|------|---------|---------------|
| CNN | Image Classification | CIFAR-10 | Convolution, Pooling |
| RNN | Sentiment Analysis | IMDB | Sequence Modeling |
| LSTM | Time Series | Synthetic Data | Long-term Memory |
| Bi-LSTM | Stock Forecasting | Apple (AAPL) | Bidirectional Context |

<hr>

<h2 align="center">📊 Datasets</h2>

| Dataset | Source | Usage |
|-------|--------|------|
| CIFAR-10 | Keras | Image Classification |
| IMDB Reviews | Keras | NLP |
| Synthetic Series | Generated | Time Series |
| Apple Stock (AAPL) | Yahoo Finance | Financial Forecasting |

<p align="center" style="font-size:15px;">
📌 Place Apple dataset at <code>data/AAPL.csv</code><br>
Expected column: <strong>Close</strong>
</p>

<hr>

<h2 align="center">▶️ How to Run</h2>

<p align="center">
<strong>1️⃣ Clone the repository</strong><br>
<code>git clone https://github.com/Adarshkumar61/Deep_Learning.git</code>
</p>

<p align="center">
<strong>2️⃣ Install dependencies</strong><br>
<code>pip install -r requirements.txt</code>
</p>

<p align="center">
<strong>3️⃣ Run the project</strong><br>
<code>python main.py</code>
</p>

<p align="center">
Select a model from the menu to execute it.
</p>

<hr>

<h2 align="center">📈 Results & Visuals</h2>

<p align="center"><strong>🖼️ CNN – Accuracy & Loss</strong></p>
<p align="center"><img src="outputs/cnn_training_curves.png" width="700"></p>

<p align="center"><strong>📝 RNN – Sentiment Classification</strong></p>
<p align="center"><img src="outputs/rnn_training_curves.png" width="700"></p>

<p align="center"><strong>📈 LSTM – Time Series Loss</strong></p>
<p align="center"><img src="outputs/lstm_training_loss.png" width="700"></p>

<p align="center"><strong>📊 Bi-LSTM – Stock Prediction</strong></p>
<p align="center"><img src="outputs/bilstm_stock_prediction.png" width="700"></p>

<hr>

<h2 align="center">🧠 Key Learnings</h2>

<p align="center">
✔ Sequence creation before splitting<br>
✔ Avoiding data leakage<br>
✔ RNN vs LSTM vs Bi-LSTM differences<br>
✔ Modular ML system design<br>
✔ Reproducible pipelines
</p>

<hr>

<h2 align="center">🔮 Future Improvements</h2>

<p align="center">
🚀 GRU & Attention models<br>
🧠 Transformer architectures<br>
📊 Advanced evaluation metrics<br>
☁️ Deployment with APIs<br>
🤖 Robotics & Vision integration
</p>

<hr>

<h2 align="center">👨‍💻 Author</h2>

<p align="center" style="font-size:16px;">
<strong>Adarsh Kumar</strong><br>
🎓 BCA Student | 🤖 Robotics • AI • Deep Learning Enthusiast<br>
🔗 <a href="https://github.com/Adarshkumar61">GitHub Profile</a>
</p>

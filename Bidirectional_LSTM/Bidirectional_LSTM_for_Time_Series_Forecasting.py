import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv('Apple.csv')
df.isnull().sum()

price = df['close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range= (0,1))
scaled_data = scaler.fit_transform(price)


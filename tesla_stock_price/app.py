# app.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Tesla 5-Day Return Predictor", layout="wide")
st.title("Tesla 5-Day Return Predictor using LSTM")

# Step 0: Upload CSV
uploaded_file = st.file_uploader("Upload CSV with 'Close' column", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:", df.head())

    # Step 1: Feature engineering
    df['Return'] = df['Close'].pct_change().fillna(0)
    df['MA5'] = df['Close'].rolling(5).mean().bfill()
    df['MA10'] = df['Close'].rolling(10).mean().bfill()
    df['MA20'] = df['Close'].rolling(20).mean().bfill()
    df['Lag1'] = df['Return'].shift(1).fillna(0)
    df['Lag2'] = df['Return'].shift(2).fillna(0)
    df['Lag3'] = df['Return'].shift(3).fillna(0)

    HORIZON = 5
    df['Target'] = df['Close'].pct_change(periods=HORIZON).shift(-HORIZON)
    df = df.dropna()

    feature_cols = ['Close','Return','MA5','MA10','MA20','Lag1','Lag2','Lag3']
    features = df[feature_cols].values
    target = df['Target'].values

    # Step 2: Scaling
    scalers = {}
    scaled_features = np.zeros_like(features)
    for i, col in enumerate(feature_cols):
        scalers[col] = MinMaxScaler()
        scaled_features[:,i:i+1] = scalers[col].fit_transform(features[:,i:i+1])
    target_scaler = MinMaxScaler()
    scaled_target = target_scaler.fit_transform(target.reshape(-1,1))

    # Step 3: Sequence creation
    SEQ_LENGTH = 90
    X, y = [], []
    for i in range(len(scaled_features) - SEQ_LENGTH):
        X.append(scaled_features[i:i+SEQ_LENGTH])
        y.append(scaled_target[i+SEQ_LENGTH,0])
    X = torch.from_numpy(np.array(X)).float()
    y = np.array(y)

    # Step 4: Load trained LSTM
    class StockLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim=64):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
            self.fc = nn.Linear(hidden_dim,1)
        def forward(self,x):
            out,_ = self.lstm(x)
            out = out[:,-1,:]
            out = self.fc(out)
            return out

    input_dim = X.shape[2]
    model = StockLSTM(input_dim)
    model.load_state_dict(torch.load("models/lstm_5day_return.pth"))
    model.eval()

    # Step 5: Make predictions
    with torch.no_grad():
        y_pred = model(X).numpy()
    y_pred_rescaled = target_scaler.inverse_transform(y_pred.reshape(-1,1))[:,0]

    # Step 6: Baselines
    baseline_yesterday = df['Return'].values[SEQ_LENGTH:]
    baseline_ma = df['MA5'].pct_change().values[SEQ_LENGTH:]

    # Compute RMSE
    lstm_rmse = np.sqrt(np.mean((y_pred_rescaled - df['Target'].values[SEQ_LENGTH:])**2))
    baseline_yesterday_rmse = np.sqrt(np.mean((baseline_yesterday - df['Target'].values[SEQ_LENGTH:])**2))
    baseline_ma_rmse = np.sqrt(np.mean((baseline_ma - df['Target'].values[SEQ_LENGTH:])**2))

    st.subheader("RMSE Comparison")
    st.write(f"LSTM RMSE: {lstm_rmse:.6f}")
    st.write(f"Yesterday Baseline RMSE: {baseline_yesterday_rmse:.6f}")
    st.write(f"MA5 Baseline RMSE: {baseline_ma_rmse:.6f}")

    # Step 7: Plot predictions
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    plt.plot(df.index[SEQ_LENGTH:], df['Target'].values[SEQ_LENGTH:], label='Actual 5-Day Return')
    plt.plot(df.index[SEQ_LENGTH:], y_pred_rescaled, label='LSTM Prediction')
    plt.xlabel("Days")
    plt.ylabel("5-Day Return")
    plt.legend()
    st.pyplot(plt)

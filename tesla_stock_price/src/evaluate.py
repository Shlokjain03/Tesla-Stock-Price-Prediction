# evaluate.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Load data
data_path = "data/TSLA.csv"
df = pd.read_csv(data_path)

# Feature engineering
HORIZON = 5
df['Return'] = df['Close'].pct_change().fillna(0)
df['MA5'] = df['Close'].rolling(5).mean().bfill()
df['MA10'] = df['Close'].rolling(10).mean().bfill()
df['MA20'] = df['Close'].rolling(20).mean().bfill()
df['Lag1'] = df['Return'].shift(1).fillna(0)
df['Lag2'] = df['Return'].shift(2).fillna(0)
df['Lag3'] = df['Return'].shift(3).fillna(0)

df['Target'] = df['Close'].pct_change(periods=HORIZON).shift(-HORIZON)
df = df.dropna()

feature_cols = ['Close','Return','MA5','MA10','MA20','Lag1','Lag2','Lag3']
features = df[feature_cols].values
target = df['Target'].values

# Define model
class StockLSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim,hidden_dim,num_layers=1,batch_first=True)
        self.fc = nn.Linear(hidden_dim,1)
    def forward(self,x):
        out,_ = self.lstm(x)
        out = out[:,-1,:]
        out = self.fc(out)
        return out

input_dim = len(feature_cols)
model = StockLSTM(input_dim)
model.load_state_dict(torch.load("models/lstm_5day_return.pth"))
model.eval()

# Walk-forward validation
SEQ_LENGTH = 90
WINDOW_SIZE = 500
STEP = 1

target_scaler = MinMaxScaler()
target_scaler.fit(target.reshape(-1,1))

lstm_rmse_list = []
baseline_yesterday_rmse = []
baseline_ma_rmse = []

for start in range(0,len(target)-WINDOW_SIZE-1,STEP):
    train_end = start+WINDOW_SIZE
    if train_end < SEQ_LENGTH:
        continue
    
    # scale features in window
    X_window = features[start:train_end]
    scalers = {}
    scaled_X_window = np.zeros_like(X_window)
    for i,col in enumerate(feature_cols):
        scalers[col] = MinMaxScaler()
        scaled_X_window[:,i:i+1] = scalers[col].fit_transform(X_window[:,i:i+1])
    
    # take last SEQ_LENGTH points
    X_seq = scaled_X_window[-SEQ_LENGTH:].reshape(1,SEQ_LENGTH,len(feature_cols))
    X_seq = torch.from_numpy(X_seq).float()
    
    y_true = target[train_end]
    
    with torch.no_grad():
        y_pred = model(X_seq).numpy()[0,0]
    lstm_rmse_list.append((y_pred - y_true)**2)
    
    # Baseline A: yesterday's return
    baseline_yesterday_rmse.append((target[train_end-1]-y_true)**2)
    
    # Baseline B: MA5 return
    ma_pred = df['MA5'].values[train_end-1]/df['Close'].values[train_end-SEQ_LENGTH] - 1
    baseline_ma_rmse.append((ma_pred - y_true)**2)

# Compute average RMSE
print("Walk-forward validation results:")
print(f"LSTM RMSE         : {np.sqrt(np.mean(lstm_rmse_list)):.6f}")
print(f"Baseline A RMSE   : {np.sqrt(np.mean(baseline_yesterday_rmse)):.6f}")
print(f"Baseline B RMSE   : {np.sqrt(np.mean(baseline_ma_rmse)):.6f}")

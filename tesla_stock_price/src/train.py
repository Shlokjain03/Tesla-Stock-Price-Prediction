# train.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Step 0: Load data
data_path = "data/TSLA.csv"  
df = pd.read_csv(data_path)

# Step 1: Feature engineering
df['Return'] = df['Close'].pct_change().fillna(0)
df['MA5'] = df['Close'].rolling(5).mean().bfill()
df['MA10'] = df['Close'].rolling(10).mean().bfill()
df['MA20'] = df['Close'].rolling(20).mean().bfill()
df['Lag1'] = df['Return'].shift(1).fillna(0)
df['Lag2'] = df['Return'].shift(2).fillna(0)
df['Lag3'] = df['Return'].shift(3).fillna(0)

# Target: 5-day forward return
HORIZON = 5
df['Target'] = df['Close'].pct_change(periods=HORIZON).shift(-HORIZON)
df = df.dropna()  # remove rows without target

feature_cols = ['Close', 'Return', 'MA5', 'MA10', 'MA20', 'Lag1', 'Lag2', 'Lag3']
features = df[feature_cols].values
target = df['Target'].values.reshape(-1,1)

# Step 2: Scaling
scalers = {}
scaled_features = np.zeros_like(features)
for i, col in enumerate(feature_cols):
    scalers[col] = MinMaxScaler()
    scaled_features[:, i:i+1] = scalers[col].fit_transform(features[:, i:i+1])

target_scaler = MinMaxScaler()
scaled_target = target_scaler.fit_transform(target)

# Step 3: Sequence creation
SEQ_LENGTH = 90
def create_sequences(features, target, seq_length=SEQ_LENGTH):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length])
        y.append(target[i+seq_length, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_features, scaled_target)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float().unsqueeze(1)

# Step 4: Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 5: LSTM model
class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

input_dim = X_train.shape[2]
model = StockLSTM(input_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 6: Training loop
EPOCHS = 100
BATCH_SIZE = 32

for epoch in range(EPOCHS):
    model.train()
    permutation = torch.randperm(X_train.size()[0])
    
    epoch_loss = 0
    for i in range(0, X_train.size()[0], BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE]
        batch_X, batch_y = X_train[indices], y_train[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    epoch_loss /= (X_train.size()[0] / BATCH_SIZE)
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.6f}")

# Step 7: Evaluate on test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test).numpy()

y_test_rescaled = target_scaler.inverse_transform(y_test.numpy())
y_pred_rescaled = target_scaler.inverse_transform(y_pred)

mse = np.mean((y_test_rescaled - y_pred_rescaled)**2)
rmse = np.sqrt(mse)
print(f"Test MSE: {mse:.6f}")
print(f"Test RMSE: {rmse:.6f}")

# Step 8: Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/lstm_5day_return.pth")
print("Model saved to models/lstm_5day_return.pth")

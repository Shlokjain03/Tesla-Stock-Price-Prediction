import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.data_loader import load_stock_data
from src.config import DATA_PATH


def get_train_test_data(split_ratio=0.8):
    """
    Loads data, scales Adj Close prices, and splits into train/test.
    Returns:
        train_data (np.array)
        test_data (np.array)
        scaler (MinMaxScaler)
    """

    df = load_stock_data(DATA_PATH)

    prices = df["Adj Close"].values.reshape(-1, 1)

    split_index = int(len(prices) * split_ratio)
    train_data = prices[:split_index]
    test_data = prices[split_index:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    return train_data, test_data, scaler


if __name__ == "__main__":
    train_data, test_data, _ = get_train_test_data()
    print("Train shape:", train_data.shape)
    print("Test shape:", test_data.shape)

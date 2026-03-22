import os
import pandas as pd


REQUIRED_COLUMNS = ["Date",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Adj Close",
                    "Volume"
                    ]


def load_stock_data(csv_path: str) -> pd.DataFrame:
    
#Load and validate Tesla stock price data for time-series analysis.

# 1. Check file existence
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")

    # 2. Load CSV
    df = pd.read_csv(csv_path)

    # 3. Validate required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # 4. Convert Date to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop rows where Date could not be parsed
    df = df.dropna(subset=["Date"])

    # 5. Sort by Date (CRITICAL)
    df = df.sort_values("Date")

    # 6. Set Date as index
    df.set_index("Date", inplace=True)

    # 7. Basic sanity checks

    # Ensure index is strictly increasing
    if not df.index.is_monotonic_increasing:
        raise ValueError("Date index is not sorted correctly.")

    # Check for negative prices (invalid in stock data)
    price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
    if (df[price_cols] < 0).any().any():
        raise ValueError("Negative values found in price columns.")

    # Volume should not be negative
    if (df["Volume"] < 0).any():
        raise ValueError("Negative values found in Volume column.")

    return df


# Local test (run once, then ignore)
if __name__ == "__main__":
    data_path = "data/TSLA.csv"
    df = load_stock_data(data_path)

    print("Data loaded successfully.")
    print("Shape:", df.shape)
    print("\nHead:")
    print(df.head())
    print("\nTail:")
    print(df.tail())
    print("\nMissing values per column:")
    print(df.isna().sum())

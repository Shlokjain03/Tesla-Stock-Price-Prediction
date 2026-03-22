import numpy as np
from typing import Tuple


def create_sequences(
    data: np.ndarray,
    lookback: int,
    horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a univariate time-series into supervised learning sequences.

    Parameters
    ----------
    data : np.ndarray
        Scaled time-series data of shape (num_samples, 1)
    lookback : int
        Number of past time steps used as input sequence
    horizon : int
        Number of days ahead to predict
    """

    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")

    if data.ndim != 2 or data.shape[1] != 1:
        raise ValueError("Input data must have shape (num_samples, 1).")

    if lookback <= 0:
        raise ValueError("lookback must be a positive integer.")

    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")

    X, y = [], []

    max_index = len(data) - lookback - horizon + 1
    if max_index <= 0:
        raise ValueError(
            "Not enough data points to create sequences with the given "
            "lookback and horizon."
        )

    for i in range(max_index):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback + horizon - 1])

    X = np.array(X)
    y = np.array(y)

    return X, y



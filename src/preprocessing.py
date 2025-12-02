# src/preprocessing.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

def prepare_dataset(df, features, target, window_size, horizon, test_size, random_seed):
    """
    Creates supervised windows:
    For each index i: X = rows [i : i+window_size-1], y = target at i+window_size+horizon-1
    Returns: X, y (numpy arrays), scalers (x_scaler, y_scaler)
    """
    # encode Sample if present (AIML/Biotech)
    if 'Sample' in df.columns:
        df = df.copy()
        df['Sample'] = df['Sample'].map({'AIML':0, 'Biotech':1})

    data_x = df[features].values.astype(float)
    data_y = df[[target]].values.astype(float)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    data_x_scaled = x_scaler.fit_transform(data_x)
    data_y_scaled = y_scaler.fit_transform(data_y)

    X_windows = []
    y_windows = []

    total_len = len(df)
    max_start = total_len - window_size - (horizon - 1)

    for start in range(0, max_start):
        end = start + window_size
        target_idx = end + (horizon - 1)
        X_seq = data_x_scaled[start:end]
        y_val = data_y_scaled[target_idx]
        X_windows.append(X_seq)
        y_windows.append(y_val)

    X = np.array(X_windows)  # shape: (samples, window_size, features)
    y = np.array(y_windows)  # shape: (samples, 1)

    # split (time-agnostic shuffle split is acceptable here since windows were constructed; for stricter time-split change this)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, shuffle=True
    )

    return X_train, X_test, y_train, y_test, x_scaler, y_scaler

# src/utils_io.py
import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def save_scaler(scaler, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)

def save_model_keras(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)

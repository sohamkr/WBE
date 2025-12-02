# src/evaluate.py
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils_io import load_scaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.config import MODELS_DIR, DATA_PATH
from src.preprocessing import prepare_dataset
from src.utils_io import load_data

def load_test_data(window=7, horizon=1):
    df = load_data(DATA_PATH)
    features = ["Day", "Sample", "Colonies", "Temperature", "pH", "Turbidity"]
    target = "CFU_g"
    X_train, X_test, y_train, y_test, x_scaler, y_scaler = prepare_dataset(
        df, features, target, window, horizon, test_size=0.2, random_seed=42
    )
    return X_test, y_test, y_scaler

def eval_model(model_path, scaler_y, X_test, y_test):
    model = load_model(model_path)
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}, y_true, y_pred

if __name__ == "__main__":
    X_test, y_test, y_scaler = None, None, None
    X_test, y_test, y_scaler = load_test_data(window=7, horizon=1)
    models = {
        "lstm": os.path.join(MODELS_DIR, "lstm_best.h5"),
        "gru": os.path.join(MODELS_DIR, "gru_best.h5")
    }

    summary = {}
    for name, path in models.items():
        if not os.path.exists(path):
            print(f"Model {path} not found, skip.")
            continue
        metrics, y_true, y_pred = eval_model(path, y_scaler, X_test, y_test)
        summary[name] = metrics
        # plot
        plt.figure(figsize=(8,4))
        plt.plot(y_true.flatten(), label="Actual")
        plt.plot(y_pred.flatten(), label=f"{name.upper()} Pred")
        plt.legend()
        plt.title(f"{name.upper()} vs Actual")
        plt.savefig(os.path.join(MODELS_DIR, f"eval_{name}.png"))
        plt.close()

    print(json.dumps(summary, indent=2))

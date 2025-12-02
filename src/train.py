# src/train.py
import argparse
import os
import numpy as np
import json

from src.config import DATA_PATH, MODELS_DIR, WINDOW_SIZE, HORIZON, TEST_SIZE, RANDOM_SEED, EPOCHS, BATCH_SIZE, LEARNING_RATE
from src.utils_io import load_data, save_scaler, save_model_keras
from src.preprocessing import prepare_dataset
from src.lstm_model import build_lstm
from src.gru_model import build_gru

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def evaluate_and_save(true, pred, out_prefix):
    mae = mean_absolute_error(true, pred)
    rmse = mean_squared_error(true, pred, squared=False)
    r2 = r2_score(true, pred)
    metrics = {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}
    with open(out_prefix + "_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics

def main(args):
    np.random.seed(42)
    df = load_data(DATA_PATH)

    features = ["Day", "Sample", "Colonies", "Temperature", "pH", "Turbidity"]
    target = "CFU_g"

    X_train, X_test, y_train, y_test, x_scaler, y_scaler = prepare_dataset(
        df, features, target, args.window, args.horizon, args.test_size, args.seed
    )

    # Save scalers
    save_scaler(x_scaler, os.path.join(MODELS_DIR, f"scaler_x.joblib"))
    save_scaler(y_scaler, os.path.join(MODELS_DIR, f"scaler_y.joblib"))

    input_shape = (X_train.shape[1], X_train.shape[2])

    models_to_run = []
    if args.model in ("lstm", "both"):
        models_to_run.append("lstm")
    if args.model in ("gru", "both"):
        models_to_run.append("gru")

    results = {}
    for mname in models_to_run:
        print(f"\nTraining {mname.upper()} ...")
        if mname == "lstm":
            model = build_lstm(input_shape, lr=args.lr)
        else:
            model = build_gru(input_shape, lr=args.lr)

        checkpoint_path = os.path.join(MODELS_DIR, f"{mname}_best.h5")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
            ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
        ]

        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=args.epochs,
            batch_size=args.batch,
            callbacks=callbacks,
            verbose=1
        )

        # Predict and inverse transform
        y_pred_scaled = model.predict(X_test)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_true = y_scaler.inverse_transform(y_test)

        # Save model and metrics
        save_model_keras(model, checkpoint_path)
        metrics = evaluate_and_save(y_true, y_pred, os.path.join(MODELS_DIR, mname))
        results[mname] = metrics

        # Simple plot
        plt.figure(figsize=(8,4))
        plt.plot(y_true.flatten(), label="Actual")
        plt.plot(y_pred.flatten(), label=f"{mname.upper()} Pred")
        plt.title(f"{mname.upper()} Actual vs Pred")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, f"{mname}_prediction_plot.png"))
        plt.close()

    # Save comparison
    with open(os.path.join(MODELS_DIR, "comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nTraining complete. Results saved to models/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm","gru","both"], default="both")
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)

# src/config.py
import os

# Paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "WBE_large_dataset.csv")
MODELS_DIR = os.path.join(ROOT, "..", "models") if "src" in ROOT else os.path.join(ROOT, "models")
# Make models dir relative to project root
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

os.makedirs(MODELS_DIR, exist_ok=True)

# Preprocessing / windowing params
WINDOW_SIZE = 7        # input days
HORIZON = 1            # predict days ahead (1 => next-day)
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Training hyperparams (defaults; can be overridden by train.py CLI)
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

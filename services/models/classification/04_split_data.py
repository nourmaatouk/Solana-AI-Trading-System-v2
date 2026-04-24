from pathlib import Path
import numpy as np


# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

X_PATH = DATA_DIR / "X.npy"
Y_PATH = DATA_DIR / "y.npy"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Safety check
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, "Ratios must sum to 1."


# =========================================================
# LOAD
# =========================================================
X = np.load(X_PATH)
y = np.load(Y_PATH)

print("Loaded:")
print("X shape:", X.shape)
print("y shape:", y.shape)


# =========================================================
# SPLIT INDICES
# =========================================================
n = len(X)

train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

# Chronological split
X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X[val_end:]
y_test = y[val_end:]


# =========================================================
# SAVE
# =========================================================
np.save(DATA_DIR / "X_train.npy", X_train)
np.save(DATA_DIR / "y_train.npy", y_train)

np.save(DATA_DIR / "X_val.npy", X_val)
np.save(DATA_DIR / "y_val.npy", y_val)

np.save(DATA_DIR / "X_test.npy", X_test)
np.save(DATA_DIR / "y_test.npy", y_test)


# =========================================================
# REPORT
# =========================================================
print("\nSplit done.")
print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Val  : X={X_val.shape}, y={y_val.shape}")
print(f"Test : X={X_test.shape}, y={y_test.shape}")


def show_distribution(name: str, arr: np.ndarray):
    unique, counts = np.unique(arr, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(unique, counts)}
    print(f"{name} distribution: {dist}")


print()
show_distribution("Train", y_train)
show_distribution("Val", y_val)
show_distribution("Test", y_test)
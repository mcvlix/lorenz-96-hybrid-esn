# compare_models.py
# Evaluates RMSE, NRMSE, and NSE for multiple models

import numpy as np
from metrics import rmse, nrmse, nse

# Replace these with real files when ready
try:
    y_true = np.load("X_true.npy")
    y_esn = np.load("X_esn.npy")
    y_imperfect = np.load("X_imperfect.npy")
    y_hybrid = np.load("X_hybrid.npy")
except:
    # Dummy test data
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_esn = np.array([1.1, 2.1, 2.9, 4.2, 5.2])
    y_imperfect = np.array([1.5, 2.0, 3.0, 4.5, 5.5])
    y_hybrid = np.array([1.0, 2.0, 3.1, 4.1, 5.0])

models = {
    "ESN": y_esn,
    "Imperfect": y_imperfect,
    "Hybrid": y_hybrid
}

for name, pred in models.items():
    print(f"\n{name} Model:")
    print("RMSE:", rmse(pred, y_true))
    print("NRMSE:", nrmse(pred, y_true))
    print("NSE:", nse(pred, y_true))

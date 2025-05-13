# metrics.py
# Implements RMSE, NRMSE, and NSE

import numpy as np

def rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true)**2))

def nrmse(y_pred, y_true):
    return rmse(y_pred, y_true) / np.std(y_true)

def nse(y_pred, y_true):
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

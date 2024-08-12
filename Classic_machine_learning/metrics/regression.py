import numpy as np


def mean_absolute_error(y_true: np.array, y_pred: np.array) -> float:
    """Mean absolute error (MAE) regression loss."""
    return np.abs(y_true - y_pred).mean()


def mean_squared_error(y_true: np.array, y_pred: np.array) -> float:
    """Mean squared error (MSE) regression loss."""
    return np.square(y_true - y_pred).mean()


def mean_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
    """Mean absolute percentage error (MAPE) regression loss"""
    return 100 * np.abs((y_true - y_pred) / y_true).mean()


def root_mean_squared_error(y_true: np.array, y_pred: np.array) -> float:
    """Root mean squared error (RMSE) regression loss"""
    return np.sqrt(np.square(y_true - y_pred).mean())


def r2_score(y_true: np.array, y_pred: np.array) -> float:
    """:math:`R^2` (coefficient of determination) regression score function"""
    return 1 - np.square(y_true - y_pred).sum() / np.square(y_true - y_pred.mean()).sum()

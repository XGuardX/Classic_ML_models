import numpy as np


def roc_auc_score(y: np.array, y_prob: np.array) -> float:
    """Compute Area Under the Curve (AUC)"""
    positives = np.sum(y == 1)
    negatives = np.sum(y == 0)

    sorted_idx = np.argsort(-np.array(np.round(y_prob, 10)))
    y_sorted = np.array(y)[sorted_idx]
    y_prob_sorted = y_prob[sorted_idx]

    roc_auc_score = 0

    for prob, pred in zip(y_prob_sorted, y_sorted):
        if pred == 0:
            roc_auc_score += (
                    np.sum(y_sorted[y_prob_sorted > prob])
                    + np.sum(y_sorted[y_prob_sorted == prob]) / 2
            )

    roc_auc_score /= positives * negatives

    return roc_auc_score


def accuracy_score(y_true: np.array, y_pred: np.array) -> float:
    """Compute accuracy from prediction scores"""
    return np.mean(y_true == y_pred)


def precision_score(y_true: np.array, y_pred: np.array) -> float:
    """Compute precision from prediction scores"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp) if tp + fp > 0 else 0


def recall_score(y_true: np.array, y_pred: np.array) -> float:
    """Compute recall from prediction scores"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn) if tp + fn > 0 else 0


def f1_score(y_true: np.array, y_pred: np.array) -> float:
    """Compute f1 from prediction scores"""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

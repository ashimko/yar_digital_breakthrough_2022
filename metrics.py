import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, recall_score
from typing import Union


def compute_single_col_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        max_metric = float('-inf')
        for tresh in np.round(np.unique(y_pred), 4):
            curr_metric = recall_score(y_true, np.where(y_pred >= tresh, 1, 0), average='macro', zero_division=0)
            if curr_metric > max_metric:
                max_metric = curr_metric
        return max_metric
    except:
        return 0
    

def compute_weird_pred_proba_score(y_true: np.ndarray, y_pred: Union[list, np.ndarray], sub_std: bool=False) -> float:
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.values

    if isinstance(y_pred, list):
        y_pred = np.hstack([pred[:, 1].reshape(-1, 1) for pred in y_pred])

    metrics = []
    for col in range(y_true.shape[1]):
        max_metric = compute_single_col_score(y_true[:, col], y_pred[:, col])
        metrics.append(max_metric)
    avg, std = np.mean(metrics), np.std(metrics)
    print(metrics)
    print(avg, std)
    if sub_std:
        return avg - std
    return avg


def get_tresholds(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict:
    if isinstance(y_pred, list):
        y_pred = np.hstack([pred[:, 1].reshape(-1, 1) for pred in y_pred])
        y_pred = pd.DataFrame(data=y_pred, index=y_true.index, columns=y_true.columns)
        
    metrics = []
    thresholds = {}
    for col in y_true.columns:
        max_metric = float('-inf')
        for thresh in y_pred[col].unique():
            curr_metric = recall_score(y_true[col].values, np.where(y_pred[col].values >= thresh, 1, 0), average='macro', zero_division=0)
            if curr_metric > max_metric:
                max_metric = curr_metric
                thresholds[col] = thresh
        metrics.append(max_metric)
    
    print(metrics)
    print(np.mean(metrics), np.std(metrics))
    return thresholds


def compute_weird_pred_score(y_true: np.ndarray, y_pred:np.ndarray) -> float:
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.values

    for col in range(y_true.shape[1]):
        curr_metric = recall_score(y_true[col], y_pred, average='macro', zero_division=0)
        avg += curr_metric
    avg /= y_true.shape[1]
    return avg


def get_weird_pred_proba_score():
    return make_scorer(score_func=compute_weird_pred_proba_score, greater_is_better=True, needs_proba=True)


def get_weird_pred_score():
    return make_scorer(score_func=compute_weird_pred_score, greater_is_better=True, needs_proba=False)


def get_weird_single_col_pred_proba_score():
    return make_scorer(score_func=compute_single_col_score, greater_is_better=True, needs_proba=True)

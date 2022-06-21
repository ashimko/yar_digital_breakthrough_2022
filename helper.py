from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
import os
from config import TARGETS, RENAME_MAP


def make_prediction(pred_proba: pd.DataFrame, tresholds: dict, sample_submission: pd.DataFrame) -> pd.DataFrame:
    assert all(pred_proba.index == sample_submission.index)
    
    for col in sample_submission.columns:
        trans_col = RENAME_MAP[col]
        sample_submission[col] = np.where(pred_proba[trans_col].values >= tresholds[trans_col], 1, 0)
        
    return sample_submission


def check_path(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

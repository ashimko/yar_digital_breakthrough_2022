from collections import defaultdict
from typing import Callable
from metrics import compute_weird_pred_proba_score, get_tresholds
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
import os
import pickle
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
import config as cfg
from tqdm.notebook import tqdm
import tensorflow as tf
import random


def make_prediction(pred_proba: pd.DataFrame, thresholds: dict, sample_submission: pd.DataFrame, rename_cols:bool=True) -> pd.DataFrame:
    assert all(pred_proba.index == sample_submission.index)
    
    for col in sample_submission.columns:
        trans_col = cfg.RENAME_MAP[col] if rename_cols else col
        sample_submission[col] = np.where(pred_proba[trans_col].values >= thresholds[trans_col], 1, 0)
        
    return sample_submission


def check_path(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def save_model(model: BaseEstimator, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def squeeze_pred_proba(pred_proba: list) -> np.ndarray:
    return np.hstack([x[:, 1].reshape(-1, 1)  for x in pred_proba])


def asemble_data(base_path: str, model_names) -> pd.DataFrame:
    df = []
    for model_name in model_names:
        for experiment in model_names[model_name]:
            path = os.path.join(base_path, model_name, f'{experiment}.pkl')
            tdf = pd.read_pickle(path)
            tdf.columns = [f'{model_name}_{experiment}_{col}' for col in tdf.columns]
            df.append(tdf)
    return pd.concat(df, axis=1)


def save_submission(submission: pd.DataFrame, experiment_family_name: str, experiment_name: str, suffix: str=''):
    submission_path = os.path.join(cfg.SUBMISSION_PATH, experiment_family_name)
    check_path(submission_path)
    if suffix:
        suffix = '_' + suffix
    submission.to_csv(os.path.join(submission_path, f'{experiment_name}{suffix}.csv'))


def save_pred_proba_oof(pred_proba_oof: pd.DataFrame, experiment_family_name: str, experiment_name: str, suffix: str=''):
    pred_proba_oof_path = os.path.join(cfg.OOF_PRED_PATH, experiment_family_name)
    check_path(pred_proba_oof_path)
    pred_proba_oof.to_pickle(os.path.join(pred_proba_oof_path, f'{experiment_name}{suffix}.pkl'))


def save_pred_proba_test(pred_proba_test: pd.DataFrame, experiment_family_name: str, experiment_name: str, suffix: str=''):
    pred_proba_test_path = os.path.join(cfg.TEST_PRED_PATH, experiment_family_name)
    check_path(pred_proba_test_path)
    pred_proba_test.to_pickle(os.path.join(pred_proba_test_path, f'{experiment_name}{suffix}.pkl'))


def get_prediction(
    train_data: pd.DataFrame,
    train_labels: pd.DataFrame,
    test_data: pd.DataFrame,
    pred_template: pd.DataFrame,
    process_input: Callable,
    save_model: Callable,
    fit_model: Callable,
    predict: Callable,
    experiment_name: str,
    experiment_family_name: str,
    n_splits: int = 5,
    random_state: int = 0,
    rename_cols=True,
    suffix=''
):
    cv = MultilabelStratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    pred_proba_oof = pd.DataFrame(data=np.zeros(shape=(len(train_data), len(cfg.TARGETS))), index=train_data.index, columns=cfg.TARGETS)
    pred_proba_test = pd.DataFrame(data=np.zeros(shape=(len(test_data), len(cfg.TARGETS))), index=test_data.index, columns=cfg.TARGETS)

    test_pool = process_input(test_data, shuffle=False)
    best_iters = []
    fold = 0
    for train_idx, val_idx in tqdm(cv.split(train_data, train_labels), total=n_splits): 
        train_pool = process_input(train_data.iloc[train_idx], train_labels.iloc[train_idx], shuffle=True)
        val_pool = process_input(train_data.iloc[val_idx], train_labels.iloc[val_idx], shuffle=False)

        model, best_iter = fit_model(train_pool, val_pool)
        best_iters.append(best_iter)
        save_model(model, experiment_name, experiment_family_name, fold, suffix)
        
        pred_proba_oof.iloc[val_idx, :] += predict(model, val_pool)
        pred_proba_test.iloc[:, :] += predict(model, test_pool)
        fold += 1
    pred_proba_test /= n_splits

    tresholds = get_tresholds(train_labels, pred_proba_oof)
    prediction = make_prediction(pred_proba_test, tresholds, pred_template, rename_cols=rename_cols)
        
    return prediction, pred_proba_oof, pred_proba_test, best_iters


def evaluate(
    test_labels: pd.DataFrame,
    prediction: pd.DataFrame, 
    pred_proba_test: pd.DataFrame
    ) -> dict:
    
    metrics = defaultdict(list)
    metrics['weird_score'] = compute_weird_pred_proba_score(test_labels, prediction)
    metrics['oof_auc'] = roc_auc_score(test_labels, pred_proba_test)
    metrics['oof_logloss'] = log_loss(test_labels, pred_proba_test)

    print('TEST METRICS')
    print(metrics)
    return metrics
    

def seed_everything(seed=0):
    def seed_basic(seed=seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        
    def seed_tf(seed=seed):
        tf.random.set_seed(seed)
    seed_basic(seed)
    seed_tf(seed)
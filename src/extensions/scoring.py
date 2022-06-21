import logging
import os
from typing import Dict

import numpy as np
import pandas as pd
import wandb
from sklearn.preprocessing import MinMaxScaler


def scale_predictions(preds: np.ndarray) -> np.ndarray:
    """
    Scale predictions to a scale of 0 to 1

    Args:
        preds (np.ndarray): raw unscaled predictions

    Returns:
        np.ndarray: scaled predictions
    """

    scaler = MinMaxScaler()
    scaled_preds = scaler.fit_transform(preds.reshape(-1, 1)).flatten()

    return scaled_preds


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def pearsonr(labels: np.ndarray, preds: np.ndarray):
    return np.corrcoef(labels, preds)[0][1]


def post_process_predictions(logits: np.ndarray, loss_type: str = "mse") -> np.ndarray:
    """
    Post-process predictions based on loss function

    Args:
        logits (np.ndarray): raw predictions
        loss_type (str): loss type

    Returns:
        np.ndarray: post-processed predictions
    """
    if loss_type == "pearson":
        print(f"Loss func: {loss_type}, post_process: scale_predictions")
        preds = scale_predictions(logits)
    elif loss_type == "bce":
        print(f"Loss func: {loss_type}, post_process: sigmoid")
        preds = sigmoid(logits)
    elif loss_type == "mse":
        print(f"Loss func: {loss_type}, post_process: none")
        preds = logits

    return preds


class Pearsonr:
    """
    Computes pearsonr and can be directly passed to HuggingFace Trainer's `compute_metrics` argument
    to monitor evaluation metrics while training.
    """

    def __init__(self, loss_type: str = "mse"):
        self.loss_type = loss_type

    def __call__(self, eval_outputs: np.ndarray):
        logits, labels = eval_outputs.predictions, eval_outputs.label_ids
        try:
            logits = logits.reshape(-1)
        except:
            logits = logits[0].reshape(-1)

        preds = post_process_predictions(logits, self.loss_type)

        # log logits & prediction distribution after post-process
        if wandb.run is not None:
            wandb.log({"val_logits": wandb.Histogram(logits)})
            wandb.log({"val_preds": wandb.Histogram(preds)})

        score = pearsonr(labels, preds)

        return {"pearsonr": score}


def calculate_cv_score(
    checkpoint_root_dir: str, oof_file_out_path: str, n_folds: int = 5
) -> float:
    """
    Calculates CV score for all folds combined (`n_folds`) from individually saved `oof.csv` file
    for a particular experiment's each fold run

    Args:
        checkpoint_root_dir (str): checkpoint directory where folders are `fold_0`, `fold_1`, etc. and each folder has `oof.csv` file
        oof_file_out_path (str): path to save final oof predictions file (has a prediction for each training sample)
        n_folds (int, optional): total folds. Defaults to 5.

    Returns:
        float: CV score
    """
    oof_df = pd.DataFrame()
    for fold in range(n_folds):
        oof_path = os.path.join(checkpoint_root_dir, f"fold_{fold}", "oof.csv")
        _oof = pd.read_csv(oof_path)

        oof_df = pd.concat([oof_df, _oof])

    oof_df.to_csv(oof_file_out_path, index=False)

    labels = oof_df["score"].to_numpy()
    preds = oof_df["preds"].to_numpy()

    cv_score = np.round(pearsonr(labels, preds), 5)

    logging.info(f"\n-> Final CV score: {cv_score}")

    return cv_score

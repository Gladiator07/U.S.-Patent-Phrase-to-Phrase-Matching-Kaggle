import logging

import numpy as np
from transformers import TrainerCallback


class SaveBestModelCallback(TrainerCallback):
    """
    Saves best model according to the competition metric and
    also logs the scores after each evaluation routine to
    custom log file created by `extensions.logger.setup_logger`
    """

    def __init__(self):
        self.bestScore = 0

    def on_train_begin(self, args, state, control, **kwargs):
        assert (
            args.evaluation_strategy != "no"
        ), "SaveBestModelCallback requires IntervalStrategy of steps or epoch"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_value = metrics.get("eval_pearsonr")
        if metric_value > self.bestScore:
            logging.info(
                f"** Pearsonr score improved from {np.round(self.bestScore, 4)} to {np.round(metric_value, 4)} **"
            )
            self.bestScore = metric_value
            control.should_save = True
        else:
            logging.info(
                f"Pearsonr score {np.round(metric_value, 4)} (Prev. Best {np.round(self.bestScore, 4)}) "
            )

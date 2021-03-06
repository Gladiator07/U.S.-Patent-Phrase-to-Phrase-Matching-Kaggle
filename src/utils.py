import copy
import glob
import logging
import os
import pickle
import random
import shutil
from typing import Any, Dict

import numpy as np
import slack
import torch
from omegaconf import OmegaConf


class SlackNotifier:
    """
    Notifies about important events to a dedicated slack channel
    """

    def __init__(self, slack_token: str, channel: str):
        self.slack_token = slack_token
        self.channel = channel
        self.client = slack.WebClient(token=self.slack_token)

    def notify(self, message: str):
        self.client.chat_postMessage(channel=self.channel, text=message)


def process_hydra_config(cfg: OmegaConf) -> OmegaConf:
    """
    Only keep relevant part of hydra config excluding `paths` field
    Args:
        cfg (OmegaConf): cfg
    """
    tmp_cfg = copy.deepcopy(cfg)
    tmp_cfg_dict = OmegaConf.to_container(tmp_cfg, resolve=True)
    del tmp_cfg_dict["paths"]
    tmp_cfg = OmegaConf.create(tmp_cfg_dict)
    return tmp_cfg


def save_pickle(file_path: str, object: Dict[Any, Any]):
    """
    Saves a object in pickle file at given path
    Args:
        file_path (str): path to save file
        object (Dict[Any, Any]): object to save
    """

    fp = open(file_path, "wb")
    pickle.dump(object, fp)


def load_pickle(file_path: str) -> Dict[Any, Any]:
    """
    Loads a pickle file from given path
    Args:
        file_path (str): path to load file from
    Returns:
        Dict[Any, Any]: loaded pickle file
    """

    fp = open(file_path, "rb")
    loaded_file = pickle.load(fp)
    return loaded_file


def delete_checkpoints(run_dir: str):
    """
    Deletes unecessary checkpoints generated by HuggingFace `Trainer`
    Args:
        run_dir (str): output directory where artifacts are saved
    """

    for file in glob.glob(f"{run_dir}/checkpoint-*"):
        shutil.rmtree(file, ignore_errors=True)


def seed_everything(seed=42):
    """
    Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.
    Makes sure to get reproducible results.
    Args:
        seed (int, optional): seed value. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.info(f"-> Global seed set to {seed}")


def asHours(seconds: float) -> str:
    """
    Returns seconds to human-readable formatted string
    Args:
        seconds (float): total seconds
    Returns:
        str: total seconds converted to human-readable formatted string
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:.0f}h:{m:.0f}m:{s:.0f}s"

import re
from pprint import pprint
from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from transformers import PreTrainedTokenizer

from data.cpc_texts import get_cpc_texts


def tokenize_func(
    row: pd.Series,
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, Any]:
    """
    General tokenize function for HuggingFace `datasets` (both for train and inference mode)
    """
    encoded = tokenizer(
        row["input_text"],
        add_special_tokens=True,
        return_token_type_ids=True,
    )
    return encoded


def tokenize_func_cocolm(
    row: pd.Series, tokenizer: PreTrainedTokenizer
) -> Dict[str, Any]:
    anchor = tokenizer.encode(row["anchor"], add_special_tokens=False)
    target = tokenizer.encode(row["target"], add_special_tokens=False)
    context_text = tokenizer.encode(row["context_text"], add_special_tokens=False)

    token_id = (
        [tokenizer.cls_token_id]
        + context_text
        + [tokenizer.sep_token_id]
        + anchor
        + [tokenizer.sep_token_id]
        + target
        + [tokenizer.sep_token_id]
    )

    attention_mask = (
        [1]
        + [1] * len(context_text)
        + [1]
        + [1] * len(anchor)
        + [1]
        + [1] * len(target)
        + [1]
    )
    token_type_ids = [0] * len(attention_mask)

    encoded = {
        "input_ids": token_id,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

    return encoded


def tokenize_func_transformer_head(
    row: pd.Series, tokenizer: PreTrainedTokenizer, max_length: int = 133
) -> Dict[str, Any]:
    """
    Tokenize function which also pads to `max_length` (both for train and inference mode)
    Required for transformer_head as dynamic padding doesn't work with it.
    """
    encoded = tokenizer(
        row["input_text"],
        add_special_tokens=True,
        return_token_type_ids=True,
        max_length=max_length,
        padding="max_length",
    )

    return encoded


def prepare_data(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    cpc_scheme_xml_dir: str,
    cpc_title_list_dir: str,
    use_custom_seperator: bool = False,
) -> pd.DataFrame:
    """
    Prepares data (applies both for `train` and `inference` mode)

    Args:
        train_df (pd.DataFrame): train dataframe
        cpc_scheme_xml_dir (str): directory where cpc_scheme_xml files are saved
        cpc_title_list_dir (str): directory where cpc_title_list files are saved

    Returns:
        pd.DataFrame: prepared `train_df`
    """

    if use_custom_seperator:
        sep = " [s] "
    else:
        sep = f" {tokenizer.sep_token} "

    cpc_texts = get_cpc_texts(
        cpc_scheme_xml_dir=cpc_scheme_xml_dir, cpc_title_list_dir=cpc_title_list_dir
    )
    df["context_text"] = df["context"].map(cpc_texts)

    df["cleaned_context_text"] = df["context_text"].map(
        lambda x: re.sub("[^A-Za-z0-9]+", " ", x)
    )

    # prepare input text
    df["input_text"] = df["context_text"] + sep + df["anchor"] + sep + df["target"]

    # adding section as special tokens
    df["section"] = df["context"].str[0]
    df["sectok"] = "[" + df.section + "]"
    return df


def create_folds(
    train_df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    print_summary: bool = False,
):
    """
    Split data grouped by anchor and stratified by score (`StratifiedGroupKFold`)

    Args:
        train_df (pd.DataFrame): train dataframe
        n_folds (int, optional): number of splits. Defaults to 5.
        seed (int, optional): random state. Defaults to 42.

    Returns:
        pd.DataFrame: dataframe with `fold` as column for splits
    """

    # CV SPLIT
    #######################################
    # grouped by anchor + stratified by score
    #######################################

    train_df["score_bin"] = pd.cut(train_df["score"], bins=5, labels=False)
    train_df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = sgkf.split(
        X=train_df,
        y=train_df["score_bin"].to_numpy(),
        groups=train_df["anchor"].to_numpy(),
    )
    for fold, (trn_idx, val_idx) in enumerate(folds):
        train_df.loc[val_idx, "fold"] = fold
    train_df["fold"] = train_df["fold"].astype(int)

    # #######################################
    if print_summary:
        print("\nSamples per fold:")
        print(train_df["fold"].value_counts())

        print("\n Mean score per fold:")
        scores = [
            train_df[train_df["fold"] == f]["score"].mean() for f in range(n_folds)
        ]
        pprint(scores)

        print("\n Score distribution per fold:")

        [
            print(train_df[train_df["fold"] == f]["score"].value_counts())
            for f in range(n_folds)
        ]

    return train_df

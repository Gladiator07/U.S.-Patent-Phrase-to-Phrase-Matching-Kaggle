import datetime
import importlib
import json
import os
import subprocess
import time
import warnings

import hydra
import numpy as np
import pandas as pd
import pytz
import torch
import wandb
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from omegaconf import OmegaConf
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from cocolm.configuration_cocolm import COCOLMConfig
from cocolm.tokenization_cocolm import COCOLMTokenizer
from data.dataset import (
    create_folds,
    prepare_data,
    tokenize_func,
    tokenize_func_transformer_head,
)
from extensions.callbacks import SaveBestModelCallback
from extensions.logger import setup_logger
from extensions.scoring import Pearsonr, calculate_cv_score, post_process_predictions
from extensions.trainer import CustomLossTrainer
from utils import (
    SlackNotifier,
    asHours,
    delete_checkpoints,
    process_hydra_config,
    seed_everything,
)


@hydra.main(config_path="../config", config_name="config")
def run(cfg: OmegaConf):

    if cfg.run.disable_warnings:
        warnings.filterwarnings("ignore")

    run_start_time = time.time()

    # change experiment name for making `out_dir` correctly
    # example: microsoft/deberta-base -> microsoft-deberta-base
    if "/" in cfg.run.experiment_name:
        cfg.run.experiment_name = cfg.run.experiment_name.replace("/", "-")

    # determine whether to train transformer head type model or not
    # if True then tokenization is max_length and dynamic padding is disabled
    if "TransformerHead" in cfg.model.class_name:
        transformer_head_trainer = True
        cfg.trainer.group_by_length = False  # disable uniform length batching
    else:
        transformer_head_trainer = False

    # create output dir
    os.makedirs(cfg.paths.out_dir, exist_ok=True)

    # setup logger
    logger = setup_logger(file_path=cfg.paths.log_file)

    # Load environment variables
    load_dotenv(dotenv_path=cfg.paths.env_file)
    os.environ["WANDB_API_KEY"]
    os.environ["WANDB_SILENT"]
    os.environ["TOKENIZERS_PARALLELISM"]

    # Seed everything
    seed_everything(seed=cfg.trainer.seed)

    # debug mode modifications
    if cfg.run.debug:
        logger.warn(
            "-> Running in debug model. Weights & Biases logging, Artifacts logging to Google Cloud & Slack notifier is disabled\n"
        )
        cfg.wandb.enabled = False
        cfg.trainer.report_to = "none"
        cfg.trainer.num_train_epochs = 2
        cfg.run.upload_artifacts_to_gcs_bucket = False
        cfg.wandb.log_artifacts = False
        cfg.run.slack_notify = False
        cfg.run.print_model_arch = True

    start_msg = (
        "\n" + f"🚀 Starting experiment {cfg.run.experiment_name}, fold_{cfg.run.fold}"
    )
    if cfg.run.slack_notify:
        sk = SlackNotifier(slack_token=os.environ["SLACK_TOKEN"], channel="#train_logs")
        sk.notify(start_msg)

    # log experiment_name and date, time
    current_date_time = datetime.datetime.now(pytz.timezone("Asia/Kolkata")).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    logger.info(f"\n-> Experiment name: {cfg.run.experiment_name}")
    logger.info(f"-> Date, Time: {current_date_time}\n")
    logger.info(f"\n-> Using GPU: {torch.cuda.get_device_name(0)}\n")
    if transformer_head_trainer:
        logger.info(
            ">>>> Using Transformer Head, padding to max_length and dynamic padding is disabled\n"
        )

    if cfg.wandb.enabled:
        print("-> Weights & Biases logging is enabled")
        # wandb init
        wandb.init(
            config=cfg,
            project=cfg.wandb.project,
            group=cfg.run.experiment_name,
            name=cfg.wandb.name,
            notes=cfg.wandb.notes,
        )

    # print and log configuration
    logger.info("*" * 100)
    logger.info(start_msg + " with following configuration\n")
    logger.info(OmegaConf.to_yaml(process_hydra_config(cfg)))
    logger.info("*" * 100)
    logger.info(f"\n-> Saving outputs to {cfg.paths.out_dir}\n")

    # read data
    train_df = pd.read_csv(cfg.paths.train)

    # tokenizer
    if "cocolm" in cfg.model.model_name:
        tokenizer = COCOLMTokenizer.from_pretrained(cfg.model.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    # prepare data
    train_df = prepare_data(
        df=train_df,
        tokenizer=tokenizer,
        cpc_scheme_xml_dir=cfg.paths.cpc_scheme_xml_202105,
        cpc_title_list_dir=cfg.paths.cpc_title_list_202202,
        use_custom_seperator=cfg.data.use_custom_seperator,
    )

    train_df = create_folds(
        train_df=train_df, n_folds=cfg.run.n_folds, seed=cfg.trainer.seed
    )
    # sample subset of data for debugging if debug flag set to True
    if cfg.run.debug:
        train_df = train_df.sample(1000, random_state=cfg.trainer.seed)

    if cfg.data.use_custom_seperator:
        logger.info(f"\nUsing custom seperator -> [s] ")

    logger.info(f"\nLoss set to -> {cfg.model.loss_type}")

    # tokenize and split data
    ds = Dataset.from_pandas(train_df).rename_column("score", "label")

    if transformer_head_trainer:
        # max_length padding
        ds = ds.map(
            lambda x: tokenize_func_transformer_head(
                x, tokenizer=tokenizer, max_length=133
            ),
            batched=True,
        )

    else:
        ds = ds.map(
            lambda x: tokenize_func(
                x,
                tokenizer=tokenizer,
            ),
            batched=True,
        )

    dds = DatasetDict(
        {
            "train": ds.filter(lambda x: x["fold"] != cfg.run.fold),
            "val": ds.filter(lambda x: x["fold"] == cfg.run.fold),
        }
    )

    print("\n")
    print(dds["train"][0])
    print("\n")

    logger.info("\nSample tokenized text:")
    logger.info(
        tokenizer.decode(tokenizer.encode(dds["train"][0]["input_text"])) + "\n"
    )

    # init config and model
    if "cocolm" in cfg.model.model_name:
        config = COCOLMConfig.from_pretrained(cfg.model.model_name)
    else:
        config = AutoConfig.from_pretrained(cfg.model.model_name, num_labels=1)

    # gets the appropiate class of the model defined in `cfg.model.class_name` in hydra configuration
    # from `src.modeling.models` package
    if "AutoModel" in cfg.model.class_name:
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.model_name, config=config
        )

    # """
    # Model class name should contain `*General`
    # This class supports
    # - different losses (mse, bce, pearson) with appropiate post-process
    # - multi_sample dropout
    # - attention_pool
    # """

    elif "General" in cfg.model.class_name:
        MODEL_CLASS = getattr(importlib.import_module("modeling"), cfg.model.class_name)
        model = MODEL_CLASS.from_pretrained(
            cfg.model.model_name,
            config=config,
            loss_type=cfg.model.loss_type,
            multi_sample_dropout=cfg.model.multi_sample_dropout,
            attention_pool=cfg.model.attention_pool,
        )
    else:
        MODEL_CLASS = getattr(importlib.import_module("modeling"), cfg.model.class_name)
        model = MODEL_CLASS.from_pretrained(
            cfg.model.model_name, config=config, loss_type=cfg.model.loss_type
        )

    # check if configuration is correctly loaded
    assert type(config) is model.config_class

    # print model (in debug mode)
    if cfg.run.print_model_arch:
        print(model)

    # training args
    trainer_args = TrainingArguments(**cfg.trainer)

    # data collator (applies dynamic padding as inputs aren't padded while tokenization)
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if cfg.trainer.fp16 else None
    )
    if "AutoModel" in cfg.model.class_name:
        trainer = CustomLossTrainer(
            model,
            trainer_args,
            train_dataset=dds["train"],
            eval_dataset=dds["val"],
            data_collator=None if transformer_head_trainer else data_collator,
            tokenizer=tokenizer,
            compute_metrics=Pearsonr(loss_type=cfg.model.loss_type),
            callbacks=[SaveBestModelCallback()],
            loss_type=cfg.model.loss_type,
        )
    else:
        trainer = Trainer(
            model,
            trainer_args,
            train_dataset=dds["train"],
            eval_dataset=dds["val"],
            data_collator=None if transformer_head_trainer else data_collator,
            tokenizer=tokenizer,
            compute_metrics=Pearsonr(loss_type=cfg.model.loss_type),
            callbacks=[SaveBestModelCallback()],
        )

    # train model
    trainer.train()

    # delete unecessary checkpoints
    delete_checkpoints(cfg.paths.out_dir)

    # save best model
    trainer.save_model(cfg.paths.out_dir)

    # predict on val set with best model loaded at end
    logits, _, metrics = trainer.predict(dds["val"])

    ##########################
    # SAVE OOF PREDICTIONS
    ##########################
    val_df = dds["val"].to_pandas()

    oof_df = pd.DataFrame()
    oof_df["id"] = val_df["id"].to_numpy()
    oof_df["score"] = val_df["label"].to_numpy()
    oof_df["preds"] = post_process_predictions(
        logits=logits.reshape(-1), loss_type=cfg.model.loss_type
    )
    oof_df.to_csv(cfg.paths.oof_file, index=False)

    # CV score
    cv_score = np.round(metrics["test_pearsonr"], 5)

    # save run summary to a json file
    elapsed_time = time.time() - run_start_time

    summary_dict = {
        "experiment": cfg.run.experiment_name,
        "cv": cv_score,
        "experiment_time": asHours(elapsed_time),
        "wandb_run": wandb.run.get_url() if cfg.wandb.enabled else None,
    }
    with open(cfg.paths.summary_file, "w") as f:
        json.dump(summary_dict, f, indent=4)

    # save run config yaml file (hydra config)
    with open(cfg.paths.experiment_config_file, "w") as fp:
        OmegaConf.save(process_hydra_config(cfg), fp)

    # upload artifacts and log final metrics to wandb
    if cfg.wandb.enabled:
        # Log config, scores to wandb
        wandb.log({"experiment_time": elapsed_time, "cv": cv_score})
        wandb.save(cfg.paths.experiment_config_file)
        wandb.save(cfg.paths.log_file)
        wandb.save(cfg.paths.summary_file)

    # log experiment time and notify to slack
    total_exp_time = asHours(time.time() - run_start_time)

    finish_statement = f"🎉 Experiment {cfg.run.experiment_name}, fold_{cfg.run.fold} score -> {cv_score} (completed in {total_exp_time})"

    logger.info("\n\n" + finish_statement)

    if cfg.run.slack_notify:
        sk.notify(finish_statement + "\n" + "*" * 40)

    # upload all fold's artifacts to GCS, calculate and log CV score after last fold's run
    if cfg.run.fold == 4:

        # gcloud login
        subprocess.run(
            f"gcloud auth activate-service-account {cfg.run.gc_service_account} --key-file={cfg.paths.gcs_credentials}".split()
        )
        upload_dir = cfg.paths.out_dir.split("/")
        upload_dir = "/".join(upload_dir[:-1])

        # calculate final cv score considering all 5 folds and save final oof dataframe
        fin_cv_score = calculate_cv_score(
            checkpoint_root_dir=upload_dir,
            oof_file_out_path=f"{upload_dir}/{cfg.run.experiment_name}_oof.csv",
            n_folds=cfg.run.n_folds,
        )

        if cfg.wandb.enabled:
            wandb.log({"fin_cv": fin_cv_score})

            # upload to GCS bucket
            # if cfg.run.upload_artifacts_to_gcs_bucket:
            subprocess.run(
                f"gsutil -m cp -r {upload_dir} {cfg.run.gcs_bucket_name}".split()
            )

        if cfg.wandb.enabled and cfg.wandb.log_artifacts:
            # Upload models, oof, etc to wandb
            logger.info("\n-> Logging artifacts to wandb")
            model_artifact = wandb.Artifact(name=cfg.run.experiment_name, type="model")
            model_artifact.add_dir(upload_dir)
            wandb.log_artifact(model_artifact)
            logger.info(
                f"-> Artifacts uploaded to wandb from {upload_dir} successfully"
            )

        if cfg.run.slack_notify:
            sk.notify(
                "=" * 29
                + "\n"
                + f"🎉🔥🎉 Experiment {cfg.run.experiment_name}, CV Score -> {fin_cv_score}"
                + "\n"
                + "=" * 29
            )

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":

    run()

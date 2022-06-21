# U.S. Patent Phrase to Phrase Matching

# Introduction
This repository contains the code that achieved 32nd place in [U.S. Patent Phrase to Phrase Matching](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching). You can see the detailed explanation of the solution [here]().

You can also view all the experiments logs on Weights & Biases dashboard [here](https://wandb.ai/gladiator/USPPPM-Kaggle).

This competition was organized by [USPTO](https://www.uspto.gov/) and [Kaggle](https://www.kaggle.com). The main aim of this competition was to extract relevant information by matching key phrases in patent documents. Determining the semantic similarity between phrases is critically important during the patent search and examination process to determine if an invention has been described before. You can read more about the problem statement on the [overview page](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/overview) of the competition.

# Requirements
You can simply install all the requirements and setup your machine completely to run the code by `install_deps.sh` script in `bash` folder. Before running this script make sure that you include your kaggle key into the script in the respective field.
```shell
$ bash bash/install_deps.sh
```

# Preparing the data
You can download the data by the following command. Make sure to download the data into a folder named as `input`
```shell
$ kaggle datasets download -d atharvaingle/uspppm-data
```

# Setting up the configuration
You can specify your configuration in `config.yaml` (present in `config/` folder) whether to log artifacts to GCP/W&B or not log them at all by boolean flags in the file.

You can run an experiment by writing a bash file as following:

```shell
#!/bin/bash

# sample run file
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v3_base" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="53" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-base" \
    model.class_name="DebertaV2ForSequenceClassificationGeneral" \
    model.loss_type="mse" \
    model.multi_sample_dropout=True \
    model.attention_pool=False \
    run.name="mse-stable-drop-msd" \
    run.comment="mse loss + multi sample dropout with deberta StableDropout"
done

# use following line only while training on jarvislabs.ai
# pause instance programatically after running a series of experiments
python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
```

- You can override any config field from bash file
- You can pick any model class imported in `modeling/__init__.py`. You just have the specify the name in `model.class_name` field of Hydra configuration.
- Loss choices: `mse`, `bce` and `pearson`
- You can choose any trainer configuration you want from various `*.yaml` files in `config/trainer` folder.

# Inference
You can check out the final ensemble inference code submission for this competition in Kaggle notebook [here](https://www.kaggle.com/code/atharvaingle/uspppm-inference-ensemble-hill-climbing/notebook)
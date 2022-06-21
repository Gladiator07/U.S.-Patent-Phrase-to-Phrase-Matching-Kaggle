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
#!/bin/bash

################
# Attention pool
################
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="cocolm_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="206" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/cocolm-large" \
    model.class_name="COCOLMForSequenceClassificationGeneral" \
    model.loss_type="mse" \
    model.multi_sample_dropout=False \
    model.attention_pool=True \
    run.name="mse-attn-pool" \
    run.comment="cocolm large + mse + attn pool"
done

cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="cocolm_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="207" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/cocolm-large" \
    model.class_name="COCOLMForSequenceClassificationGeneral" \
    model.loss_type="pearson" \
    model.multi_sample_dropout=False \
    model.attention_pool=True \
    run.name="pearson-attn-pool" \
    run.comment="cocolm large + pearson + attention pool"
done

python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
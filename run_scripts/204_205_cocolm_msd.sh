#!/bin/bash

################
# MSD
################
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="cocolm_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="204" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/cocolm-large" \
    model.class_name="COCOLMForSequenceClassificationGeneral" \
    model.loss_type="mse" \
    model.multi_sample_dropout=True \
    model.attention_pool=False \
    run.name="mse-msd" \
    run.comment="cocolm large + mse + msd"
done

cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="cocolm_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="205" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/cocolm-large" \
    model.class_name="COCOLMForSequenceClassificationGeneral" \
    model.loss_type="pearson" \
    model.multi_sample_dropout=True \
    model.attention_pool=False \
    run.name="pearson-msd" \
    run.comment="cocolm large + pearson + msd"
done

python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
#!/bin/bash

cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="cocolm_base" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="145" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/cocolm-base" \
    model.class_name="COCOLMForSequenceClassification" \
    model.loss_type="mse" \
    run.name="mse-baseline" \
    run.comment="cocolm base + mse baseline"
done

cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="cocolm_base" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="146" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/cocolm-base" \
    model.class_name="COCOLMForSequenceClassification" \
    model.loss_type="pearson" \
    run.name="pearson-baseline" \
    run.comment="cocolm base + pearson baseline"
done

cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="cocolm_base" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="147" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/cocolm-large" \
    model.class_name="COCOLMForSequenceClassification" \
    model.loss_type="mse" \
    run.name="mse-baseline" \
    run.comment="cocolm large + mse baseline"
done

cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="cocolm_base" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="148" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/cocolm-large" \
    model.class_name="COCOLMForSequenceClassification" \
    model.loss_type="pearson" \
    run.name="pearson-baseline" \
    run.comment="cocolm large + pearson baseline"
done

python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
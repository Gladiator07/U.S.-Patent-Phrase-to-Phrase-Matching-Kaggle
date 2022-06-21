#!/bin/bash

# mse (xlarge)
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v2x_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="149" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v2-xlarge" \
    model.class_name="DebertaV2ForSequenceClassificationGeneral" \
    model.loss_type="mse" \
    run.name="mse" \
    run.comment="deberta v2x large + mse"
done

# pearson (xlarge)
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v2x_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="150" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v2-xlarge" \
    model.class_name="DebertaV2ForSequenceClassificationGeneral" \
    model.loss_type="pearson" \
    run.name="pearson" \
    run.comment="deberta v2x large + pearson"
done

python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
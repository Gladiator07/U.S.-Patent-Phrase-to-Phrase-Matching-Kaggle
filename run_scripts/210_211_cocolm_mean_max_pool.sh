#!/bin/bash

###############################
# Mean max concatenate pool
################################

cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="cocolm_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="210" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/cocolm-large" \
    model.class_name="COCOLMMeanMaxPool" \
    model.loss_type="mse" \
    run.name="mse-mean-max-concatenate-pool" \
    run.comment="cocolm large + mse + mean max pool"
done


cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="cocolm_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="211" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/cocolm-large" \
    model.class_name="COCOLMMeanMaxPool" \
    model.loss_type="pearson" \
    run.name="pearson-mean-max-concatenate-pool" \
    run.comment="cocolm large + pearson + mean max pool"
done

python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
#!/bin/bash

################
# Conv1D pool
################
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="cocolm_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="208" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/cocolm-large" \
    model.class_name="COCOLMConv1DPooling" \
    model.loss_type="mse" \
    run.name="mse-conv1d-pool" \
    run.comment="cocolm large + mse + conv1d pool"
done


cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="cocolm_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="209" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/cocolm-large" \
    model.class_name="COCOLMConv1DPooling" \
    model.loss_type="pearson" \
    run.name="pearson-conv1d-pool" \
    run.comment="cocolm large + pearson + conv1d pool"
done

python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
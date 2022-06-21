#!/bin/bash

# mse
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="nystromformer_512" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="191" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="uw-madison/nystromformer-512" \
    model.class_name="AutoModel" \
    model.loss_type="mse" \
    run.name="mse" \
    run.comment="nystromformer + mse"
done

# bce
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="nystromformer_512" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="192" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="uw-madison/nystromformer-512" \
    model.class_name="AutoModel" \
    model.loss_type="bce" \
    run.name="bce" \
    run.comment="nystromformer + bce"
done

# pearson
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="nystromformer_512" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="193" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="uw-madison/nystromformer-512" \
    model.class_name="AutoModel" \
    model.loss_type="pearson" \
    run.name="pearson" \
    run.comment="nystromformer + pearson"
done
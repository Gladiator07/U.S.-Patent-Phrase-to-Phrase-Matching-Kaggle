#!/bin/bash

# mse
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="bart_base" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="194" \
    trainer.dataloader_num_workers=6 \
    ++trainer.eval_accumulation_steps=20 \
    data.use_custom_seperator=True \
    model.model_name="facebook/bart-base" \
    model.class_name="AutoModel" \
    model.loss_type="mse" \
    run.name="mse" \
    run.comment="bart base + mse"
done

# bce
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="bart_base" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="195" \
    trainer.dataloader_num_workers=6 \
    ++trainer.eval_accumulation_steps=20 \
    data.use_custom_seperator=True \
    model.model_name="facebook/bart-base" \
    model.class_name="AutoModel" \
    model.loss_type="bce" \
    run.name="bce" \
    run.comment="bart base + bce"
done

# pearson
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="bart_base" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="196" \
    trainer.dataloader_num_workers=6 \
    ++trainer.eval_accumulation_steps=20 \
    data.use_custom_seperator=True \
    model.model_name="facebook/bart-base" \
    model.class_name="AutoModel" \
    model.loss_type="pearson" \
    run.name="pearson" \
    run.comment="bart base + pearson"
done
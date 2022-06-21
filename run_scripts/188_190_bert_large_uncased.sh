#!/bin/bash

# mse
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="bert_large_uncased" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="188" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="bert-large-uncased" \
    model.class_name="AutoModel" \
    model.loss_type="mse" \
    run.name="mse" \
    run.comment="bert large + mse"
done

# bce
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="bert_large_uncased" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="189" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="bert-large-uncased" \
    model.class_name="AutoModel" \
    model.loss_type="bce" \
    run.name="bce" \
    run.comment="bert large + bce"
done


# pearson
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="bert_large_uncased" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="190" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="bert-large-uncased" \
    model.class_name="AutoModel" \
    model.loss_type="pearson" \
    run.name="pearson" \
    run.comment="bert large + pearson"
done
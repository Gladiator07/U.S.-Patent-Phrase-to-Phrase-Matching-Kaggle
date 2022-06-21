#!/bin/bash

# mse
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="patentsberta_v3" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="185" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="AI-Growth-Lab/PatentSBERTa_V3" \
    model.class_name="AutoModel" \
    model.loss_type="mse" \
    run.name="mse" \
    run.comment="patentsberta + mse"
done

# bce
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="patentsberta_v3" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="186" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="AI-Growth-Lab/PatentSBERTa_V3" \
    model.class_name="AutoModel" \
    model.loss_type="bce" \
    run.name="bce" \
    run.comment="patentsberta + bce"
done

# pearson
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="patentsberta_v3" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="187" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="AI-Growth-Lab/PatentSBERTa_V3" \
    model.class_name="AutoModel" \
    model.loss_type="pearson" \
    run.name="pearson" \
    run.comment="patentsberta + pearson"
done
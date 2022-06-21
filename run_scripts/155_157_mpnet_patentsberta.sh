#!/bin/bash

# mse
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="mpnet_patentsberta" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="155" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="AI-Growth-Lab/PatentSBERTa" \
    model.class_name="AutoModel" \
    model.loss_type="mse" \
    run.name="mse" \
    run.comment="mpnet patentsberta + mse"
done

# bce
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="mpnet_patentsberta" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="156" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="AI-Growth-Lab/PatentSBERTa" \
    model.class_name="AutoModel" \
    model.loss_type="bce" \
    run.name="bce" \
    run.comment="mpnet patentsberta + bce"
done

# bce
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="mpnet_patentsberta" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="157" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="AI-Growth-Lab/PatentSBERTa" \
    model.class_name="AutoModel" \
    model.loss_type="pearson" \
    run.name="pearson" \
    run.comment="mpnet patentsberta + pearson"
done

python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
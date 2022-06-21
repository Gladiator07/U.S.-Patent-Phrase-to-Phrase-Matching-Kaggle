#!/bin/bash

# mse + base
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="roberta_base" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="158" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="roberta-base" \
    model.class_name="AutoModel" \
    model.loss_type="mse" \
    run.name="mse" \
    run.comment="roberta base + mse"
done


# bce + base
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="roberta_base" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="159" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="roberta-base" \
    model.class_name="AutoModel" \
    model.loss_type="bce" \
    run.name="bce" \
    run.comment="roberta base + bce"
done

# pearson + base
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="roberta_base" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="160" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="roberta-base" \
    model.class_name="AutoModel" \
    model.loss_type="pearson" \
    run.name="pearson" \
    run.comment="roberta base + pearson"
done


# mse + large
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="roberta_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="161" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="roberta-large" \
    model.class_name="AutoModel" \
    model.loss_type="mse" \
    run.name="mse" \
    run.comment="roberta large + mse"
done


# bce + large
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="roberta_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="162" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="roberta-large" \
    model.class_name="AutoModel" \
    model.loss_type="bce" \
    run.name="bce" \
    run.comment="roberta large + bce"
done

# pearson + large
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="roberta_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="163" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="roberta-large" \
    model.class_name="AutoModel" \
    model.loss_type="pearson" \
    run.name="pearson" \
    run.comment="roberta large + pearson"
done

python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
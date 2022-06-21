#!/bin/bash

#####################
# MSD
######################
# mse
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="electra_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="212" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="google/electra-large-discriminator" \
    model.class_name="ElectraForSequenceClassificationGeneral" \
    model.loss_type="mse" \
    model.multi_sample_dropout=True \
    model.attention_pool=False \
    run.name="mse-msd" \
    run.comment="electra large + mse + msd"
done

# bce
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="electra_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="213" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="google/electra-large-discriminator" \
    model.class_name="ElectraForSequenceClassificationGeneral" \
    model.loss_type="bce" \
    model.multi_sample_dropout=True \
    model.attention_pool=False \
    run.name="bce-msd" \
    run.comment="electra large + bce + msd"
done

# pearson
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="electra_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="214" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="google/electra-large-discriminator" \
    model.class_name="ElectraForSequenceClassificationGeneral" \
    model.loss_type="pearson" \
    model.multi_sample_dropout=True \
    model.attention_pool=False \
    run.name="pearson-msd" \
    run.comment="electra large + pearson + msd"
done

##########################
# Attention Pool
##########################
# mse
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="electra_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="215" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="google/electra-large-discriminator" \
    model.class_name="ElectraForSequenceClassificationGeneral" \
    model.loss_type="mse" \
    model.multi_sample_dropout=False \
    model.attention_pool=True \
    run.name="mse-attention-pool" \
    run.comment="electra large + mse + attention pool"
done

# bce
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="electra_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="216" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="google/electra-large-discriminator" \
    model.class_name="ElectraForSequenceClassificationGeneral" \
    model.loss_type="bce" \
    model.multi_sample_dropout=False \
    model.attention_pool=True \
    run.name="bce-attention-pool" \
    run.comment="electra large + bce + attention pool"
done

# pearson
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="electra_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="217" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="google/electra-large-discriminator" \
    model.class_name="ElectraForSequenceClassificationGeneral" \
    model.loss_type="pearson" \
    model.multi_sample_dropout=False \
    model.attention_pool=True \
    run.name="pearson-attention-pool" \
    run.comment="electra large + pearson + attention pool"
done

python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
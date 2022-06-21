#!/bin/bash

#########################
# MSD
########################

# mse + msd
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v2x_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="196" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v2-xlarge" \
    model.class_name="DebertaV2ForSequenceClassificationGeneral" \
    model.loss_type="mse" \
    model.multi_sample_dropout=True \
    model.attention_pool=False \
    run.name="mse-msd" \
    run.comment="deberta v2x large + mse + msd"
done

# pearson + msd
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v2x_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="197" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v2-xlarge" \
    model.class_name="DebertaV2ForSequenceClassificationGeneral" \
    model.loss_type="pearson" \
    model.multi_sample_dropout=True \
    model.attention_pool=False \
    run.name="mse-pearson" \
    run.comment="deberta v2x large + mse + pearson"
done

#############################
# Attention pool
#############################

# mse + attention pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v2x_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="198" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v2-xlarge" \
    model.class_name="DebertaV2ForSequenceClassificationGeneral" \
    model.loss_type="mse" \
    model.multi_sample_dropout=False \
    model.attention_pool=True \
    run.name="mse-attention-pool" \
    run.comment="deberta v2x large + mse + attention pool"
done

# pearson + attention pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v2x_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="199" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v2-xlarge" \
    model.class_name="DebertaV2ForSequenceClassificationGeneral" \
    model.loss_type="pearson" \
    model.multi_sample_dropout=False \
    model.attention_pool=True \
    run.name="pearson-attention-pool" \
    run.comment="deberta v2x large + pearson + attention pool"
done


python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
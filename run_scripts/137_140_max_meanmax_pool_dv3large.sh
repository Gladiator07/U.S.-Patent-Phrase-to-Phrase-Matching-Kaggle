#!/bin/bash

# mse + max pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v3_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="137" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-large" \
    model.class_name="DebertaV2MaxPooling" \
    model.loss_type="mse" \
    run.name="mse-max-pool" \
    run.comment="mse + max pooling"
done

# pearson + max pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v3_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="138" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-large" \
    model.class_name="DebertaV2MeanPoolingLayerNorm" \
    model.loss_type="pearson" \
    run.name="pearson-max-pool" \
    run.comment="pearson + max pooling"
done

#############################################
# mse + max pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v3_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="139" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-large" \
    model.class_name="DebertaMeanMaxPooling" \
    model.loss_type="mse" \
    run.name="mse-mean-max-concatenate-pool" \
    run.comment="mse + mean max concatenate pooling"
done

# pearson + max pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v3_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="140" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-large" \
    model.class_name="DebertaMeanMaxPooling" \
    model.loss_type="pearson" \
    run.name="pearson-mean-max-concatenate-pool" \
    run.comment="pearson + mean max concatenate pooling"
done

python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
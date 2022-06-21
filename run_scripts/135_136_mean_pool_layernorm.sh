#!/bin/bash

# mse + mean pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v3_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="135" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-large" \
    model.class_name="DebertaV2MeanPoolingLayerNorm" \
    model.loss_type="mse" \
    run.name="mse-mean-pool-layernorm" \
    run.comment="mse + mean pooling + layernorm"
done

# pearson + mean pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v3_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="136" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-large" \
    model.class_name="DebertaV2MeanPoolingLayerNorm" \
    model.loss_type="pearson" \
    run.name="pearson-mean-pool-layernorm" \
    run.comment="pearson + mean pooling + layernorm"
done

python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
#!/bin/bash

# MSE + Attention Head
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v3_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="112" \
    trainer.dataloader_num_workers=6 \
    trainer.per_device_train_batch_size=64 \
    trainer.per_device_eval_batch_size=96 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-large" \
    model.class_name="DebertaV2ForSequenceClassificationTransformerHead" \
    model.loss_type="mse" \
    model.multi_sample_dropout=False \
    model.attention_pool=False \
    run.name="mse-transformer-head" \
    run.comment="transformer head + mse loss"
done

cd ~/.
rm -rf us_patent_artifacts/

# Pearson + Attention Head
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v3_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="113" \
    trainer.dataloader_num_workers=6 \
    trainer.per_device_train_batch_size=64 \
    trainer.per_device_eval_batch_size=96 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-large" \
    model.class_name="DebertaV2ForSequenceClassificationTransformerHead" \
    model.loss_type="pearson" \
    model.multi_sample_dropout=False \
    model.attention_pool=False \
    run.name="pearson-transformer-head" \
    run.comment="transformer head + pearson loss"
done

# pause instance
python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
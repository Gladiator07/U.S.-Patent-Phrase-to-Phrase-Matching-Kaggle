#!/bin/bash

cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v3_base" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="95" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-base" \
    model.class_name="DebertaV2ForSequenceClassificationGeneral" \
    model.loss_type="mse" \
    model.multi_sample_dropout=False \
    model.attention_pool=False \
    run.name="sgkf-anchor-stratify-fin-val-strategy-mse-baseline" \
    run.comment="StratifiedGroupKFold + grouped by anchor and stratified by score + seed 42 + final val strategy baseline + mse loss"
done

cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v3_base" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="96" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-base" \
    model.class_name="DebertaV2ForSequenceClassificationGeneral" \
    model.loss_type="bce" \
    model.multi_sample_dropout=False \
    model.attention_pool=False \
    run.name="sgkf-anchor-stratify-fin-val-strategy-bce-baseline" \
    run.comment="StratifiedGroupKFold + grouped by anchor and stratified by score + seed 42 + final val strategy baseline + bce loss"
done

cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v3_base" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="97" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-base" \
    model.class_name="DebertaV2ForSequenceClassificationGeneral" \
    model.loss_type="pearson" \
    model.multi_sample_dropout=False \
    model.attention_pool=False \
    run.name="sgkf-anchor-stratify-fin-val-strategy-pearson-baseline" \
    run.comment="StratifiedGroupKFold + grouped by anchor and stratified by score + seed 42 + final val strategy baseline + pearson loss"
done
# use following line only while training on jarvislabs.ai
# pause instance programatically after running a series of experiments
python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
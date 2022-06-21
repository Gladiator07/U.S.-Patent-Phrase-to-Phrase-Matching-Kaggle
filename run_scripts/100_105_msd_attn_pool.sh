#!/bin/bash

# MSE + MSD
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v3_base" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="100" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-base" \
    model.class_name="DebertaV2ForSequenceClassificationGeneral" \
    model.loss_type="mse" \
    model.multi_sample_dropout=True \
    model.attention_pool=False \
    run.name="mse-msd-stable-dropout" \
    run.comment="mse loss + StratifiedGroupKFold + grouped by anchor and stratified by score + seed 42 + final val strategy baseline + msd stabledropout"
done

# BCE + MSD
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v3_base" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="101" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-base" \
    model.class_name="DebertaV2ForSequenceClassificationGeneral" \
    model.loss_type="bce" \
    model.multi_sample_dropout=True \
    model.attention_pool=False \
    run.name="bce-msd-stable-dropout" \
    run.comment="bce loss + StratifiedGroupKFold + grouped by anchor and stratified by score + seed 42 + final val strategy baseline + msd stabledropout"
done

# PEARSON + MSD
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v3_base" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="102" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-base" \
    model.class_name="DebertaV2ForSequenceClassificationGeneral" \
    model.loss_type="pearson" \
    model.multi_sample_dropout=True \
    model.attention_pool=False \
    run.name="pearson-msd-stable-dropout" \
    run.comment="pearson loss + StratifiedGroupKFold + grouped by anchor and stratified by score + seed 42 + final val strategy baseline + msd stabledropout"
done

cd ~/.
rm -rf us_patent_artifacts/

# MSE + Attention pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v3_base" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="103" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-base" \
    model.class_name="DebertaV2ForSequenceClassificationGeneral" \
    model.loss_type="mse" \
    model.multi_sample_dropout=False \
    model.attention_pool=True \
    run.name="mse-attention-pool" \
    run.comment="pearson loss + StratifiedGroupKFold + grouped by anchor and stratified by score + seed 42 + final val strategy baseline + attention pool"
done

# BCE + Attention pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v3_base" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="104" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-base" \
    model.class_name="DebertaV2ForSequenceClassificationGeneral" \
    model.loss_type="bce" \
    model.multi_sample_dropout=False \
    model.attention_pool=True \
    run.name="bce-attention-pool" \
    run.comment="bce loss + StratifiedGroupKFold + grouped by anchor and stratified by score + seed 42 + final val strategy baseline + attention pool"
done

# Pearson + Attention pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v3_base" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="105" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-base" \
    model.class_name="DebertaV2ForSequenceClassificationGeneral" \
    model.loss_type="pearson" \
    model.multi_sample_dropout=False \
    model.attention_pool=True \
    run.name="pearson-attention-pool" \
    run.comment="pearson loss + StratifiedGroupKFold + grouped by anchor and stratified by score + seed 42 + final val strategy baseline + attention pool"
done


# use following line only while training on jarvislabs.ai
# pause instance programatically after running a series of experiments
python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
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
    run.exp_num="127" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-large" \
    model.class_name="DebertaV2ForSequenceClassificationMeanPooling" \
    model.loss_type="mse" \
    run.name="mse-mean-pool" \
    run.comment="mse + mean pooling"
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
    run.exp_num="128" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-large" \
    model.class_name="DebertaV2ForSequenceClassificationMeanPooling" \
    model.loss_type="pearson" \
    run.name="pearson-mean-pool" \
    run.comment="pearson + mean pooling"
done


########################################################

# mse + conv1d pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v3_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="129" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-large" \
    model.class_name="DebertaV2ForSequenceClassificationConv1DPooling" \
    model.loss_type="mse" \
    run.name="mse-conv1d-pool" \
    run.comment="mse + conv1d pool"
done

# pearson + conv1d pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v3_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="130" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v3-large" \
    model.class_name="DebertaV2ForSequenceClassificationConv1DPooling" \
    model.loss_type="pearson" \
    run.name="pearson-conv1d-pool" \
    run.comment="pearson + conv1d pool"
done


python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
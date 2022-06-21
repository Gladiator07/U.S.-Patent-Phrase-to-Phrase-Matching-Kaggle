#!/bin/bash

#########################
# Conv1D Pooling
########################

# mse + conv1d
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v2x_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="200" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v2-xlarge" \
    model.class_name="DebertaV2ForSequenceClassificationConv1DPooling" \
    model.loss_type="mse" \
    run.name="mse-conv1d-pool" \
    run.comment="deberta v2x large + mse + conv1d pool"
done


# pearson + conv1d
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v2x_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="201" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v2-xlarge" \
    model.class_name="DebertaV2ForSequenceClassificationConv1DPooling" \
    model.loss_type="pearson" \
    run.name="pearson-conv1d-pool" \
    run.comment="deberta v2x large + pearson + conv1d pool"
done


#################################
# Mean-Max Concatenate Pooling
#################################

# mse + mean-max pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v2x_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="202" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v2-xlarge" \
    model.class_name="DebertaMeanMaxPooling" \
    model.loss_type="mse" \
    run.name="mse-mean-max-pool" \
    run.comment="deberta v2x large + mse + mean max pool"
done

# pearson + mean-max pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="deberta_v2x_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="203" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="microsoft/deberta-v2-xlarge" \
    model.class_name="DebertaMeanMaxPooling" \
    model.loss_type="pearson" \
    run.name="pearson-mean-max-pool" \
    run.comment="deberta v2x large + pearson + mean max pool"
done

python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
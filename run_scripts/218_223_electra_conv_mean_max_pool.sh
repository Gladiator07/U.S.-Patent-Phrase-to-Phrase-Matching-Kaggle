#!/bin/bash

#####################
# Conv1D Pooling
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
    run.exp_num="218" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="google/electra-large-discriminator" \
    model.class_name="ElectraConv1DPooling" \
    model.loss_type="mse" \
    run.name="mse-conv1d-pool" \
    run.comment="electra large + mse + conv1d pool"
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
    run.exp_num="219" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="google/electra-large-discriminator" \
    model.class_name="ElectraConv1DPooling" \
    model.loss_type="bce" \
    run.name="bce-conv1d-pool" \
    run.comment="electra large + bce + conv1d pool"
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
    run.exp_num="220" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="google/electra-large-discriminator" \
    model.class_name="ElectraConv1DPooling" \
    model.loss_type="pearson" \
    run.name="pearson-conv1d-pool" \
    run.comment="electra large + pearson + conv1d pool"
done

###########################
# Mean Max Pooling
###########################
# mse
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="electra_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="221" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="google/electra-large-discriminator" \
    model.class_name="ElectraMeanMaxPooling" \
    model.loss_type="mse" \
    run.name="mse-mean-max-pool" \
    run.comment="electra large + mse + mean max pool"
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
    run.exp_num="222" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="google/electra-large-discriminator" \
    model.class_name="ElectraMeanMaxPooling" \
    model.loss_type="bce" \
    run.name="bce-mean-max-pool" \
    run.comment="electra large + bce + mean max pool"
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
    run.exp_num="223" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="google/electra-large-discriminator" \
    model.class_name="ElectraMeanMaxPooling" \
    model.loss_type="pearson" \
    run.name="pearson-mean-max-pool" \
    run.comment="electra large + pearson + mean max pool"
done

python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
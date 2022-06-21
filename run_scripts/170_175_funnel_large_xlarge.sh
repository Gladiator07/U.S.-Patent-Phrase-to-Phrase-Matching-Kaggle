#!/bin/bash

# mse
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="funnel_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="170" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="funnel-transformer/large" \
    model.class_name="AutoModel" \
    model.loss_type="mse" \
    run.name="mse" \
    run.comment="funnel large + mse"
done

# bce
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="funnel_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="171" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="funnel-transformer/large" \
    model.class_name="AutoModel" \
    model.loss_type="bce" \
    run.name="bce" \
    run.comment="funnel large + bce"
done

# pearson
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="funnel_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="172" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="funnel-transformer/large" \
    model.class_name="AutoModel" \
    model.loss_type="pearson" \
    run.name="pearson" \
    run.comment="funnel large + pearson"
done


################
# funnel xlarge
###############

# mse
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="funnel_xlarge" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="173" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="funnel-transformer/xlarge" \
    model.class_name="AutoModel" \
    model.loss_type="mse" \
    run.name="mse" \
    run.comment="funnel xlarge + mse"
done

# bce
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="funnel_xlarge" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="174" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="funnel-transformer/xlarge" \
    model.class_name="AutoModel" \
    model.loss_type="bce" \
    run.name="bce" \
    run.comment="funnel xlarge + bce"
done

# pearson
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="funnel_xlarge" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="175" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="funnel-transformer/xlarge" \
    model.class_name="AutoModel" \
    model.loss_type="pearson" \
    run.name="pearson" \
    run.comment="funnel xlarge + pearson"
done
python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
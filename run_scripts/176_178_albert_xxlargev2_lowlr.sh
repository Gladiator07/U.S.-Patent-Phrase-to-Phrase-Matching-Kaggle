#!/bin/bash

# mse
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="alberta_xxlarge_v2" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="176" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="albert-xxlarge-v2" \
    model.class_name="AutoModel" \
    model.loss_type="mse" \
    run.name="mse-lowlr" \
    run.comment="albert xxlarge v2 + mse + low lr"
done

# bce
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="alberta_xxlarge_v2" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="177" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="albert-xxlarge-v2" \
    model.class_name="AutoModel" \
    model.loss_type="bce" \
    run.name="bce-lowlr" \
    run.comment="albert xxlarge v2 + bce + low lr"
done

# pearson
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="alberta_xxlarge_v2" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="178" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="albert-xxlarge-v2" \
    model.class_name="AutoModel" \
    model.loss_type="pearson" \
    run.name="pearson-lowlr" \
    run.comment="albert xxlarge v2 + pearson + lowlr"
done

python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
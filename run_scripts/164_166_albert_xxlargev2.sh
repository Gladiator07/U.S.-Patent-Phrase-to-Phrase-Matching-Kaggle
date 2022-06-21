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
    run.exp_num="164" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="albert-xxlarge-v2" \
    model.class_name="AutoModel" \
    model.loss_type="mse" \
    run.name="mse" \
    run.comment="albert xxlarge v2 + mse"
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
    run.exp_num="165" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="albert-xxlarge-v2" \
    model.class_name="AutoModel" \
    model.loss_type="bce" \
    run.name="bce" \
    run.comment="albert xxlarge v2 + bce"
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
    run.exp_num="166" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="albert-xxlarge-v2" \
    model.class_name="AutoModel" \
    model.loss_type="pearson" \
    run.name="pearson" \
    run.comment="albert xxlarge v2 + pearson"
done

python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
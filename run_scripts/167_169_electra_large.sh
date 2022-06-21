
#!/bin/bash

# mse
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="electra_large" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="167" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="google/electra-large-discriminator" \
    model.class_name="AutoModel" \
    model.loss_type="mse" \
    run.name="mse" \
    run.comment="electra large + mse"
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
    run.exp_num="168" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="google/electra-large-discriminator" \
    model.class_name="AutoModel" \
    model.loss_type="bce" \
    run.name="bce" \
    run.comment="electra large + bce"
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
    run.exp_num="169" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="google/electra-large-discriminator" \
    model.class_name="AutoModel" \
    model.loss_type="pearson" \
    run.name="pearson" \
    run.comment="electra large + pearson"
done

python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
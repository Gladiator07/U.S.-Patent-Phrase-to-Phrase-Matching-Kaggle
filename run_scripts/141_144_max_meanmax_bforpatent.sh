#!/bin/bash

# mse + max pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="bert_for_patents" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="141" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="anferico/bert-for-patents" \
    model.class_name="BertMaxPooling" \
    model.loss_type="mse" \
    run.name="mse-max-pool" \
    run.comment="mse + max pool"
done

# pearson + max pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="bert_for_patents" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="142" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="anferico/bert-for-patents" \
    model.class_name="BertMaxPooling" \
    model.loss_type="pearson" \
    run.name="pearson-max-pool" \
    run.comment="pearson + max pool"
done


###################################

# mse + mean max pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="bert_for_patents" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="143" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="anferico/bert-for-patents" \
    model.class_name="BertMeanMaxPooling" \
    model.loss_type="mse" \
    run.name="mse-mean-max-concatenate-pool" \
    run.comment="mse + mean max concatenate pool"
done

# pearson + mean max pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="bert_for_patents" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="144" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="anferico/bert-for-patents" \
    model.class_name="BertMeanMaxPooling" \
    model.loss_type="pearson" \
    run.name="pearson-mean-max-concatenate-pool" \
    run.comment="pearson + mean max concatenate pool"
done

python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
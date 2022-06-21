#!/bin/bash

# mse + mean pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="bert_for_patents" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="131" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="anferico/bert-for-patents" \
    model.class_name="BertForSequenceClassificationMeanPooling" \
    model.loss_type="mse" \
    run.name="mse-mean-pool" \
    run.comment="mse + mean pool"
done

# pearson + mean pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="bert_for_patents" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="132" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="anferico/bert-for-patents" \
    model.class_name="BertForSequenceClassificationMeanPooling" \
    model.loss_type="pearson" \
    run.name="pearson-mean-pool" \
    run.comment="pearson + mean pool"
done

#########################################################

# mse + conv1d pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="bert_for_patents" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="133" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="anferico/bert-for-patents" \
    model.class_name="BertForSequenceClassificationConv1DPooling" \
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
    trainer="bert_for_patents" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="134" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="anferico/bert-for-patents" \
    model.class_name="BertForSequenceClassificationConv1DPooling" \
    model.loss_type="pearson" \
    run.name="pearson-conv1d-pool" \
    run.comment="pearson + conv1d pool"
done


python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
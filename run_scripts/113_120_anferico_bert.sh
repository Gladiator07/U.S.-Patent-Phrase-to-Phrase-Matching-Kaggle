#!/bin/bash

# mse baseline
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="bert_for_patents" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="113" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="anferico/bert-for-patents" \
    model.class_name="BertForSequenceClassificationGeneral" \
    model.loss_type="mse" \
    model.multi_sample_dropout=False \
    model.attention_pool=False \
    run.name="mse-baseline" \
    run.comment="bert for patents + mse baseline"
done

# Pearson baseline
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="bert_for_patents" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="114" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="anferico/bert-for-patents" \
    model.class_name="BertForSequenceClassificationGeneral" \
    model.loss_type="pearson" \
    model.multi_sample_dropout=False \
    model.attention_pool=False \
    run.name="pearson-baseline" \
    run.comment="bert for patents + pearson baseline"
done

# mse + msd
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="bert_for_patents" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="115" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="anferico/bert-for-patents" \
    model.class_name="BertForSequenceClassificationGeneral" \
    model.loss_type="mse" \
    model.multi_sample_dropout=True \
    model.attention_pool=False \
    run.name="mse-msd" \
    run.comment="bert for patents + mse + msd"
done

# pearson + msd
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="bert_for_patents" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="116" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="anferico/bert-for-patents" \
    model.class_name="BertForSequenceClassificationGeneral" \
    model.loss_type="pearson" \
    model.multi_sample_dropout=True \
    model.attention_pool=False \
    run.name="pearson-msd" \
    run.comment="bert for patents + pearson + msd"
done


# mse + attention pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="bert_for_patents" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="117" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="anferico/bert-for-patents" \
    model.class_name="BertForSequenceClassificationGeneral" \
    model.loss_type="mse" \
    model.multi_sample_dropout=False \
    model.attention_pool=True \
    run.name="mse-attention-pool" \
    run.comment="bert for patents + mse + attention pool"
done

# pearson + attention pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="bert_for_patents" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="118" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="anferico/bert-for-patents" \
    model.class_name="BertForSequenceClassificationGeneral" \
    model.loss_type="pearson" \
    model.multi_sample_dropout=False \
    model.attention_pool=True \
    run.name="pearson-attention-pool" \
    run.comment="bert for patents + pearson + attention pool"
done

# mse + msd + attention pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="bert_for_patents" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="119" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="anferico/bert-for-patents" \
    model.class_name="BertForSequenceClassificationGeneral" \
    model.loss_type="mse" \
    model.multi_sample_dropout=True \
    model.attention_pool=True \
    run.name="mse-msd-attention-pool" \
    run.comment="bert for patents + mse + msd + attention pool"
done

# pearson + msd + attention pool
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0 1 2 3 4
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="bert_for_patents" \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="120" \
    trainer.dataloader_num_workers=6 \
    data.use_custom_seperator=True \
    model.model_name="anferico/bert-for-patents" \
    model.class_name="BertForSequenceClassificationGeneral" \
    model.loss_type="pearson" \
    model.multi_sample_dropout=True \
    model.attention_pool=True \
    run.name="pearson-msd-attention-pool" \
    run.comment="bert for patents + mse + pearson + attention pool"
done

# use following line only while training on jarvislabs.ai
# pause instance programatically after running a series of experiments
python3 -c "from jarviscloud import jarviscloud; jarviscloud.pause()"
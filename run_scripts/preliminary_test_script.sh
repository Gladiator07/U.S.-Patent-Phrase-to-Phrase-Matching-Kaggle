#!/bin/bash

# mse
cd /home/US-Patent-Matching-Kaggle/src

for fold in 0
do
    python3 train.py \
    paths="jarvislabs" \
    trainer="default_trainer" \
    trainer.save_steps=200 \
    trainer.eval_steps=200 \
    trainer.logging_steps=200 \
    trainer.per_device_train_batch_size=32 \
    trainer.per_device_eval_batch_size=32 \
    run.debug=False \
    run.fold=$fold \
    run.exp_num="-1" \
    trainer.dataloader_num_workers=6 \
    trainer.fp16=True \
    trainer.learning_rate=2e-5 \
    data.use_custom_seperator=True \
    model.model_name="facebook/bart-base" \
    model.class_name="AutoModel" \
    model.loss_type="mse" \
    run.name="testing" \
    run.comment="testing new model"
done
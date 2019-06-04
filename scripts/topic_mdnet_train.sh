#!/usr/bin/env bash
logdir='checkpoints/report/TopicMDNet'
mkdir -v $logdir
CUDA_VISIBLE_DEVICES=${device} python topic_mdnet_train.py \
        --dataset_dir='data/report' \
        --batch_size=32 \
        --log_dir=$logdir \
        --init_learning_rate=0.001 \
        --finetune_after=5 \
        --epoch=20 \
        --lr_decay_rate=0.9 \
        --decay_every_iter=1 \
        --model='TopicMDNet8' \
        | tee -a $logdir/'log.txt'

#!/usr/bin/env bash
logdir='/home/zizhao/work2/mdnet_checkpoints/report/checkpoint_mdnet6_ftafter10'
CUDA_VISIBLE_DEVICES=${device} python mdnet_train.py \
        --dataset_dir='/home/zizhao/work2/dataset/bladder/report' \
        --batch_size=32 \
        --log_dir=$logdir \
        --init_learning_rate=0.001 \
        --finetune_after=5 \
        --epoch=15 \
        --lr_decay_rate=0.9 \
        --decay_every_iter=1 \
        --model='MDNet6' \
        | tee $logdir/'log.txt'
        #--load_from_checkpoint $logdir/'model5.5_BEST(0.73)-2210' \

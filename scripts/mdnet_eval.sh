#!/usr/bin/env bash
log_dir='/home/zizhao/work2/mdnet_checkpoints/report/checkpoint_TopicMDNet8v2_ftat5/'

CUDA_VISIBLE_DEVICES=${device} python mdnet_test.py \
        --dataset_dir='/home/zizhao/work2/dataset/bladder/report' \
        --log_dir=${log_dir} \
        --load_from_checkpoint=${log_dir}"model7.0_BEST(0.74)-2813" \
        --model='MDNet6' \
        --sample_max=True \
        --test_mode=True \
        --batch_size 50 \
        | tee $logdir'test_log.txt'
